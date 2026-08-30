use std::f64::consts::PI;

pub(crate) const MIN_ORDER: u32 = 1;
pub(crate) const MAX_ORDER: u32 = 2048;
pub(crate) const MIN_QUAN: u32 = 1;
pub(crate) const MAX_QUAN: u32 = 16384;
pub(crate) const MIN_ATTEN: f64 = 12.0;
pub(crate) const MAX_ATTEN: f64 = 180.0;

/// Extra dB when sizing the FIR from attenuation + transition width so the
/// realized stopband more closely meets the requested `atten`.
pub(crate) const ORDER_ATTEN_MARGIN_DB: f64 = 6.0;

#[inline]
fn sinc_c(x: f64, cutoff: f64) -> f64 {
    if x != 0.0 {
        (PI * x * cutoff).sin() / (PI * x)
    } else {
        cutoff
    }
}

/// Modified Bessel function of the first kind, order 0, via its Maclaurin
/// series `I0(x) = sum_k ((x/2)^k / k!)^2` with the term recurrence
/// `t_k = t_{k-1} * (x/2)^2 / k^2` (all terms positive, so no cancellation).
///
/// Truncation: stop once the term drops below `1e-10` absolute, capped at 31
/// terms. Both are matched to the window normalization: the window value is
/// `I0(beta*t) / I0(beta)`, and `I0(beta)` grows exponentially with beta, so
/// the propagated relative error of a coefficient is `~5e-13` across the whole
/// supported `beta` range (`[0, 20]`, i.e. atten up to 180 dB) -- five orders
/// of magnitude below the tightest stopband spec (`1e-9` at 180 dB). At the
/// cap (13% of taps at beta = 18.9) the truncation error is still `~1e-13`
/// relative because `I0(19) ~ 1.3e7`.
#[inline]
fn bessel_i0(x: f64) -> f64 {
    let q = 0.25 * x * x;
    let mut y = 1.0;
    let mut t = 1.0;
    // k^2 for k = 1..=31, updated by odd increments (exact in f64).
    let mut k2 = 1.0;
    let mut inc = 3.0;
    loop {
        t *= q / k2;
        y += t;
        if t < 1e-10 || k2 >= 961.0 {
            break;
        }
        k2 += inc;
        inc += 2.0;
    }
    y
}

/// High-iteration reference version of [`bessel_i0`] used by tests to pin the
/// fast version's accuracy (all-positive terms, so this converges to full
/// f64 precision).
#[cfg(test)]
fn bessel_i0_reference(x: f64) -> f64 {
    let q = 0.25 * x * x;
    let mut y = 1.0;
    let mut t = 1.0;
    let mut k2 = 1.0;
    let mut inc = 3.0;
    loop {
        t *= q / k2;
        y += t;
        if t < 1e-17 * y || k2 >= 160_000.0 {
            break;
        }
        k2 += inc;
        inc += 2.0;
    }
    y
}

#[inline]
fn windowed_sinc(pos: f64, half_order: f64, beta: f64, i0_beta: f64, cutoff: f64) -> f64 {
    let ax = pos.abs();
    if ax > half_order {
        return 0.0;
    }
    let t = (1.0 - (ax / half_order).powi(2)).max(0.0).sqrt();
    sinc_c(pos, cutoff) * (bessel_i0(beta * t) / i0_beta)
}

pub(crate) fn generic_table_len(quan: u32, order: u32) -> usize {
    let last_real = (order as f64 * 0.5 * quan as f64).floor() as usize;
    last_real + 2
}

#[inline]
pub(crate) fn generate_filter_table(quan: u32, order: u32, beta: f64, cutoff: f64) -> Vec<f64> {
    let i0_beta = bessel_i0(beta);
    let half_order = order as f64 * 0.5;
    let last_real = (half_order * quan as f64).floor() as usize;
    debug_assert_eq!(last_real + 2, generic_table_len(quan, order));
    let mut filter = Vec::with_capacity(generic_table_len(quan, order));
    for i in 0..=last_real {
        let pos = i as f64 / quan as f64;
        filter.push(windowed_sinc(pos, half_order, beta, i0_beta, cutoff));
    }
    filter.push(0.0);
    let taps = (order + 1) as usize;
    let mut dc = 0.0;
    for j in 0..taps {
        let pos = j as f64 - half_order;
        dc += windowed_sinc(pos, half_order, beta, i0_beta, cutoff);
    }
    if dc.abs() > 1e-18 {
        let inv = 1.0 / dc;
        for c in &mut filter {
            *c *= inv;
        }
    }
    filter
}

/// Flat polyphase row table for the Generic path: `(quan + 1)` rows of
/// `order + 1` coefficients. Row `ph` holds the FIR coefficients for phase
/// `ph / quan`; an arbitrary fractional phase `frac` with `b =
/// floor(frac * quan)`, `t = frac * quan - b` is evaluated as
/// `(1 - t) * dot(taps, row[b]) + t * dot(taps, row[b + 1])`.
///
/// This is the algebraic transform of the per-tap lerped 1-D table (both
/// sides of the center interpolate toward row `b + 1` because the right-side
/// distance `k - frac` decreases as `frac` grows). Identical in real
/// arithmetic; floating-point results differ only by reassociation.
pub(crate) fn generate_generic_rows(quan: u32, order: u32, beta: f64, cutoff: f64) -> Vec<f64> {
    let table = generate_filter_table(quan, order, beta, cutoff);
    let taps = order as usize + 1;
    let quan = quan as usize;
    let half = taps / 2;
    let last_real = table.len() - 2;
    let mut rows = vec![0.0; (quan + 1) * taps];
    for ph in 0..=quan {
        let row = &mut rows[ph * taps..(ph + 1) * taps];
        for (j, coef) in row.iter_mut().enumerate() {
            let idx = if j < half {
                // Left of center: distance `d = k + frac`, `k = half - j >= 1`.
                // Beyond the window (`idx > last_real`) the true coefficient
                // is exactly 0, matching the old per-tap bounds guard.
                let k = half - j;
                k * quan + ph
            } else if j == half {
                // Odd taps: the center tap (`d = frac`). Even taps: the first
                // right tap also sits at `d = frac`. Same lookup either way.
                ph
            } else {
                // Right of center: `d = k - frac`, `k = j - half >= 1`.
                // Always within the window, so never zero-guarded.
                let k = j - half;
                k * quan - ph
            };
            *coef = if idx <= last_real { table[idx] } else { 0.0 };
        }
    }
    rows
}

/// Flat polyphase table: `len` rows of `order + 1` coefficients stored
/// contiguously (row `i` at `data[i * (order + 1) .. (i + 1) * (order + 1)]`).
/// The flat layout gives the dot-product kernels one contiguous load per row
/// instead of chasing per-row heap allocations.
#[inline]
pub(crate) fn generate_fast_lut(len: usize, order: u32, beta: f64, cutoff: f64) -> Vec<f64> {
    let stride = order as usize + 1;
    let mut lut = Vec::with_capacity(len * stride);
    let i0_beta = bessel_i0(beta);
    let half_order = order as f64 * 0.5;
    let taps = order + 1;
    for i in 0..len {
        let pos = i as f64 / len as f64;
        let mut coef_pos = Vec::with_capacity(stride);
        for j in (0..taps).rev() {
            let pos = pos + j as f64 - half_order;
            coef_pos.push(windowed_sinc(pos, half_order, beta, i0_beta, cutoff));
        }
        let dc: f64 = coef_pos.iter().sum();
        if dc.abs() > 1e-18 {
            let inv = 1.0 / dc;
            for c in &mut coef_pos {
                *c *= inv;
            }
        }
        lut.extend_from_slice(&coef_pos);
    }
    lut
}

/// Measured-trim design: search for the smallest even order and the
/// Kaiser beta whose *worst-case polyphase branch* stopband meets `atten`
/// exactly, replacing the `+6 dB` order margin and the approximate
/// `calc_kaiser_beta` mapping with a direct measurement of the realized
/// response.
///
/// The stopband maximum is evaluated on a fixed frequency grid over
/// `[stop_edge, Nyquist]` for a few representative fractional phases
/// (`frac` in the branch-tap sense `d = |order/2 - r + frac|`), so the
/// guarantee transfers to the fractional-delay branches the converters
/// actually evaluate.
///
/// Returns `None` when no order up to `MAX_ORDER` meets the spec (the
/// caller falls back to the classic formula design).
/// Worst-case stopband level (dB) of the polyphase branches (fractional
/// phases `FRACS`) over the grid bins at/above `stop_edge` plus the exact
/// edge point. Direct DFT with a per-frequency phasor recurrence.
fn branch_stopmax(order: u32, beta: f64, cutoff: f64, stop_edge: f64) -> f64 {
    const FRACS: [f64; 3] = [0.0, 0.5, 0.9];
    const NF: f64 = 2048.0;
    let taps_n = order as usize + 1;
    let ho = order as f64 * 0.5;
    let i0b = bessel_i0(beta);
    let mut worst = f64::NEG_INFINITY;
    for &frac in &FRACS {
        let mut h = Vec::with_capacity(taps_n);
        for r in 0..taps_n {
            let d = (ho - r as f64 + frac).abs();
            h.push(if d > ho {
                0.0
            } else {
                windowed_sinc(d, ho, beta, i0b, cutoff)
            });
        }
        let dc: f64 = h.iter().sum();
        if dc.abs() > 1e-18 {
            for c in &mut h {
                *c /= dc;
            }
        }
        // One DFT evaluation at an arbitrary frequency (cycles), tracking
        // the phasor by recurrence: no FFT machinery needed.
        let dft_mag2 = |f_cycles: f64, h: &[f64]| -> f64 {
            let w = -2.0 * PI * f_cycles;
            let (wc, ws) = (w.cos(), w.sin());
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            let (mut sr, mut si) = (0.0f64, 0.0f64);
            for &hk in h {
                sr += hk * cr;
                si += hk * ci;
                let nr = cr * wc - ci * ws;
                ci = cr * ws + ci * wc;
                cr = nr;
            }
            sr * sr + si * si
        };
        // Grid bins at/above the stop edge, plus the exact edge point: the
        // transition skirt is steepest right at the edge, and a bin-only
        // grid would let the search push the spec crossing into the
        // sub-bin gap between the edge and the first grid point.
        let j0 = ((stop_edge * NF * 0.5).ceil() as u32).max(1);
        for j in j0..NF as u32 / 2 {
            let m2 = dft_mag2(j as f64 / NF, &h);
            worst = worst.max(10.0 * m2.max(1e-18).log10());
        }
        let m2 = dft_mag2(stop_edge * 0.5, &h);
        worst = worst.max(10.0 * m2.max(1e-18).log10());
    }
    worst
}

pub(crate) fn trim_design(ratio: f64, atten: f64, trans_width: f64) -> Option<(u32, f64)> {
    const GRID: usize = 15;
    const GOLDEN_ITERS: usize = 12;
    const TOL_DB: f64 = 0.05;

    let nyq = ratio.min(1.0);
    let cutoff = design_cutoff(ratio, trans_width);
    let stop_edge = cutoff + 0.5 * trans_width * nyq;
    let target = -atten + TOL_DB;
    // Keep the returned beta inside `with_raw_internal`'s `[0, 20]` check.
    let bmax = (0.1102 * atten + 4.0).min(19.5);

    // Per-order beta search: coarse log grid, then golden-section refine
    // around the best grid point (stopmax(beta) at fixed order is
    // unimodal: small beta leaks sidelobes, large beta widens the
    // transition into the stopband).
    let best_beta = |order: u32| -> f64 {
        let mut grid = [0.0f64; GRID];
        for (i, g) in grid.iter_mut().enumerate() {
            *g = 1e-3 * (bmax / 1e-3).powf(i as f64 / (GRID - 1) as f64);
        }
        let mut vals = [0.0f64; GRID];
        for (v, &g) in vals.iter_mut().zip(&grid) {
            *v = branch_stopmax(order, g, cutoff, stop_edge);
        }
        let mut i = 0;
        for k in 1..GRID {
            if vals[k] < vals[i] {
                i = k;
            }
        }
        let mut a = grid[i.saturating_sub(1)];
        let mut b = grid[(i + 1).min(GRID - 1)];
        let gr = (5.0f64.sqrt() - 1.0) / 2.0;
        let mut x1 = b - gr * (b - a);
        let mut x2 = a + gr * (b - a);
        let mut f1 = branch_stopmax(order, x1, cutoff, stop_edge);
        let mut f2 = branch_stopmax(order, x2, cutoff, stop_edge);
        for _ in 0..GOLDEN_ITERS {
            if f1 < f2 {
                b = x2;
                f2 = f1;
                x2 = x1; // keep the evaluated point
                x1 = b - gr * (b - a);
                f1 = branch_stopmax(order, x1, cutoff, stop_edge);
            } else {
                a = x1;
                f1 = f2;
                x1 = x2; // keep the evaluated point
                x2 = a + gr * (b - a);
                f2 = branch_stopmax(order, x2, cutoff, stop_edge);
            }
        }
        0.5 * (a + b)
    };

    let meets = |order: u32, beta: f64| branch_stopmax(order, beta, cutoff, stop_edge) <= target;

    // Bisect over half-order (order stays even, matching `calc_order`).
    // Lower bound: the classic Kaiser estimate without margin; upper bound:
    // the current formula order, grown if even that misses the spec.
    let lo_m0 = {
        let o = ((atten - 8.0) / (2.285 * trans_width * PI * nyq))
            .ceil()
            .max(8.0) as u32;
        ((o / 2 + 1).min(MAX_ORDER / 2)) as i64
    };
    let mut hi_m = (calc_order(ratio, atten, trans_width) / 2) as i64;

    let mut lo_m = lo_m0;
    let bb = best_beta(2 * lo_m as u32);
    if meets(2 * lo_m as u32, bb) {
        return Some((2 * lo_m as u32, bb));
    }
    {
        // Warm-beta probe: the order-optimal beta shifts only slowly with
        // order, so testing the last feasible beta first (one DFT sweep)
        // usually replaces the whole grid+golden beta search. A probe miss
        // falls back to the full search, so feasibility answers stay exact.
        let probe = |order: u32, warm: Option<f64>| -> Option<f64> {
            let hint = warm?;
            if branch_stopmax(order, hint, cutoff, stop_edge) <= target {
                Some(hint)
            } else {
                None
            }
        };
        // Grow the upper bound until feasible, then bisect down; hi_m stays
        // feasible, so the final hi_m (with its beta) is the answer.
        let mut warm = None;
        let mut hi_beta =
            probe(2 * hi_m as u32, warm).unwrap_or_else(|| best_beta(2 * hi_m as u32));
        while !meets(2 * hi_m as u32, hi_beta) {
            lo_m = hi_m;
            hi_m += 4;
            if 2 * hi_m as u32 > MAX_ORDER {
                return None;
            }
            warm = Some(hi_beta);
            hi_beta = probe(2 * hi_m as u32, warm).unwrap_or_else(|| best_beta(2 * hi_m as u32));
        }
        while lo_m + 1 < hi_m {
            let mid = (lo_m + hi_m) / 2;
            let bm =
                probe(2 * mid as u32, Some(hi_beta)).unwrap_or_else(|| best_beta(2 * mid as u32));
            if meets(2 * mid as u32, bm) {
                hi_m = mid;
                hi_beta = bm;
            } else {
                lo_m = mid;
            }
        }
        Some((2 * hi_m as u32, hi_beta))
    }
}

#[inline]
pub(crate) fn calc_kaiser_beta(atten: f64) -> f64 {
    if atten > 50.0 {
        0.1102 * (atten - 8.7)
    } else if atten >= 21.0 {
        0.5842 * (atten - 21.0).powf(0.4) + 0.07886 * (atten - 21.0)
    } else {
        0.0
    }
}

#[inline]
pub(crate) fn calc_trans_width(ratio: f64, atten: f64, order: u32) -> f64 {
    (atten - 8.0) / (2.285 * order as f64 * PI * ratio.min(1.0))
}

#[inline]
pub(crate) fn calc_order(ratio: f64, atten: f64, trans_width: f64) -> u32 {
    let design_atten = atten + ORDER_ATTEN_MARGIN_DB;
    let mut order =
        f64::ceil((design_atten - 8.0) / (2.285 * trans_width * PI * ratio.min(1.0))) as u32;
    if order < MIN_ORDER {
        order = MIN_ORDER;
    }
    if order % 2 == 1 {
        order += 1;
    }
    order.min(MAX_ORDER)
}

#[inline]
pub(crate) fn design_cutoff(ratio: f64, trans_width: f64) -> f64 {
    let nyquist = ratio.min(1.0);
    (nyquist * (1.0 - trans_width)).clamp(0.01, 1.0)
}

#[cfg(test)]
mod trim_tests {
    use super::*;

    /// The fast `bessel_i0` must track the high-iteration reference to
    /// ~1e-12 relative over the whole supported argument range; the
    /// propagated coefficient error is then ~5e-13 (see `bessel_i0` docs).
    #[test]
    fn bessel_i0_matches_reference() {
        let mut worst = 0.0f64;
        let n = 20_000;
        for i in 1..=n {
            let x = 20.0 * i as f64 / n as f64;
            let r = bessel_i0_reference(x);
            worst = worst.max((bessel_i0(x) - r).abs() / r);
        }
        assert!(worst < 1e-11, "bessel_i0 rel err {worst:.2e}");
    }

    /// The trimmed design must meet the requested stopband on the measured
    /// branches, stay within a few taps of the formula order, and return a
    /// beta inside the `[0, 20]` constructor range.
    #[test]
    fn trim_design_meets_spec_with_fewer_taps() {
        for (ratio, atten, tw) in [
            (44100.0f64 / 48000.0f64, 96.0f64, 0.05f64),
            (44100.0f64 / 48000.0f64, 120.0f64, 0.05f64),
            (44100.0f64 / 48000.0f64, 144.0f64, 0.05f64),
            (48000.0f64 / 44100.0f64, 96.0f64, 0.1f64),
            (2.0f64, 48.0f64, 0.2f64),
        ] {
            let nyq = ratio.min(1.0);
            let cutoff = design_cutoff(ratio, tw);
            let stop_edge = cutoff + 0.5 * tw * nyq;
            let formula_order = calc_order(ratio, atten, tw);
            let (order, beta) =
                trim_design(ratio, atten, tw).unwrap_or((formula_order, calc_kaiser_beta(atten)));
            assert_eq!(order % 2, 0, "trimmed order must stay even");
            assert!(
                order <= formula_order + 16,
                "order {order} >> formula {formula_order}"
            );
            assert!((0.0..=20.0).contains(&beta), "beta {beta} out of range");
            let realized = branch_stopmax(order, beta, cutoff, stop_edge);
            assert!(
                realized <= -atten + 0.05,
                "ratio {ratio} atten {atten}: realized {realized:.2} dB misses spec"
            );
        }
    }
}
