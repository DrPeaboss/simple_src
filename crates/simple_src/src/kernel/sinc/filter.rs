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

#[inline]
fn bessel_i0(x: f64) -> f64 {
    let mut y = 1.0;
    let mut t = 1.0;
    for k in 1..32 {
        t *= (x / (2.0 * k as f64)).powi(2);
        y += t;
        if t < 1e-10 {
            break;
        }
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
