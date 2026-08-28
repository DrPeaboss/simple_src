//! f64 dot-product kernels for the sinc Fast (polyphase) path.
//!
//! One hot kernel shape: `dot(tap, row)` over equal-length slices. The AVX2
//! variant uses four f64x4 FMA accumulators with unaligned loads and a scalar
//! tail; the portable variant is a zip-sum that LLVM auto-vectorizes at the
//! baseline ISA. The kernel is chosen once when a converter is built and
//! stored as a function pointer, so the hot loop never re-checks features.
//!
//! `#[inline(never)]`-style isolation matters here: letting LLVM inline every
//! arm into the caller can merge the loops into an indirect-jump mega-loop
//! (same failure mode that regressed the linear batch path). The AVX2 body is
//! an `unsafe fn` under `#[target_feature]`, reached only through the
//! function pointer selected after runtime detection.

/// Kernel signature: `tap` and `row` have equal lengths (the FIR delay line
/// slices and the matching LUT row segment).
pub(crate) type DotFn = unsafe fn(&[f64], &[f64]) -> f64;

/// Portable kernel. Reads are plain; LLVM auto-vectorizes this to the
/// baseline ISA (SSE2 on x86-64) with multiple accumulators.
#[inline]
pub(crate) fn dot_scalar(tap: &[f64], row: &[f64]) -> f64 {
    debug_assert_eq!(tap.len(), row.len());
    tap.iter().zip(row).map(|(t, r)| t * r).sum()
}

/// AVX2+FMA kernel: 4 × f64x4 FMA accumulators, unaligned loads, scalar tail.
///
/// # Safety
/// Must only be called on a CPU with `avx2` and `fma` (guaranteed by
/// [`select_dot`], which is the only place this function is stored into a
/// [`DotFn`]) — plus direct unit tests guarded by runtime detection.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(tap: &[f64], row: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    debug_assert_eq!(tap.len(), row.len());
    // SAFETY (edition 2024): the intrinsics below require the avx2,fma
    // features guaranteed by the caller contract documented above.
    unsafe {
        let n = tap.len();
        let tp = tap.as_ptr();
        let rp = row.as_ptr();
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut acc2 = _mm256_setzero_pd();
        let mut acc3 = _mm256_setzero_pd();
        let mut i = 0;
        while i + 16 <= n {
            acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(tp.add(i)), _mm256_loadu_pd(rp.add(i)), acc0);
            acc1 = _mm256_fmadd_pd(
                _mm256_loadu_pd(tp.add(i + 4)),
                _mm256_loadu_pd(rp.add(i + 4)),
                acc1,
            );
            acc2 = _mm256_fmadd_pd(
                _mm256_loadu_pd(tp.add(i + 8)),
                _mm256_loadu_pd(rp.add(i + 8)),
                acc2,
            );
            acc3 = _mm256_fmadd_pd(
                _mm256_loadu_pd(tp.add(i + 12)),
                _mm256_loadu_pd(rp.add(i + 12)),
                acc3,
            );
            i += 16;
        }
        while i + 4 <= n {
            acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(tp.add(i)), _mm256_loadu_pd(rp.add(i)), acc0);
            i += 4;
        }
        let sum = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
        let mut lanes = [0.0f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), sum);
        let mut tail = 0.0;
        while i < n {
            tail += tap[i] * row[i];
            i += 1;
        }
        lanes.iter().sum::<f64>() + tail
    }
}

/// Pick the best kernel for this CPU. Called once per converter build; the
/// result is stored as a function pointer so the hot loop stays branch-free.
pub(crate) fn select_dot() -> DotFn {
    if cfg!(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )) {
        // Statically enabled: no runtime check needed.
        dot_avx2
    } else if cfg!(target_arch = "x86_64")
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        dot_avx2
    } else {
        // Portable fallback (also covers aarch64 and other targets for now).
        dot_scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random-ish data with odd/even lengths.
    fn sample(n: usize, salt: usize) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 37 + salt * 91) % 23) as f64 / 7.0 - 1.5)
            .collect()
    }

    fn kahan(a: &[f64], b: &[f64]) -> f64 {
        let (mut s, mut c) = (0.0, 0.0);
        for (x, y) in a.iter().zip(b.iter()) {
            let p = x * y - c;
            let t = s + p;
            c = (t - s) - p;
            s = t;
        }
        s
    }

    #[test]
    fn avx2_matches_scalar_within_epsilon() {
        if !cfg!(all(target_arch = "x86_64", target_feature = "avx2"))
            && !(cfg!(target_arch = "x86_64")
                && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma"))
        {
            return; // AVX2 kernel not callable on this CPU
        }
        for n in [0, 1, 2, 3, 4, 5, 7, 16, 17, 97, 145, 1000] {
            let tap = sample(n, 1);
            let row = sample(n, 2);
            let scalar = dot_scalar(&tap, &row);
            let kahan = kahan(&tap, &row);
            let avx = unsafe { dot_avx2(&tap, &row) };
            let reference = scalar.abs().max(kahan.abs()).max(1.0);
            assert!(
                (avx - kahan).abs() / reference < 1e-12,
                "n={n}: avx {avx} vs kahan {kahan}"
            );
            assert!(
                (scalar - kahan).abs() / reference < 1e-12,
                "n={n}: scalar {scalar} vs kahan {kahan}"
            );
        }
    }

    #[test]
    fn select_dot_returns_a_working_kernel() {
        let dot = select_dot();
        let tap = sample(97, 3);
        let row = sample(97, 4);
        let got = unsafe { dot(&tap, &row) };
        let want = kahan(&tap, &row);
        assert!((got - want).abs() / want.abs().max(1.0) < 1e-12);
    }
}
