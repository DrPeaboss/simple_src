//! f64 dot-product kernels for the sinc Fast (polyphase) and Generic paths.
//!
//! One hot kernel shape: `dot(tap, row)` over equal-length slices. On x86_64
//! an AVX2+FMA variant (four f64x4 FMA accumulators, unaligned loads, scalar
//! tail) is selected at runtime; on aarch64 a NEON variant (four f64x2 FMA
//! accumulators, unaligned loads, scalar tail) is used. Elsewhere (and on
//! CPUs without those features) a portable zip-sum is used, which LLVM
//! auto-vectorizes at the baseline ISA. The kernel is chosen once when a
//! converter is built and stored as a function pointer, so the hot loop never
//! re-checks features.
//!
//! `#[inline(never)]`-style isolation matters here: letting LLVM inline every
//! arm into the caller can merge the loops into an indirect-jump mega-loop
//! (same failure mode that regressed the linear batch path). The AVX2/NEON
//! bodies are `unsafe fn` under `#[target_feature]`, reached only through the
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

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::DotFn;

    /// AVX2+FMA kernel: 4 × f64x4 FMA accumulators, unaligned loads, scalar
    /// tail.
    ///
    /// # Safety
    /// Must only be called on a CPU with `avx2` and `fma` (guaranteed by
    /// [`select`], which is the only place this function is stored into a
    /// [`DotFn`]) — plus direct unit tests guarded by runtime detection.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn dot_avx2(tap: &[f64], row: &[f64]) -> f64 {
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
                acc0 =
                    _mm256_fmadd_pd(_mm256_loadu_pd(tp.add(i)), _mm256_loadu_pd(rp.add(i)), acc0);
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
                acc0 =
                    _mm256_fmadd_pd(_mm256_loadu_pd(tp.add(i)), _mm256_loadu_pd(rp.add(i)), acc0);
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

    /// Pick the best kernel for this CPU. Called once per converter build;
    /// the result is stored as a function pointer so the hot loop stays
    /// branch-free.
    pub(crate) fn select() -> DotFn {
        if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
            // Statically enabled: no runtime check needed.
            dot_avx2
        } else if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            dot_avx2
        } else {
            super::dot_scalar
        }
    }

    #[cfg(test)]
    pub(crate) fn avx2_available() -> bool {
        cfg!(all(target_feature = "avx2", target_feature = "fma"))
            || (std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma"))
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use x86::select as select_dot;

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::DotFn;

    /// NEON kernel: 4 × f64x2 FMA accumulators, unaligned loads, scalar tail.
    ///
    /// # Safety
    /// Must only be called on a CPU with NEON (guaranteed by [`select`],
    /// which is the only place this function is stored into a [`DotFn`]).
    /// NEON is mandatory on aarch64, so this is always safe on that arch.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn dot_neon(tap: &[f64], row: &[f64]) -> f64 {
        use std::arch::aarch64::*;
        debug_assert_eq!(tap.len(), row.len());
        // SAFETY (edition 2024): the intrinsics below require the neon
        // feature guaranteed by the caller contract documented above.
        unsafe {
            let n = tap.len();
            let tp = tap.as_ptr();
            let rp = row.as_ptr();
            let mut acc0 = vdupq_n_f64(0.0);
            let mut acc1 = vdupq_n_f64(0.0);
            let mut acc2 = vdupq_n_f64(0.0);
            let mut acc3 = vdupq_n_f64(0.0);
            let mut i = 0;
            while i + 8 <= n {
                let t0 = vld1q_f64(tp.add(i));
                let r0 = vld1q_f64(rp.add(i));
                acc0 = vfmaq_f64(acc0, t0, r0);

                let t1 = vld1q_f64(tp.add(i + 2));
                let r1 = vld1q_f64(rp.add(i + 2));
                acc1 = vfmaq_f64(acc1, t1, r1);

                let t2 = vld1q_f64(tp.add(i + 4));
                let r2 = vld1q_f64(rp.add(i + 4));
                acc2 = vfmaq_f64(acc2, t2, r2);

                let t3 = vld1q_f64(tp.add(i + 6));
                let r3 = vld1q_f64(rp.add(i + 6));
                acc3 = vfmaq_f64(acc3, t3, r3);

                i += 8;
            }
            while i + 4 <= n {
                let t0 = vld1q_f64(tp.add(i));
                let r0 = vld1q_f64(rp.add(i));
                acc0 = vfmaq_f64(acc0, t0, r0);

                let t1 = vld1q_f64(tp.add(i + 2));
                let r1 = vld1q_f64(rp.add(i + 2));
                acc1 = vfmaq_f64(acc1, t1, r1);

                i += 4;
            }
            while i + 2 <= n {
                let t0 = vld1q_f64(tp.add(i));
                let r0 = vld1q_f64(rp.add(i));
                acc0 = vfmaq_f64(acc0, t0, r0);
                i += 2;
            }
            let sum = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
            let mut total = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);
            while i < n {
                total += tap[i] * row[i];
                i += 1;
            }
            total
        }
    }

    pub(crate) fn select() -> DotFn {
        if cfg!(target_feature = "neon") || std::arch::is_aarch64_feature_detected!("neon") {
            dot_neon
        } else {
            super::dot_scalar
        }
    }

    #[cfg(test)]
    pub(crate) fn neon_available() -> bool {
        cfg!(target_feature = "neon") || std::arch::is_aarch64_feature_detected!("neon")
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) use aarch64::select as select_dot;

/// Non-x86_64/non-aarch64 targets use the portable auto-vectorized fallback.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn select_dot() -> DotFn {
    dot_scalar
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
    fn simd_matches_scalar_within_epsilon() {
        #[cfg(target_arch = "x86_64")]
        if !super::x86::avx2_available() {
            return; // SIMD kernel not callable on this CPU
        }
        #[cfg(target_arch = "aarch64")]
        if !super::aarch64::neon_available() {
            return; // SIMD kernel not callable on this CPU
        }
        for n in [0, 1, 2, 3, 4, 5, 7, 16, 17, 97, 145, 1000] {
            let tap = sample(n, 1);
            let row = sample(n, 2);
            let scalar = dot_scalar(&tap, &row);
            let kahan = kahan(&tap, &row);
            #[cfg(target_arch = "x86_64")]
            if super::x86::avx2_available() {
                // SAFETY: guarded by runtime detection above.
                let avx = unsafe { super::x86::dot_avx2(&tap, &row) };
                let reference = scalar.abs().max(kahan.abs()).max(1.0);
                assert!(
                    (avx - kahan).abs() / reference < 1e-12,
                    "n={n}: avx {avx} vs kahan {kahan}"
                );
            }
            #[cfg(target_arch = "aarch64")]
            if super::aarch64::neon_available() {
                // SAFETY: guarded by runtime detection above.
                let neon = unsafe { super::aarch64::dot_neon(&tap, &row) };
                let reference = scalar.abs().max(kahan.abs()).max(1.0);
                assert!(
                    (neon - kahan).abs() / reference < 1e-12,
                    "n={n}: neon {neon} vs kahan {kahan}"
                );
            }
            let reference = scalar.abs().max(kahan.abs()).max(1.0);
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
