//! Small fixed-size FFT helpers for the measured-trim filter design.
//!
//! The trim search evaluates the stopband response on a 2048-bin grid.
//! When the `rustfft` feature is enabled (the default) we use RustFFT's
//! optimized planner; otherwise a compact hand-written radix-2 FFT is used.

#[cfg(any(test, not(feature = "rustfft")))]
use std::f64::consts::PI;

/// Number of bins used by the trim stopband sweep (see `filter.rs`).
pub(super) const FFT_N: usize = 2048;

/// Precomputed radix-2 twiddle factors: `W_N^k = exp(-2πi k / N)`,
/// `k = 0..N/2`.
#[cfg(any(test, not(feature = "rustfft")))]
fn twiddles() -> &'static [(f64, f64)] {
    use std::sync::OnceLock;
    static TWIDDLES: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    TWIDDLES.get_or_init(|| {
        (0..FFT_N / 2)
            .map(|k| {
                let a = -2.0 * PI * k as f64 / FFT_N as f64;
                (a.cos(), a.sin())
            })
            .collect()
    })
}

/// In-place radix-2 decimation-in-time FFT.
///
/// `re`/`im` must have length [`FFT_N`]. This is intentionally small and
/// dependency-free: it only needs to be fast enough for the trim search, not
/// to compete with RustFFT across arbitrary sizes.
#[cfg(any(test, not(feature = "rustfft")))]
pub(super) fn fft_radix2(re: &mut [f64], im: &mut [f64]) {
    debug_assert_eq!(re.len(), FFT_N);
    debug_assert_eq!(im.len(), FFT_N);

    // Bit-reversal permutation.
    let n = FFT_N;
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let tw = twiddles();
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for i in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = tw[k * step];
                let u_re = re[i + k];
                let u_im = im[i + k];
                let v_re = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi;
                let v_im = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + len / 2] = u_re - v_re;
                im[i + k + len / 2] = u_im - v_im;
            }
        }
        len <<= 1;
    }
}

#[cfg(feature = "rustfft")]
mod rustfft_impl {
    use super::FFT_N;
    use rustfft::num_complex::Complex;
    use std::cell::RefCell;
    use std::sync::{Arc, OnceLock};

    /// Lazily built RustFFT plan. The planner cost is paid once per process;
    /// every trim search then reuses the same plan.
    fn plan() -> &'static Arc<dyn rustfft::Fft<f64>> {
        static PLAN: OnceLock<Arc<dyn rustfft::Fft<f64>>> = OnceLock::new();
        PLAN.get_or_init(|| {
            let mut planner = rustfft::FftPlanner::<f64>::new();
            planner.plan_fft_forward(FFT_N)
        })
    }

    /// Allocate a reusable in-place scratch buffer for the current plan.
    pub fn new_scratch() -> Vec<Complex<f64>> {
        vec![Complex::new(0.0, 0.0); plan().get_inplace_scratch_len()]
    }

    /// Run the RustFFT forward transform, copying between the plain
    /// `re`/`im` slices and RustFFT's `Complex` buffer.
    pub fn forward(re: &mut [f64], im: &mut [f64], scratch: &mut [Complex<f64>]) {
        thread_local! {
            static BUF: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
        }
        BUF.with(|b| {
            let mut buf = b.borrow_mut();
            buf.resize(FFT_N, Complex::new(0.0, 0.0));
            for ((c, &r), &i) in buf.iter_mut().zip(re.iter()).zip(im.iter()) {
                *c = Complex::new(r, i);
            }
            plan().process_with_scratch(&mut buf, scratch);
            for (c, (r, i)) in buf.iter().zip(re.iter_mut().zip(im.iter_mut())) {
                *r = c.re;
                *i = c.im;
            }
        });
    }
}

#[cfg(feature = "rustfft")]
pub(super) use rustfft_impl::{forward as rustfft_forward, new_scratch as rustfft_scratch};

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct O(N) DFT for a single bin; used as the reference in tests.
    fn dft_at(re: &[f64], im: &[f64], bin: usize) -> (f64, f64) {
        let n = re.len();
        let w = -2.0 * PI * bin as f64 / n as f64;
        let (wc, ws) = (w.cos(), w.sin());
        let (mut cr, mut ci) = (1.0f64, 0.0f64);
        let (mut sr, mut si) = (0.0f64, 0.0f64);
        for (&r, &i) in re.iter().zip(im) {
            sr += r * cr - i * ci;
            si += r * ci + i * cr;
            let nr = cr * wc - ci * ws;
            ci = cr * ws + ci * wc;
            cr = nr;
        }
        (sr, si)
    }

    fn sample_input() -> (Vec<f64>, Vec<f64>) {
        // Deterministic broadband-ish input.
        let re: Vec<f64> = (0..FFT_N)
            .map(|i| (i as f64 * 0.01).sin() + 0.5 * (i as f64 * 0.037).cos())
            .collect();
        let im: Vec<f64> = (0..FFT_N)
            .map(|i| (i as f64 * 0.013).cos() - 0.25 * (i as f64 * 0.071).sin())
            .collect();
        (re, im)
    }

    #[test]
    fn radix2_matches_direct_dft() {
        let (re0, im0) = sample_input();
        let mut re = re0.clone();
        let mut im = im0.clone();
        fft_radix2(&mut re, &mut im);

        for &bin in &[0usize, 1, 2, 7, 128, 1023] {
            let (er, ei) = dft_at(&re0, &im0, bin);
            assert!(
                (re[bin] - er).abs() < 1e-8 && (im[bin] - ei).abs() < 1e-8,
                "bin {bin}: fft=({}, {}), direct=({}, {})",
                re[bin],
                im[bin],
                er,
                ei
            );
        }
    }

    #[cfg(feature = "rustfft")]
    #[test]
    fn rustfft_matches_handwritten() {
        let (re0, im0) = sample_input();
        let mut re_rust = re0.clone();
        let mut im_rust = im0.clone();
        let mut re_hand = re0.clone();
        let mut im_hand = im0.clone();

        let mut scratch = rustfft_scratch();
        rustfft_forward(&mut re_rust, &mut im_rust, &mut scratch);
        fft_radix2(&mut re_hand, &mut im_hand);

        for i in 0..FFT_N {
            assert!(
                (re_rust[i] - re_hand[i]).abs() < 1e-8 && (im_rust[i] - im_hand[i]).abs() < 1e-8,
                "bin {i}: rustfft=({}, {}), handwritten=({}, {})",
                re_rust[i],
                im_rust[i],
                re_hand[i],
                im_hand[i]
            );
        }
    }
}
