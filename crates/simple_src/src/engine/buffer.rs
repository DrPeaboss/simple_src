use std::collections::VecDeque;

pub(crate) struct FirTap {
    buf: VecDeque<f64>,
}

impl FirTap {
    #[inline]
    pub(crate) fn new(taps: usize) -> Self {
        let mut buf = VecDeque::with_capacity(taps);
        buf.extend(std::iter::repeat_n(0.0, taps));
        Self { buf }
    }

    #[inline]
    pub(crate) fn shift(&mut self, sample: f64) {
        self.buf.pop_front();
        self.buf.push_back(sample);
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.buf.iter().all(|&x| x == 0.0)
    }

    /// Contents in tap order (oldest first) as two contiguous slices, split
    /// where the ring wraps. Feeds the SIMD dot-product kernels.
    #[inline]
    pub(crate) fn slices(&self) -> (&[f64], &[f64]) {
        self.buf.as_slices()
    }
}
