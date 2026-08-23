use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TwoTap {
    data: [f64; 2],
}

impl TwoTap {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub(crate) fn set_second(&mut self, sample: f64) {
        self.data[1] = sample;
    }

    #[inline]
    pub(crate) fn shift(&mut self, sample: f64) {
        self.data[0] = self.data[1];
        self.data[1] = sample;
    }

    /// Advance the left tap when input is unavailable mid-interval.
    #[inline]
    pub(crate) fn advance_left(&mut self) {
        self.data[0] = self.data[1];
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.data[0] == 0.0 && self.data[1] == 0.0
    }

    #[inline]
    pub(crate) fn interpolate(&self, coef: f64) -> f64 {
        self.data[0] + (self.data[1] - self.data[0]) * coef
    }
}

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

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub(crate) fn get(&self, index: usize) -> f64 {
        self.buf[index]
    }

    #[inline]
    pub(crate) fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.buf.iter().copied()
    }
}
