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
