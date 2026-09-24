//! FIR delay line.
//!
//! Mirrored ring buffer: the tap window is always one contiguous slice, so
//! the sinc dot kernels run one call per (row, window) pair instead of two
//! across a ring split. Cost is double storage (`2 * taps` f64) and one
//! window memmove per `taps` shifts (amortized one element per shift); each
//! individual shift is a single bounded store.

pub(crate) struct FirTap {
    /// `2 * taps` slots. The live window is `buf[w - taps + 1 ..= w]`,
    /// oldest first; `buf[.. w - taps + 1]` mirrors the window head so the
    /// window stays contiguous as `w` advances.
    buf: Vec<f64>,
    taps: usize,
    /// Index of the newest sample. Invariant: `taps - 1 <= w < 2 * taps`.
    w: usize,
}

impl FirTap {
    #[inline]
    pub(crate) fn new(taps: usize) -> Self {
        Self {
            buf: vec![0.0; 2 * taps],
            taps,
            w: taps - 1,
        }
    }

    #[inline]
    pub(crate) fn shift(&mut self, sample: f64) {
        self.w += 1;
        if self.w == 2 * self.taps {
            // The next window would start at `taps + 1` and run off the end;
            // move the live window to the front so writes can continue
            // contiguously.
            self.buf.copy_within(self.taps.., 0);
            self.w = self.taps;
        }
        self.buf[self.w] = sample;
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.window().iter().all(|&x| x == 0.0)
    }

    /// Contents in tap order (oldest first) as one contiguous slice. Feeds
    /// the SIMD dot-product kernels.
    #[inline]
    pub(crate) fn window(&self) -> &[f64] {
        &self.buf[self.w + 1 - self.taps..self.w + 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_matches_ring_semantics() {
        let taps = 4;
        let mut t = FirTap::new(taps);
        assert!(t.is_empty());
        // Ring semantics over a zero-filled line: after shift i the window is
        // the last `taps` values of the zero-prefixed stream [0,…,0, 1,2,…].
        let mut stream = vec![0.0; taps];
        stream.extend((1..=20).map(f64::from));
        for i in 0..7 {
            t.shift(stream[taps + i]);
            let expect = &stream[i + 1..i + 1 + taps];
            assert_eq!(t.window(), expect, "after shift {i}");
        }
        // Long run crosses the mirror boundary repeatedly (taps = 4 → every
        // 4 shifts); sweep well past it.
        let mut t = FirTap::new(4);
        let mut next = 1.0;
        for i in 0..97 {
            t.shift(next);
            let w = t.window();
            assert_eq!(w.len(), 4);
            assert_eq!(w[3], next, "newest at shift {i}");
            if i >= 3 {
                assert_eq!(w[..3], [next - 3.0, next - 2.0, next - 1.0]);
            }
            next += 1.0;
        }
    }

    #[test]
    fn empty_after_zero_fill_and_flush_shape() {
        // The flush path relies on is_empty(): all-zero window.
        let mut t = FirTap::new(6);
        for s in [0.0; 20] {
            t.shift(s);
        }
        assert!(t.is_empty());
        t.shift(1.0);
        assert!(!t.is_empty());
    }
}
