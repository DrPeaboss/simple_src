use crate::{
    Convert, ConvertMode, Ratio, convert_with, engine::LinearState, engine::PhaseAccum,
    engine::TwoTap, output_len, ratio::LINEAR_FAST_NUMER_MAX,
};

struct LinearCore {
    phase: PhaseAccum,
    state: LinearState,
    taps: TwoTap,
}

impl LinearCore {
    fn new(phase: PhaseAccum) -> Self {
        Self {
            phase,
            state: LinearState::new(),
            taps: TwoTap::new(),
        }
    }
}

impl Convert for LinearCore {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        loop {
            match self.state {
                LinearState::Priming => {
                    let s = iter.next()?;
                    self.taps.set_second(s);
                    self.phase.prepare_linear_priming();
                    self.state = self.state.finish_priming();
                }
                LinearState::Running => {
                    while self.phase.needs_input_advance() {
                        self.phase.consume_input_step();
                        if let Some(s) = iter.next() {
                            self.taps.shift(s);
                        } else {
                            self.taps.advance_left();
                            self.state = self.state.on_input_exhausted();
                            return None;
                        }
                    }
                    let interp = self.taps.interpolate(self.phase.coef());
                    self.phase.advance_output();
                    return Some(interp);
                }
                LinearState::Suspended => {
                    let s = iter.next()?;
                    self.taps.set_second(s);
                    self.state = self.state.on_input_resumed();
                }
            }
        }
    }
}

pub(crate) struct Converter {
    inner: LinearCore,
}

impl Converter {
    fn delay_empty(&self) -> bool {
        self.inner.state.is_priming() || self.inner.taps.is_empty()
    }

    pub(crate) fn new(ratio: Ratio) -> Self {
        let phase = match ratio {
            Ratio::Float(ratio) => PhaseAccum::float(ratio.recip()),
            Ratio::Rational(ratio) => {
                if *ratio.numer() <= LINEAR_FAST_NUMER_MAX {
                    PhaseAccum::rational_fast_linear(ratio.recip())
                } else {
                    PhaseAccum::rational(ratio.recip())
                }
            }
        };
        Self {
            inner: LinearCore::new(phase),
        }
    }
}

impl Convert for Converter {
    #[inline]
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        self.inner.next_sample(iter)
    }

    fn flush(&mut self, output: &mut [f64]) -> usize
    where
        Self: Sized,
    {
        if self.delay_empty() {
            return 0;
        }
        let mut zeros = std::iter::repeat(0.0);
        let mut produced = 0;
        while produced < output.len() {
            match self.next_sample(&mut zeros) {
                Some(sample) => {
                    output[produced] = sample;
                    produced += 1;
                    if self.delay_empty() {
                        break;
                    }
                }
                None => break,
            }
        }
        produced
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Backend {
    ratio: Ratio,
}

impl Backend {
    pub(crate) fn new(ratio: Ratio) -> Self {
        Self { ratio }
    }

    #[inline]
    pub(crate) fn converter(&self) -> Converter {
        Converter::new(self.ratio)
    }

    #[inline]
    pub(crate) fn ratio(&self) -> f64 {
        self.ratio.as_float()
    }

    #[inline]
    pub(crate) fn ratio_parts(&self) -> Option<(i64, i64)> {
        self.ratio.parts()
    }

    #[inline]
    pub(crate) fn mode(&self) -> ConvertMode {
        self.ratio.linear_mode()
    }

    #[inline]
    pub(crate) fn latency(&self) -> usize {
        0
    }

    #[inline]
    pub(crate) fn output_len(&self, input_len: usize) -> usize {
        output_len(self.ratio(), input_len)
    }

    pub(crate) fn convert(&self, input: &[f64]) -> Vec<f64> {
        convert_with(self.converter(), self.latency(), self.ratio(), input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Ratio;

    #[test]
    fn ratio_bounds() {
        let ratio_ok = vec![0.0625, 0.063, 1.0, 15.9, 16.0, 0.123456];
        for ratio in ratio_ok {
            assert!(Ratio::try_from_float(ratio).is_ok());
        }
        let ratio_err = vec![
            -1.0,
            0.0,
            0.0624,
            16.01,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for ratio in ratio_err {
            assert!(Ratio::try_from_float(ratio).is_err());
        }
    }

    #[test]
    fn mode_and_sample_rate() {
        let m = Backend::new(Ratio::try_from_integers(48000, 44100).unwrap());
        assert_eq!(m.mode(), ConvertMode::RationalFast);
        assert_eq!(m.ratio_parts(), Some((160, 147)));
        let two = Backend::new(Ratio::try_from_float(2.0).unwrap());
        assert_eq!(two.mode(), ConvertMode::RationalFast);
        let generic = Backend::new(Ratio::try_from_integers(20000, 19999).unwrap());
        assert_eq!(generic.mode(), ConvertMode::Rational);
        let pi = Backend::new(Ratio::try_from_float(std::f64::consts::PI).unwrap());
        assert_eq!(pi.mode(), ConvertMode::Float);
        assert_eq!(pi.ratio_parts(), None);
    }

    #[test]
    fn chunked_input_matches_continuous() {
        let backend = Backend::new(Ratio::try_from_float(2.0).unwrap());
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let continuous = collect_linear(&backend, &input);
        let chunked = collect_linear_chunks(&backend, &[&input[..5], &input[5..]]);

        assert_eq!(continuous.len(), chunked.len());
        for (a, b) in continuous.iter().zip(chunked.iter()) {
            assert!((a - b).abs() < 1e-12, "chunked resume mismatch: {a} vs {b}");
        }
    }

    fn collect_linear(backend: &Backend, input: &[f64]) -> Vec<f64> {
        let mut converter = backend.converter();
        let mut out = Vec::new();
        let mut iter = input.iter().copied();
        while let Some(sample) = converter.next_sample(&mut iter) {
            out.push(sample);
        }
        drain_linear_flush(&mut converter, &mut out);
        out
    }

    fn collect_linear_chunks(backend: &Backend, chunks: &[&[f64]]) -> Vec<f64> {
        let mut converter = backend.converter();
        let mut out = Vec::new();
        for chunk in chunks {
            let mut iter = chunk.iter().copied();
            while let Some(sample) = converter.next_sample(&mut iter) {
                out.push(sample);
            }
        }
        drain_linear_flush(&mut converter, &mut out);
        out
    }

    fn drain_linear_flush(converter: &mut Converter, out: &mut Vec<f64>) {
        let mut tail = [0.0; 16];
        loop {
            let n = converter.flush(&mut tail);
            if n == 0 {
                break;
            }
            out.extend_from_slice(&tail[..n]);
        }
    }
}
