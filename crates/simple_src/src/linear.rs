//! Linear interpolation converter
//!
//! ```
//! use simple_src::{linear, Convert};
//!
//! let samples = vec![1.0, 2.0, 3.0, 4.0];
//! let manager = linear::Manager::new(2.0).unwrap();
//! let mut converter = manager.converter();
//! for s in converter.process(samples.into_iter()) {
//!     println!("{s}");
//! }
//! ```

use super::{
    Convert, ConvertMode, Ratio, Rational, Result, convert_with, engine::LinearState,
    engine::PhaseAccum, engine::TwoTap, output_len,
};

pub(crate) struct FloatConverter {
    phase: PhaseAccum,
    state: LinearState,
    taps: TwoTap,
}

pub(crate) struct RationalConverter {
    phase: PhaseAccum,
    state: LinearState,
    taps: TwoTap,
}

pub(crate) struct RationalFastConverter {
    phase: PhaseAccum,
    state: LinearState,
    taps: TwoTap,
}

enum ConverterKind {
    Float(FloatConverter),
    Rational(RationalConverter),
    RationalFast(RationalFastConverter),
}

/// Opaque sample-rate converter created by [`Manager::converter`].
pub struct Converter {
    inner: ConverterKind,
}

impl FloatConverter {
    fn new(step: f64) -> Self {
        Self {
            phase: PhaseAccum::float(step),
            state: LinearState::new(),
            taps: TwoTap::new(),
        }
    }
}

impl Convert for FloatConverter {
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

impl RationalConverter {
    fn new(step: Rational) -> Self {
        Self {
            phase: PhaseAccum::rational(step),
            state: LinearState::new(),
            taps: TwoTap::new(),
        }
    }
}

impl Convert for RationalConverter {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
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

impl RationalFastConverter {
    fn new(step: Rational) -> Self {
        Self {
            phase: PhaseAccum::rational_fast_linear(step),
            state: LinearState::new(),
            taps: TwoTap::new(),
        }
    }
}

impl Convert for RationalFastConverter {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
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

impl Converter {
    fn delay_empty(&self) -> bool {
        match &self.inner {
            ConverterKind::Float(c) => c.state.is_priming() || c.taps.is_empty(),
            ConverterKind::Rational(c) => c.state.is_priming() || c.taps.is_empty(),
            ConverterKind::RationalFast(c) => c.state.is_priming() || c.taps.is_empty(),
        }
    }

    fn new(ratio: Ratio) -> Self {
        let inner = match ratio {
            Ratio::Float(ratio) => ConverterKind::Float(FloatConverter::new(ratio.recip())),
            Ratio::Rational(ratio) => {
                if *ratio.numer() <= 16384 {
                    ConverterKind::RationalFast(RationalFastConverter::new(ratio.recip()))
                } else {
                    ConverterKind::Rational(RationalConverter::new(ratio.recip()))
                }
            }
        };
        Self { inner }
    }
}

impl Convert for Converter {
    #[inline]
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        match &mut self.inner {
            ConverterKind::Float(converter) => converter.next_sample(iter),
            ConverterKind::Rational(converter) => converter.next_sample(iter),
            ConverterKind::RationalFast(converter) => converter.next_sample(iter),
        }
    }

    fn flush(&mut self, output: &mut [f64]) -> usize
    where
        Self: Sized,
    {
        // Overrides Convert::flush: stop when the two-sample delay is empty.
        // Still call until 0 if `output` fills first.
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
pub struct Manager {
    ratio: Ratio,
}

impl Manager {
    #[inline]
    pub fn new(ratio: f64) -> Result<Self> {
        let ratio = Ratio::try_from_float(ratio)?;
        Ok(Self { ratio })
    }

    #[inline]
    pub fn with_sample_rate(old_sr: u32, new_sr: u32) -> Result<Self> {
        let ratio = Ratio::try_from_integers(new_sr, old_sr)?;
        Ok(Self { ratio })
    }

    #[inline]
    pub fn converter(&self) -> Converter {
        Converter::new(self.ratio)
    }

    #[inline]
    pub fn ratio(&self) -> f64 {
        self.ratio.as_float()
    }

    /// Reduced integer ratio when a bounded float approximation (or integer
    /// sample rates) selected a rational; `None` for float phase. See
    /// [`ConvertMode`].
    #[inline]
    pub fn ratio_parts(&self) -> Option<(i64, i64)> {
        self.ratio.parts()
    }

    /// Which interpolation path this manager will construct. See [`ConvertMode`].
    #[inline]
    pub fn mode(&self) -> ConvertMode {
        self.ratio.linear_mode()
    }

    #[inline]
    pub fn latency(&self) -> usize {
        0
    }

    #[inline]
    pub fn output_len(&self, input_len: usize) -> usize {
        output_len(self.ratio(), input_len)
    }

    /// Convert a complete buffer, padding the end with zeros.
    pub fn convert(&self, input: &[f64]) -> Vec<f64> {
        convert_with(self.converter(), self.latency(), self.ratio(), input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_ok() {
        let ratio_ok = vec![0.0625, 0.063, 1.0, 15.9, 16.0, 0.123456];
        for ratio in ratio_ok {
            assert!(Manager::new(ratio).is_ok());
        }
    }

    #[test]
    fn test_manager_err() {
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
            assert!(Manager::new(ratio).is_err());
        }
    }

    #[test]
    fn test_mode_and_sample_rate() {
        let m = Manager::with_sample_rate(44100, 48000).unwrap();
        assert_eq!(m.mode(), ConvertMode::RationalFast);
        assert_eq!(m.ratio_parts(), Some((160, 147)));
        let two = Manager::new(2.0).unwrap();
        assert_eq!(two.mode(), ConvertMode::RationalFast);
        let generic = Manager::with_sample_rate(19999, 20000).unwrap();
        assert_eq!(generic.mode(), ConvertMode::Rational);
        let pi = Manager::new(std::f64::consts::PI).unwrap();
        assert_eq!(pi.mode(), ConvertMode::Float);
        assert_eq!(pi.ratio_parts(), None);
    }

    #[test]
    fn chunked_input_matches_continuous() {
        let manager = Manager::new(2.0).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let continuous = collect_linear(&manager, &input);
        let chunked = collect_linear_chunks(&manager, &[&input[..5], &input[5..]]);

        assert_eq!(continuous.len(), chunked.len());
        for (a, b) in continuous.iter().zip(chunked.iter()) {
            assert!((a - b).abs() < 1e-12, "chunked resume mismatch: {a} vs {b}");
        }
    }

    fn collect_linear(manager: &Manager, input: &[f64]) -> Vec<f64> {
        let mut converter = manager.converter();
        let mut out = Vec::new();
        let mut iter = input.iter().copied();
        while let Some(sample) = converter.next_sample(&mut iter) {
            out.push(sample);
        }
        drain_linear_flush(&mut converter, &mut out);
        out
    }

    fn collect_linear_chunks(manager: &Manager, chunks: &[&[f64]]) -> Vec<f64> {
        let mut converter = manager.converter();
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
