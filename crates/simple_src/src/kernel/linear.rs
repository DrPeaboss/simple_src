use crate::{Convert, ConvertMode, Ratio, Rational, convert_with, output_len};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    First,
    Normal,
    Suspend,
}

/// Float-phase linear core: all state lives in one struct so the hot loop
/// stays register-resident after inlining.
pub(crate) struct FloatCore {
    state: State,
    last_in: [f64; 2],
    step: f64,
    pos: f64,
}

impl FloatCore {
    pub(crate) fn new(step: f64) -> Self {
        Self {
            state: State::First,
            last_in: [0.0; 2],
            step,
            pos: 0.0,
        }
    }
}

impl Convert for FloatCore {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        loop {
            match self.state {
                State::First => {
                    let s = iter.next()?;
                    // Aligned start: first output equals the first input.
                    // Seed both taps with the duplicate; the forced advance
                    // below replaces the second tap with the next input.
                    self.last_in = [s, s];
                    self.pos = 1.0;
                    self.state = State::Normal;
                }
                State::Normal => {
                    while self.pos >= 1.0 {
                        self.pos -= 1.0;
                        self.last_in[0] = self.last_in[1];
                        if let Some(s) = iter.next() {
                            self.last_in[1] = s;
                        } else {
                            self.state = State::Suspend;
                            return None;
                        }
                    }
                    let interp = self.last_in[0] + (self.last_in[1] - self.last_in[0]) * self.pos;
                    self.pos += self.step;
                    return Some(interp);
                }
                State::Suspend => {
                    let s = iter.next()?;
                    self.last_in[1] = s;
                    self.state = State::Normal;
                }
            }
        }
    }
}

/// Generic rational-phase linear core (no precomputed coefficient table).
pub(crate) struct RationalCore {
    state: State,
    last_in: [f64; 2],
    numer: usize,
    denom: usize,
    pos: usize,
    recip: f64,
}

impl RationalCore {
    /// `step` is the input-consumption step (the reciprocal of the ratio).
    pub(crate) fn new(step: Rational) -> Self {
        let numer = *step.numer() as usize;
        let denom = *step.denom() as usize;
        Self {
            state: State::First,
            last_in: [0.0; 2],
            numer,
            denom,
            pos: 0,
            recip: (denom as f64).recip(),
        }
    }
}

impl Convert for RationalCore {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
    {
        loop {
            match self.state {
                State::First => {
                    let s = iter.next()?;
                    // Aligned start: see FloatCore::First.
                    self.last_in = [s, s];
                    self.pos = self.denom;
                    self.state = State::Normal;
                }
                State::Normal => {
                    while self.pos >= self.denom {
                        self.pos -= self.denom;
                        self.last_in[0] = self.last_in[1];
                        if let Some(s) = iter.next() {
                            self.last_in[1] = s;
                        } else {
                            self.state = State::Suspend;
                            return None;
                        }
                    }
                    let coef = self.pos as f64 * self.recip;
                    let interp = self.last_in[0] + (self.last_in[1] - self.last_in[0]) * coef;
                    self.pos += self.numer;
                    return Some(interp);
                }
                State::Suspend => {
                    let s = iter.next()?;
                    self.last_in[1] = s;
                    self.state = State::Normal;
                }
            }
        }
    }
}

/// Rational-phase linear core with a precomputed fractional coefficient
/// table; the common case for exact integer rate pairs.
pub(crate) struct RationalFastCore {
    state: State,
    last_in: [f64; 2],
    numer: usize,
    denom: usize,
    pos: usize,
    coef: Vec<f64>,
}

impl RationalFastCore {
    /// `step` is the input-consumption step (the reciprocal of the ratio).
    pub(crate) fn new(step: Rational) -> Self {
        let numer = *step.numer() as usize;
        let denom = *step.denom() as usize;
        let coef = (0..denom).map(|i| i as f64 / denom as f64).collect();
        Self {
            state: State::First,
            last_in: [0.0; 2],
            numer,
            denom,
            pos: 0,
            coef,
        }
    }
}

impl Convert for RationalFastCore {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
    {
        loop {
            match self.state {
                State::First => {
                    let s = iter.next()?;
                    // Aligned start: see FloatCore::First.
                    self.last_in = [s, s];
                    self.pos = self.denom;
                    self.state = State::Normal;
                }
                State::Normal => {
                    while self.pos >= self.denom {
                        self.pos -= self.denom;
                        self.last_in[0] = self.last_in[1];
                        if let Some(s) = iter.next() {
                            self.last_in[1] = s;
                        } else {
                            self.state = State::Suspend;
                            return None;
                        }
                    }
                    let coef = self.coef[self.pos];
                    let interp = self.last_in[0] + (self.last_in[1] - self.last_in[0]) * coef;
                    self.pos += self.numer;
                    return Some(interp);
                }
                State::Suspend => {
                    let s = iter.next()?;
                    self.last_in[1] = s;
                    self.state = State::Normal;
                }
            }
        }
    }
}

impl FloatCore {
    pub(crate) fn delay_empty(&self) -> bool {
        self.state == State::First || (self.last_in[0] == 0.0 && self.last_in[1] == 0.0)
    }
}

impl RationalCore {
    pub(crate) fn delay_empty(&self) -> bool {
        self.state == State::First || (self.last_in[0] == 0.0 && self.last_in[1] == 0.0)
    }
}

impl RationalFastCore {
    pub(crate) fn delay_empty(&self) -> bool {
        self.state == State::First || (self.last_in[0] == 0.0 && self.last_in[1] == 0.0)
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
    pub(crate) fn ratio(&self) -> f64 {
        self.ratio.as_float()
    }

    #[inline]
    pub(crate) fn ratio_enum(&self) -> Ratio {
        self.ratio
    }

    #[inline]
    pub(crate) fn ratio_parts(&self) -> Option<(i64, i64)> {
        self.ratio.parts()
    }

    #[inline]
    pub(crate) fn mode(&self) -> ConvertMode {
        self.ratio.polynomial_mode()
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
        let converter = crate::kernel::spec::KernelSpec::converter(self);
        convert_with(converter, self.latency(), self.ratio(), input)
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
        let mut converter = crate::kernel::spec::KernelSpec::converter(backend);
        let mut out = Vec::new();
        let mut iter = input.iter().copied();
        while let Some(sample) = converter.next_sample(&mut iter) {
            out.push(sample);
        }
        drain_linear_flush(&mut converter, &mut out);
        out
    }

    fn collect_linear_chunks(backend: &Backend, chunks: &[&[f64]]) -> Vec<f64> {
        let mut converter = crate::kernel::spec::KernelSpec::converter(backend);
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

    fn drain_linear_flush(
        converter: &mut crate::kernel::spec::KernelConverter,
        out: &mut Vec<f64>,
    ) {
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
