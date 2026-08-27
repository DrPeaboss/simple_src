use crate::{
    Convert, ConvertMode, Ratio, convert_with,
    engine::{
        FloatPhase, FourTap, LinearState, PolynomialPhase, RationalFastPhase, RationalPhase,
        polynomial_next_sample, polynomial_phase,
    },
    kernel::spec::drain_flush,
    output_len,
};

/// Monomorphic cubic core; the phase mode is fixed at construction so the
/// hot loop compiles to register-resident code with a single dispatch per
/// output sample.
pub(crate) struct PolyCore<P: PolynomialPhase> {
    phase: P,
    state: LinearState,
    taps: FourTap,
}

impl<P: PolynomialPhase> PolyCore<P> {
    pub(crate) fn new(phase: P) -> Self {
        Self {
            phase,
            state: LinearState::new_cubic(),
            taps: FourTap::new(),
        }
    }

    pub(crate) fn delay_empty(&self) -> bool {
        self.state.is_priming() || self.taps.is_empty()
    }
}

impl<P: PolynomialPhase> Convert for PolyCore<P> {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        polynomial_next_sample(
            crate::engine::PolynomialKind::FourTap,
            &mut self.state,
            &mut self.phase,
            &mut self.taps,
            iter,
        )
    }
}

/// Cubic converter: mode-level dispatch nested one level below the kernel
/// dispatch. Cubic is not on the linear hot path, so the extra match here is
/// amortized (its per-sample cost is dominated by the Catmull-Rom math).
pub(crate) enum Converter {
    Float(PolyCore<FloatPhase>),
    Rational(PolyCore<RationalPhase>),
    RationalFast(PolyCore<RationalFastPhase>),
}

impl Convert for Converter {
    #[inline]
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        match self {
            Self::Float(c) => c.next_sample(iter),
            Self::Rational(c) => c.next_sample(iter),
            Self::RationalFast(c) => c.next_sample(iter),
        }
    }

    fn flush(&mut self, output: &mut [f64]) -> usize
    where
        Self: Sized,
    {
        match self {
            Self::Float(c) => drain_flush(c, output),
            Self::Rational(c) => drain_flush(c, output),
            Self::RationalFast(c) => drain_flush(c, output),
        }
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
    pub(crate) fn converter(&self) -> Converter {
        use crate::engine::PhaseFor;
        match polynomial_phase(&self.ratio) {
            PhaseFor::Float(phase) => Converter::Float(PolyCore::new(phase)),
            PhaseFor::Rational(phase) => Converter::Rational(PolyCore::new(phase)),
            PhaseFor::RationalFast(phase) => Converter::RationalFast(PolyCore::new(phase)),
        }
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
    fn mode_and_sample_rate() {
        let m = Backend::new(Ratio::try_from_integers(48000, 44100).unwrap());
        assert_eq!(m.mode(), ConvertMode::RationalFast);
        let pi = Backend::new(Ratio::try_from_float(std::f64::consts::PI).unwrap());
        assert_eq!(pi.mode(), ConvertMode::Float);
    }

    #[test]
    fn chunked_input_matches_continuous() {
        let backend = Backend::new(Ratio::try_from_float(2.0).unwrap());
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let continuous = collect_cubic(&backend, &input);
        let chunked = collect_cubic_chunks(&backend, &[&input[..5], &input[5..]]);

        assert_eq!(continuous.len(), chunked.len());
        for (a, b) in continuous.iter().zip(chunked.iter()) {
            assert!((a - b).abs() < 1e-12, "chunked resume mismatch: {a} vs {b}");
        }
    }

    fn collect_cubic(backend: &Backend, input: &[f64]) -> Vec<f64> {
        let mut converter = crate::kernel::spec::KernelSpec::converter(backend);
        let mut out = Vec::new();
        let mut iter = input.iter().copied();
        while let Some(sample) = converter.next_sample(&mut iter) {
            out.push(sample);
        }
        drain_flush(&mut converter, &mut out);
        out
    }

    fn collect_cubic_chunks(backend: &Backend, chunks: &[&[f64]]) -> Vec<f64> {
        let mut converter = crate::kernel::spec::KernelSpec::converter(backend);
        let mut out = Vec::new();
        for chunk in chunks {
            let mut iter = chunk.iter().copied();
            while let Some(sample) = converter.next_sample(&mut iter) {
                out.push(sample);
            }
        }
        drain_flush(&mut converter, &mut out);
        out
    }

    fn drain_flush(converter: &mut crate::kernel::spec::KernelConverter, out: &mut Vec<f64>) {
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
