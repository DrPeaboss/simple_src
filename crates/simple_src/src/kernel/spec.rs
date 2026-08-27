use crate::{Convert, ConvertMode};

use super::{cubic, linear, sinc};

/// Drain an output buffer from a converter that stops once its delay line is
/// empty. Shared by the linear and cubic cores.
pub(crate) trait Delayed: Convert {
    fn is_drained(&self) -> bool;
}

/// Run the conversion loop for one concrete core type; `C` is monomorphic so
/// the loop body needs no per-sample dispatch.
///
/// `#[inline(never)]` keeps each arm's loop a separate function: if LLVM
/// inlines all arms into `process_block` it merges them into a single loop
/// with an indirect jump table, which defeats register promotion of the
/// converter state and costs ~2x on the batch path.
#[inline(never)]
fn batch<C: Convert>(converter: &mut C, input: &[f64], output: &mut [f64]) -> (usize, usize) {
    let mut iter = crate::SliceIter {
        data: input,
        pos: 0,
    };
    let mut produced = 0;
    while produced < output.len() {
        match converter.next_sample(&mut iter) {
            Some(sample) => {
                output[produced] = sample;
                produced += 1;
            }
            None => break,
        }
    }
    (iter.pos, produced)
}

pub(crate) fn drain_flush<C: Delayed>(converter: &mut C, output: &mut [f64]) -> usize {
    if converter.is_drained() {
        return 0;
    }
    let mut zeros = std::iter::repeat(0.0);
    let mut produced = 0;
    while produced < output.len() {
        match converter.next_sample(&mut zeros) {
            Some(sample) => {
                output[produced] = sample;
                produced += 1;
                if converter.is_drained() {
                    break;
                }
            }
            None => break,
        }
    }
    produced
}

/// Internal contract for sample-rate conversion backends.
pub(crate) trait KernelSpec {
    fn ratio(&self) -> f64;
    fn ratio_parts(&self) -> Option<(i64, i64)>;
    fn mode(&self) -> ConvertMode;
    fn latency(&self) -> usize;
    fn output_len(&self, input_len: usize) -> usize;
    fn order(&self) -> Option<u32>;
    fn lut_len(&self) -> Option<usize>;
    fn convert(&self, input: &[f64]) -> Vec<f64>;
    fn converter(&self) -> KernelConverter;
}

/// Kernel converters, flattened to mode level: a single match per output
/// sample keeps the polynomial hot loops register-resident (the previous
/// nested `Kernel -> mode` enum layering defeated LLVM's register promotion
/// and cost ~3x on the linear path).
pub(crate) enum OtherKernel {
    Cubic(cubic::Converter),
    Sinc(sinc::Converter),
}

impl Convert for OtherKernel {
    #[inline]
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        match self {
            Self::Cubic(c) => c.next_sample(iter),
            Self::Sinc(c) => c.next_sample(iter),
        }
    }

    fn process_block(&mut self, input: &[f64], output: &mut [f64]) -> (usize, usize)
    where
        Self: Sized,
    {
        match self {
            Self::Cubic(c) => batch(c, input, output),
            Self::Sinc(c) => batch(c, input, output),
        }
    }

    fn flush(&mut self, output: &mut [f64]) -> usize
    where
        Self: Sized,
    {
        match self {
            Self::Cubic(c) => c.flush(output),
            Self::Sinc(c) => c.flush(output),
        }
    }
}

pub(crate) enum KernelConverter {
    LinearFloat(linear::FloatCore),
    LinearRational(linear::RationalCore),
    LinearRationalFast(linear::RationalFastCore),
    /// Cubic and sinc kernels, boxed behind one variant: the linear arms stay
    /// in front of the match so the linear hot path keeps the same shape as a
    /// single-kernel converter (LLVM threads 3-4 compare arms but degrades to
    /// an indirect jump table with more variants, costing ~2x).
    Other(Box<OtherKernel>),
}

impl Delayed for linear::FloatCore {
    fn is_drained(&self) -> bool {
        Self::delay_empty(self)
    }
}

impl Delayed for linear::RationalCore {
    fn is_drained(&self) -> bool {
        Self::delay_empty(self)
    }
}

impl Delayed for linear::RationalFastCore {
    fn is_drained(&self) -> bool {
        Self::delay_empty(self)
    }
}

impl<P: crate::engine::PolynomialPhase> Delayed for cubic::PolyCore<P> {
    fn is_drained(&self) -> bool {
        Self::delay_empty(self)
    }
}

impl Convert for KernelConverter {
    /// Batch override: the sample loop runs *inside* each matched arm, so the
    /// variant dispatch happens once per block instead of once per sample and
    /// the concrete core's state machine stays register-resident.
    fn process_block(&mut self, input: &[f64], output: &mut [f64]) -> (usize, usize)
    where
        Self: Sized,
    {
        match self {
            Self::LinearFloat(c) => batch(c, input, output),
            Self::LinearRational(c) => batch(c, input, output),
            Self::LinearRationalFast(c) => batch(c, input, output),
            Self::Other(other) => other.process_block(input, output),
        }
    }

    #[inline]
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        match self {
            Self::LinearFloat(c) => c.next_sample(iter),
            Self::LinearRational(c) => c.next_sample(iter),
            Self::LinearRationalFast(c) => c.next_sample(iter),
            Self::Other(other) => other.next_sample(iter),
        }
    }

    fn flush(&mut self, output: &mut [f64]) -> usize
    where
        Self: Sized,
    {
        match self {
            Self::LinearFloat(c) => drain_flush(c, output),
            Self::LinearRational(c) => drain_flush(c, output),
            Self::LinearRationalFast(c) => drain_flush(c, output),
            Self::Other(other) => other.flush(output),
        }
    }
}

impl KernelSpec for linear::Backend {
    #[inline]
    fn ratio(&self) -> f64 {
        linear::Backend::ratio(self)
    }

    #[inline]
    fn ratio_parts(&self) -> Option<(i64, i64)> {
        linear::Backend::ratio_parts(self)
    }

    #[inline]
    fn mode(&self) -> ConvertMode {
        linear::Backend::mode(self)
    }

    #[inline]
    fn latency(&self) -> usize {
        linear::Backend::latency(self)
    }

    #[inline]
    fn output_len(&self, input_len: usize) -> usize {
        linear::Backend::output_len(self, input_len)
    }

    #[inline]
    fn order(&self) -> Option<u32> {
        None
    }

    #[inline]
    fn lut_len(&self) -> Option<usize> {
        None
    }

    fn convert(&self, input: &[f64]) -> Vec<f64> {
        linear::Backend::convert(self, input)
    }

    fn converter(&self) -> KernelConverter {
        match self.ratio_enum() {
            crate::Ratio::Float(r) => {
                KernelConverter::LinearFloat(linear::FloatCore::new(r.recip()))
            }
            crate::Ratio::Rational(r) => {
                if *r.numer() <= crate::ratio::LINEAR_FAST_NUMER_MAX {
                    KernelConverter::LinearRationalFast(linear::RationalFastCore::new(r.recip()))
                } else {
                    KernelConverter::LinearRational(linear::RationalCore::new(r.recip()))
                }
            }
        }
    }
}

impl KernelSpec for cubic::Backend {
    #[inline]
    fn ratio(&self) -> f64 {
        cubic::Backend::ratio(self)
    }

    #[inline]
    fn ratio_parts(&self) -> Option<(i64, i64)> {
        cubic::Backend::ratio_parts(self)
    }

    #[inline]
    fn mode(&self) -> ConvertMode {
        cubic::Backend::mode(self)
    }

    #[inline]
    fn latency(&self) -> usize {
        cubic::Backend::latency(self)
    }

    #[inline]
    fn output_len(&self, input_len: usize) -> usize {
        cubic::Backend::output_len(self, input_len)
    }

    #[inline]
    fn order(&self) -> Option<u32> {
        None
    }

    #[inline]
    fn lut_len(&self) -> Option<usize> {
        None
    }

    fn convert(&self, input: &[f64]) -> Vec<f64> {
        cubic::Backend::convert(self, input)
    }

    fn converter(&self) -> KernelConverter {
        KernelConverter::Other(Box::new(OtherKernel::Cubic(
            self.ratio_enum().cubic_converter(),
        )))
    }
}

impl KernelSpec for sinc::Backend {
    #[inline]
    fn ratio(&self) -> f64 {
        sinc::Backend::ratio(self)
    }

    #[inline]
    fn ratio_parts(&self) -> Option<(i64, i64)> {
        sinc::Backend::ratio_parts(self)
    }

    #[inline]
    fn mode(&self) -> ConvertMode {
        sinc::Backend::mode(self)
    }

    #[inline]
    fn latency(&self) -> usize {
        sinc::Backend::latency(self)
    }

    #[inline]
    fn output_len(&self, input_len: usize) -> usize {
        sinc::Backend::output_len(self, input_len)
    }

    #[inline]
    fn order(&self) -> Option<u32> {
        Some(sinc::Backend::order(self))
    }

    #[inline]
    fn lut_len(&self) -> Option<usize> {
        Some(sinc::Backend::lut_len(self))
    }

    fn convert(&self, input: &[f64]) -> Vec<f64> {
        sinc::Backend::convert(self, input)
    }

    fn converter(&self) -> KernelConverter {
        KernelConverter::Other(Box::new(OtherKernel::Sinc(sinc::Backend::converter(self))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Kernel, SrcManager};

    #[test]
    fn backends_implement_kernelspec_smoke() {
        let linear = SrcManager::with_ratio(2.0).unwrap();
        assert_eq!(linear.mode(), ConvertMode::RationalFast);
        assert_eq!(linear.order(), None);
        assert_eq!(linear.lut_len(), None);

        let cubic = SrcManager::builder()
            .ratio(2.0)
            .kernel(Kernel::Cubic)
            .build()
            .unwrap();
        assert_eq!(cubic.latency(), 0);
        assert_eq!(cubic.order(), None);

        let sinc = SrcManager::builder()
            .ratio(2.0)
            .generic()
            .attenuation(48.0)
            .quantify(8)
            .trans_width(0.1)
            .build()
            .unwrap();
        assert!(sinc.order().is_some());
        assert!(sinc.lut_len().is_some());
    }
}
