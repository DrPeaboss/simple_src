use crate::{Convert, ConvertMode};

use super::{cubic, linear, sinc};

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

pub(crate) enum KernelConverter {
    Linear(linear::Converter),
    Cubic(cubic::Converter),
    Sinc(sinc::Converter),
}

impl Convert for KernelConverter {
    #[inline]
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        match self {
            Self::Linear(c) => c.next_sample(iter),
            Self::Cubic(c) => c.next_sample(iter),
            Self::Sinc(c) => c.next_sample(iter),
        }
    }

    fn flush(&mut self, output: &mut [f64]) -> usize
    where
        Self: Sized,
    {
        match self {
            Self::Linear(c) => c.flush(output),
            Self::Cubic(c) => c.flush(output),
            Self::Sinc(c) => c.flush(output),
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
        KernelConverter::Linear(linear::Backend::converter(self))
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
        KernelConverter::Cubic(cubic::Backend::converter(self))
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
        KernelConverter::Sinc(sinc::Backend::converter(self))
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
