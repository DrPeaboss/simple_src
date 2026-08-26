pub(crate) mod linear;
pub(crate) mod sinc;

use crate::ConvertMode;

/// Sample-rate conversion kernel.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kernel {
    #[default]
    Sinc,
    Linear,
}

/// Sinc interpolation path selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SincPath {
    #[default]
    Auto,
    Generic,
    Fast,
}

#[derive(Clone)]
pub(crate) enum KernelBackend {
    Linear(linear::Backend),
    Sinc(sinc::Backend),
}

impl KernelBackend {
    pub(crate) fn linear(ratio: crate::Ratio) -> Self {
        Self::Linear(linear::Backend::new(ratio))
    }

    pub(crate) fn sinc(backend: sinc::Backend) -> Self {
        Self::Sinc(backend)
    }

    #[inline]
    pub(crate) fn ratio(&self) -> f64 {
        match self {
            Self::Linear(b) => b.ratio(),
            Self::Sinc(b) => b.ratio(),
        }
    }

    #[inline]
    pub(crate) fn ratio_parts(&self) -> Option<(i64, i64)> {
        match self {
            Self::Linear(b) => b.ratio_parts(),
            Self::Sinc(b) => b.ratio_parts(),
        }
    }

    #[inline]
    pub(crate) fn mode(&self) -> ConvertMode {
        match self {
            Self::Linear(b) => b.mode(),
            Self::Sinc(b) => b.mode(),
        }
    }

    #[inline]
    pub(crate) fn latency(&self) -> usize {
        match self {
            Self::Linear(b) => b.latency(),
            Self::Sinc(b) => b.latency(),
        }
    }

    #[inline]
    pub(crate) fn output_len(&self, input_len: usize) -> usize {
        match self {
            Self::Linear(b) => b.output_len(input_len),
            Self::Sinc(b) => b.output_len(input_len),
        }
    }

    pub(crate) fn convert(&self, input: &[f64]) -> Vec<f64> {
        match self {
            Self::Linear(b) => b.convert(input),
            Self::Sinc(b) => b.convert(input),
        }
    }

    #[inline]
    pub(crate) fn order(&self) -> Option<u32> {
        match self {
            Self::Linear(_) => None,
            Self::Sinc(b) => Some(b.order()),
        }
    }

    #[inline]
    pub(crate) fn lut_len(&self) -> Option<usize> {
        match self {
            Self::Linear(_) => None,
            Self::Sinc(b) => Some(b.lut_len()),
        }
    }
}
