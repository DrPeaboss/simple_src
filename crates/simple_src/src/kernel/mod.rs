pub(crate) mod cubic;
pub(crate) mod linear;
pub(crate) mod sinc;
pub(crate) mod spec;

use spec::KernelSpec;

/// Sample-rate conversion kernel.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kernel {
    #[default]
    Sinc,
    Linear,
    Cubic,
}

/// Sinc interpolation path selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SincPath {
    /// Use fast polyphase when the ratio is eligible; otherwise generic half-table.
    ///
    /// Any `quantify` set on the builder (including via [`crate::Quality`]) is
    /// ignored when this path selects fast.
    #[default]
    Auto,
    /// Half-table Kaiser-sinc interpolation; `quantify` is required.
    Generic,
    /// Polyphase LUT; `quantify` is ignored.
    Fast,
}

#[derive(Clone)]
pub(crate) enum KernelBackend {
    Linear(linear::Backend),
    Cubic(cubic::Backend),
    Sinc(sinc::Backend),
}

impl KernelBackend {
    pub(crate) fn linear(ratio: crate::Ratio) -> Self {
        Self::Linear(linear::Backend::new(ratio))
    }

    pub(crate) fn cubic(ratio: crate::Ratio) -> Self {
        Self::Cubic(cubic::Backend::new(ratio))
    }

    pub(crate) fn sinc(backend: sinc::Backend) -> Self {
        Self::Sinc(backend)
    }
}

impl KernelSpec for KernelBackend {
    fn ratio(&self) -> f64 {
        match self {
            Self::Linear(b) => b.ratio(),
            Self::Cubic(b) => b.ratio(),
            Self::Sinc(b) => b.ratio(),
        }
    }

    fn ratio_parts(&self) -> Option<(i64, i64)> {
        match self {
            Self::Linear(b) => b.ratio_parts(),
            Self::Cubic(b) => b.ratio_parts(),
            Self::Sinc(b) => b.ratio_parts(),
        }
    }

    fn mode(&self) -> crate::ConvertMode {
        match self {
            Self::Linear(b) => b.mode(),
            Self::Cubic(b) => b.mode(),
            Self::Sinc(b) => b.mode(),
        }
    }

    fn latency(&self) -> usize {
        match self {
            Self::Linear(b) => b.latency(),
            Self::Cubic(b) => b.latency(),
            Self::Sinc(b) => b.latency(),
        }
    }

    fn output_len(&self, input_len: usize) -> usize {
        match self {
            Self::Linear(b) => b.output_len(input_len),
            Self::Cubic(b) => b.output_len(input_len),
            Self::Sinc(b) => b.output_len(input_len),
        }
    }

    fn order(&self) -> Option<u32> {
        match self {
            Self::Linear(_) | Self::Cubic(_) => None,
            Self::Sinc(b) => Some(b.order()),
        }
    }

    fn lut_len(&self) -> Option<usize> {
        match self {
            Self::Linear(_) | Self::Cubic(_) => None,
            Self::Sinc(b) => Some(b.lut_len()),
        }
    }

    fn convert(&self, input: &[f64]) -> Vec<f64> {
        match self {
            Self::Linear(b) => b.convert(input),
            Self::Cubic(b) => b.convert(input),
            Self::Sinc(b) => b.convert(input),
        }
    }

    fn converter(&self) -> spec::KernelConverter {
        match self {
            Self::Linear(b) => b.converter(),
            Self::Cubic(b) => {
                spec::KernelConverter::Other(Box::new(spec::OtherKernel::Cubic(b.converter())))
            }
            Self::Sinc(b) => {
                spec::KernelConverter::Other(Box::new(spec::OtherKernel::Sinc(b.converter())))
            }
        }
    }
}
