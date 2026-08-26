use crate::converter::Converter;
use crate::kernel::sinc::builder::Builder as SincBuilder;
use crate::kernel::spec::KernelSpec;
use crate::kernel::{Kernel, KernelBackend, SincPath};
use crate::{ConvertMode, Quality, Result};

/// Immutable sample-rate conversion configuration.
#[derive(Clone)]
pub struct SrcManager {
    backend: KernelBackend,
}

/// Builds a [`SrcManager`].
#[derive(Default)]
pub struct SrcBuilder {
    kernel: Kernel,
    sinc_path: SincPath,
    sinc: SincBuilder,
}

impl SrcManager {
    /// Start a builder with default kernel [`Kernel::Sinc`] and [`SincPath::Auto`].
    #[inline]
    pub fn builder() -> SrcBuilder {
        SrcBuilder::default()
    }

    /// Build a **linear** converter from a conversion ratio.
    ///
    /// Sinc needs filter parameters (attenuation, `quantify`, path, and so on);
    /// use [`Self::builder`] instead.
    #[inline]
    pub fn with_ratio(ratio: f64) -> Result<Self> {
        Self::builder().ratio(ratio).kernel(Kernel::Linear).build()
    }

    /// Build a **linear** converter from integer sample rates.
    ///
    /// Sinc needs filter parameters; use [`Self::builder`] instead.
    #[inline]
    pub fn with_sample_rate(old_sr: u32, new_sr: u32) -> Result<Self> {
        Self::builder()
            .sample_rate(old_sr, new_sr)
            .kernel(Kernel::Linear)
            .build()
    }

    /// Create a streaming converter instance.
    #[inline]
    pub fn converter(&self) -> Converter {
        Converter::from_backend(&self.backend)
    }

    /// Convert a complete buffer, padding the end with zeros.
    pub fn convert(&self, input: &[f64]) -> Vec<f64> {
        self.backend.convert(input)
    }

    /// Conversion ratio `fs_new / fs_old`.
    #[inline]
    pub fn ratio(&self) -> f64 {
        self.backend.ratio()
    }

    /// Reduced integer ratio when a rational mode was selected.
    #[inline]
    pub fn ratio_parts(&self) -> Option<(i64, i64)> {
        self.backend.ratio_parts()
    }

    /// Which interpolation path converters from this manager will use.
    #[inline]
    pub fn mode(&self) -> ConvertMode {
        self.backend.mode()
    }

    /// Output latency in samples (zero for linear and cubic).
    #[inline]
    pub fn latency(&self) -> usize {
        self.backend.latency()
    }

    /// Expected output length for a complete input buffer.
    #[inline]
    pub fn output_len(&self, input_len: usize) -> usize {
        self.backend.output_len(input_len)
    }

    /// FIR order for sinc; `None` for linear and cubic.
    #[inline]
    pub fn order(&self) -> Option<u32> {
        self.backend.order()
    }

    /// LUT size for sinc; `None` for linear and cubic.
    #[inline]
    pub fn lut_len(&self) -> Option<usize> {
        self.backend.lut_len()
    }
}

impl SrcBuilder {
    /// Select conversion kernel.
    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Select sinc interpolation path. Ignored for [`Kernel::Linear`] and [`Kernel::Cubic`].
    #[inline]
    pub fn sinc_path(mut self, path: SincPath) -> Self {
        self.sinc_path = path;
        self
    }

    /// Set ratio in `[1/16, 16]`.
    #[inline]
    pub fn ratio(mut self, ratio: f64) -> Self {
        self.sinc = self.sinc.ratio(ratio);
        self
    }

    /// Set input and output sample rates.
    #[inline]
    pub fn sample_rate(mut self, old_sr: u32, new_sr: u32) -> Self {
        self.sinc = self.sinc.sample_rate(old_sr, new_sr);
        self
    }

    /// Set quantify for generic sinc (`[1, 16384]`).
    #[inline]
    pub fn quantify(mut self, quan: u32) -> Self {
        self.sinc = self.sinc.quantify(quan);
        self
    }

    /// Set FIR order (`[1, 2048]`).
    #[inline]
    pub fn order(mut self, order: u32) -> Self {
        self.sinc = self.sinc.order(order);
        self
    }

    /// Set Kaiser window beta (`[0, 20]`).
    #[inline]
    pub fn kaiser_beta<B: Into<f64>>(mut self, beta: B) -> Self {
        self.sinc = self.sinc.kaiser_beta(beta);
        self
    }

    /// Set normalized cutoff (`[0.01, 1.0]`).
    #[inline]
    pub fn cutoff(mut self, cutoff: f64) -> Self {
        self.sinc = self.sinc.cutoff(cutoff);
        self
    }

    /// Set stop-band attenuation in dB (`[12, 180]`).
    #[inline]
    pub fn attenuation<A: Into<f64>>(mut self, atten: A) -> Self {
        self.sinc = self.sinc.attenuation(atten);
        self
    }

    /// Set transition band width (`[0.01, 1.0]`).
    #[inline]
    pub fn trans_width(mut self, width: f64) -> Self {
        self.sinc = self.sinc.trans_width(width);
        self
    }

    /// Set pass-band width as a fraction of Nyquist (`[0, 0.99]`).
    #[inline]
    pub fn pass_width(mut self, width: f64) -> Self {
        self.sinc = self.sinc.pass_width(width);
        self
    }

    /// Set pass-band edge frequency in Hz.
    #[inline]
    pub fn pass_freq(mut self, freq: u32) -> Self {
        self.sinc = self.sinc.pass_freq(freq);
        self
    }

    /// Apply a quality preset (sinc only).
    #[inline]
    pub fn quality(mut self, quality: Quality) -> Self {
        self.sinc = self.sinc.quality(quality);
        self
    }

    /// Require generic half-table sinc interpolation.
    #[inline]
    pub fn generic(mut self) -> Self {
        self.sinc_path = SincPath::Generic;
        self
    }

    /// Require fast polyphase sinc interpolation.
    #[inline]
    pub fn fast(mut self) -> Self {
        self.sinc_path = SincPath::Fast;
        self
    }

    /// Build the manager.
    pub fn build(self) -> Result<SrcManager> {
        match self.kernel {
            Kernel::Linear => {
                let ratio = self.sinc.resolved_ratio()?;
                Ok(SrcManager {
                    backend: KernelBackend::linear(ratio),
                })
            }
            Kernel::Cubic => {
                let ratio = self.sinc.resolved_ratio()?;
                Ok(SrcManager {
                    backend: KernelBackend::cubic(ratio),
                })
            }
            Kernel::Sinc => {
                let backend = match self.sinc_path {
                    SincPath::Fast => self.sinc.fast().build()?,
                    SincPath::Generic => self.sinc.generic().build()?,
                    SincPath::Auto => {
                        let ratio = self.sinc.resolved_ratio()?;
                        if ratio.require_fast().is_ok() {
                            self.sinc.fast().build()?
                        } else {
                            self.sinc.generic().build()?
                        }
                    }
                };
                Ok(SrcManager {
                    backend: KernelBackend::sinc(backend),
                })
            }
        }
    }
}
