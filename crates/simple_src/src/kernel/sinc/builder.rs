use super::Backend;
use crate::{Error, Quality, Ratio, Result};

/// The Builder to build `Manager`
///
/// Defaults to Generic interpolation (`quantify` is required). Call
/// [`.fast()`](Builder::fast) for a polyphase LUT; then `quantify` is ignored
/// and an ineligible ratio returns [`Error::FastUnavailable`].
///
/// ```
/// use simple_src::{Kernel, Quality, SrcBuilder, SrcManager};
///
/// let manager = SrcManager::builder()
///     .sample_rate(44100, 48000)
///     .quantify(32)
///     .attenuation(72)
///     .pass_freq(20000)
///     .build();
/// assert!(manager.is_ok());
/// ```
#[derive(Default)]
pub(crate) struct Builder {
    ratio: Option<Ratio>,
    ratio_error: Option<Error>,
    order: Option<u32>,
    quan: Option<u32>,
    kaiser_beta: Option<f64>,
    cutoff: Option<f64>,
    atten: Option<f64>,
    trans_width: Option<f64>,
    old_sr: Option<u32>,
    new_sr: Option<u32>,
    pass_freq: Option<u32>,
    use_fast: bool,
    /// Measure-and-trim the Kaiser design at build time (see
    /// `filter::trim_design`).
    trim: bool,
}

impl Builder {
    /// Set `ratio` in `[1/16, 16]`.
    ///
    /// May reduce to a bounded rational (see [`ConvertMode`]); use
    /// [`Self::sample_rate`] for an exact integer rate pair.
    pub(crate) fn ratio(mut self, ratio: f64) -> Self {
        match Ratio::try_from_float(ratio) {
            Ok(r) => {
                self.ratio = Some(r);
                self.ratio_error = None;
            }
            Err(e) => self.ratio_error = Some(e),
        }
        self
    }

    /// Set old sample rate and new sample rate
    pub(crate) fn sample_rate(mut self, old_sr: u32, new_sr: u32) -> Self {
        self.old_sr = Some(old_sr);
        self.new_sr = Some(new_sr);
        self
    }

    /// Set quantify number in `[1, 16384]`.
    ///
    /// Required for Generic. Ignored after [`.fast()`](Self::fast).
    pub(crate) fn quantify(mut self, quan: u32) -> Self {
        self.quan = Some(quan);
        self
    }

    /// Set order of filter in `[1, 2048]`
    pub(crate) fn order(mut self, order: u32) -> Self {
        self.order = Some(order);
        self
    }

    /// Set beta of kaiser window function in `[0, 20]`
    pub(crate) fn kaiser_beta<B: Into<f64>>(mut self, beta: B) -> Self {
        self.kaiser_beta = Some(beta.into());
        self
    }

    /// Set cutoff of filter in `[0.01, 1.0]`
    pub(crate) fn cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = Some(cutoff);
        self
    }

    /// Set attenuation of stop band in `[12, 180]`
    pub(crate) fn attenuation<A: Into<f64>>(mut self, atten: A) -> Self {
        self.atten = Some(atten.into());
        self
    }

    /// Set transition band width in `[0.01, 1.0]`
    pub(crate) fn trans_width(mut self, width: f64) -> Self {
        self.trans_width = Some(width);
        self
    }

    /// Set pass band width in `[0, 0.99]`
    pub(crate) fn pass_width(mut self, width: f64) -> Self {
        self.trans_width = Some(1.0 - width);
        self
    }

    /// Set pass band frequency in Hz, the calculated transition band width
    /// should not less than 0.01
    pub(crate) fn pass_freq(mut self, freq: u32) -> Self {
        self.pass_freq = Some(freq);
        self
    }

    /// Set attenuation and quantify from a [`Quality`] preset.
    ///
    /// After [`.fast()`](Self::fast), only attenuation is used; quantify is
    /// ignored.
    pub(crate) fn quality(mut self, quality: Quality) -> Self {
        self.atten = Some(quality.attenuation());
        self.quan = Some(quality.quantify());
        self
    }

    /// Build a Fast polyphase LUT. `quantify` is not required and is ignored
    /// if set. Ineligible ratios return [`Error::FastUnavailable`].
    pub(crate) fn fast(mut self) -> Self {
        self.use_fast = true;
        self
    }

    /// Build Generic half-table interpolation (the default). `quantify` is
    /// required.
    pub(crate) fn generic(mut self) -> Self {
        self.use_fast = false;
        self
    }

    /// Replace the `+6 dB` order margin and the approximate Kaiser
    /// beta mapping with an init-time search that measures the realized
    /// stopband of the worst polyphase branch and picks the smallest order
    /// that meets `attenuation` exactly. Applies only to the
    /// attenuation-based constructors.
    pub(crate) fn trimmed(mut self, enable: bool) -> Self {
        self.trim = enable;
        self
    }

    pub(crate) fn resolved_ratio(&self) -> Result<Ratio> {
        self.ratio_error.clone().map_or(Ok(()), Err)?;
        match (self.ratio, self.old_sr, self.new_sr) {
            (Some(ratio), _, _) => Ok(ratio),
            (_, Some(old_sr), Some(new_sr)) => Ratio::try_from_integers(new_sr, old_sr),
            _ => Err(Error::missing("ratio or sample_rate")),
        }
    }

    /// Build the `Manager`, there are the following combinations in order:
    ///
    /// Generic (default; `quantify` required):
    ///
    /// - ratio, quantify, order, kaiser_beta, cutoff
    /// - ratio, attenuation, quantify, trans_width or pass_width
    /// - ratio, attenuation, quantify, order
    /// - sample_rate, attenuation, quantify, pass_freq
    ///
    /// Fast ([`.fast()`](Self::fast); `quantify` ignored):
    ///
    /// - ratio, order, kaiser_beta, cutoff
    /// - ratio, attenuation, trans_width or pass_width
    /// - ratio, attenuation, order
    /// - sample_rate, attenuation, pass_freq
    ///
    /// For example, this is the first Generic situation:
    ///
    /// ```
    /// use simple_src::{Kernel, Quality, SrcBuilder, SrcManager};
    ///
    /// let manager = SrcManager::builder()
    ///     .ratio(0.5)
    ///     .quantify(32)
    ///     .order(32)
    ///     .kaiser_beta(7.0)
    ///     .cutoff(0.8)
    ///     .build();
    /// assert!(manager.is_ok());
    /// ```
    pub(crate) fn build(self) -> Result<Backend> {
        if self.use_fast {
            return self.build_fast();
        }
        let ratio = self.resolved_ratio()?;
        let Some(quan) = self.quan else {
            return Err(Error::missing("quantify"));
        };
        match (
            self.order,
            self.kaiser_beta,
            self.cutoff,
            self.atten,
            self.trans_width,
            self.old_sr,
            self.new_sr,
            self.pass_freq,
        ) {
            (Some(order), Some(kaiser_beta), Some(cutoff), _, _, _, _, _) => {
                super::Backend::with_raw_internal(ratio, quan, order, kaiser_beta, cutoff)
            }
            (_, _, _, Some(atten), Some(trans_width), _, _, _) => {
                if self.trim {
                    super::Backend::trimmed_new_internal(ratio, atten, quan, trans_width)
                } else {
                    super::Backend::new_internal(ratio, atten, quan, trans_width)
                }
            }
            (Some(order), _, _, Some(atten), _, _, _, _) => {
                super::Backend::with_order_internal(ratio, atten, quan, order)
            }
            (_, _, _, Some(atten), _, Some(old_sr), Some(new_sr), Some(pass_freq)) => {
                if self.trim {
                    super::Backend::trimmed_with_sample_rate(old_sr, new_sr, atten, quan, pass_freq)
                } else {
                    super::Backend::with_sample_rate(old_sr, new_sr, atten, quan, pass_freq)
                }
            }
            _ => Err(Error::missing(
                "attenuation with trans_width/order/pass_freq, or raw cutoff",
            )),
        }
    }

    fn build_fast(self) -> Result<Backend> {
        let ratio = self.resolved_ratio()?;
        match (
            self.order,
            self.kaiser_beta,
            self.cutoff,
            self.atten,
            self.trans_width,
            self.old_sr,
            self.new_sr,
            self.pass_freq,
        ) {
            (Some(order), Some(kaiser_beta), Some(cutoff), _, _, _, _, _) => {
                let rational = ratio.require_fast()?;
                super::Backend::with_raw_fast_internal(rational, order, kaiser_beta, cutoff)
            }
            (_, _, _, Some(atten), Some(trans_width), _, _, _) => {
                if self.trim {
                    super::Backend::fast_trimmed_new_internal(ratio, atten, trans_width)
                } else {
                    super::Backend::fast_new_internal(ratio, atten, trans_width)
                }
            }
            (Some(order), _, _, Some(atten), _, _, _, _) => {
                super::Backend::fast_with_order_internal(ratio, atten, order)
            }
            (_, _, _, Some(atten), _, Some(old_sr), Some(new_sr), Some(pass_freq)) => {
                if self.trim {
                    super::Backend::fast_trimmed_with_sample_rate(old_sr, new_sr, atten, pass_freq)
                } else {
                    super::Backend::fast_with_sample_rate(old_sr, new_sr, atten, pass_freq)
                }
            }
            _ => Err(Error::missing(
                "attenuation with trans_width/order/pass_freq, or raw cutoff",
            )),
        }
    }
}
