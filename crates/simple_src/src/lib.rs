//! Simple sample rate conversion lib.
//!
//! ## Usage
//!
//! Use [`SrcManager`] to create converters. Select [`Kernel::Linear`] or
//! [`Kernel::Sinc`] (default). Sinc converters have FIR latency; prefer
//! [`SrcManager::convert`] for complete buffers, or skip [`SrcManager::latency`]
//! samples when streaming and call [`Convert::flush`] after the last input.
//!
//! Float ratios may be reduced to a rational when a continued-fraction fit has
//! numerator and denominator ≤ 16384 and relative error ≤ `1e-12`. Prefer
//! [`SrcBuilder::sample_rate`] for exact integer rate pairs such as 44100/48000.
//!
//! Multi-channel audio uses N independent mono converters. Keep planar buffers
//! frame-aligned with [`process_planar`] / [`flush_planar`].

mod converter;
mod engine;
mod kernel;
mod manager;
mod quality;
mod ratio;

pub use converter::Converter;
pub use kernel::{Kernel, SincPath};
pub use manager::{SrcBuilder, SrcManager};
pub use quality::Quality;
use ratio::{Ratio, Rational};

/// Interpolation implementation selected from the conversion ratio.
///
/// For float inputs, a rational mode is chosen only when a continued-fraction
/// approximation has both terms ≤ 16384 and relative error ≤ `1e-12`. Exact
/// integer sample-rate pairs always keep their reduced rational.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvertMode {
    /// Floating-point phase increment. Used when no bounded rational
    /// approximation meets the error limit (for example `π`).
    Float,
    /// Integer phase, coefficients computed or interpolated at run time.
    Rational,
    /// Integer phase with a fully precomputed coefficient table.
    RationalFast,
}

pub struct ConvertIter<'a, I, C> {
    iter: I,
    cvtr: &'a mut C,
}

impl<'a, I, C> ConvertIter<'a, I, C> {
    #[inline]
    pub fn new(iter: I, cvtr: &'a mut C) -> Self {
        Self { iter, cvtr }
    }
}

impl<I, C> Iterator for ConvertIter<'_, I, C>
where
    I: Iterator<Item = f64>,
    C: Convert,
{
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.cvtr.next_sample(&mut self.iter)
    }
}

pub trait Convert {
    /// Get the next sample converted, return `None` until the input samples is
    /// not enough.
    ///
    /// Note that the output can be continued after `None` returned.
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized;

    /// Process samples and return an iterator, can be called multiple times.
    fn process<I>(&mut self, iter: I) -> ConvertIter<'_, I, Self>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        ConvertIter::new(iter, self)
    }

    /// Convert as many samples as fit in `output`, consuming from `input`.
    ///
    /// Returns `(consumed, produced)`.
    fn process_block(&mut self, input: &[f64], output: &mut [f64]) -> (usize, usize)
    where
        Self: Sized,
    {
        let mut iter = SliceIter {
            data: input,
            pos: 0,
        };
        let mut produced = 0;
        while produced < output.len() {
            match self.next_sample(&mut iter) {
                Some(sample) => {
                    output[produced] = sample;
                    produced += 1;
                }
                None => break,
            }
        }
        (iter.pos, produced)
    }

    /// Continue converting by feeding zeros, writing into `output`.
    ///
    /// Call this after the last input block to drain FIR delay (or the last
    /// linear interval). Returns the number of samples written.
    ///
    /// A single call may not finish draining if `output` fills first. Keep
    /// calling until the return value is 0 (and provide a fresh buffer each
    /// time). A converter that has not yet seen any input writes nothing.
    ///
    /// **Default implementation:** fills `output` by repeatedly calling
    /// [`Self::next_sample`] with zeros until the slice is full or
    /// `next_sample` returns `None`. It does **not** detect an empty delay
    /// line, so custom `Convert` types that rely on this default will keep
    /// writing for the whole buffer. Built-in converters override `flush` to
    /// stop once their delay line is empty (still call until 0 if the buffer
    /// was too small).
    fn flush(&mut self, output: &mut [f64]) -> usize
    where
        Self: Sized,
    {
        let mut zeros = std::iter::repeat(0.0);
        let mut produced = 0;
        while produced < output.len() {
            match self.next_sample(&mut zeros) {
                Some(sample) => {
                    output[produced] = sample;
                    produced += 1;
                }
                None => break,
            }
        }
        produced
    }
}

struct SliceIter<'a> {
    data: &'a [f64],
    pos: usize,
}

impl Iterator for SliceIter<'_> {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let sample = *self.data.get(self.pos)?;
        self.pos += 1;
        Some(sample)
    }
}

/// Process planar channel buffers in lockstep.
///
/// Each converter is independent (no stereo mixing). The returned
/// `(consumed, produced)` applies to every channel.
///
/// The caller must still pass matching layouts: one buffer per converter, and
/// the same input/output length on every channel. Mismatches return
/// [`Error::MismatchedChannels`] or [`Error::MismatchedLength`] instead of
/// panicking. Converters should come from the same manager and stay in
/// lockstep; if they have drifted, this returns [`Error::UnalignedConverters`]
/// after some channels may already have advanced.
pub fn process_planar<C: Convert>(
    converters: &mut [C],
    inputs: &[&[f64]],
    outputs: &mut [&mut [f64]],
) -> Result<(usize, usize)> {
    check_channel_count(converters.len(), inputs.len(), "input")?;
    check_channel_count(converters.len(), outputs.len(), "output")?;
    check_equal_channel_lens(inputs.iter().map(|s| s.len()), "input")?;
    check_equal_channel_lens(outputs.iter().map(|s| s.len()), "output")?;
    let mut consumed = 0;
    let mut produced = 0;
    for (i, converter) in converters.iter_mut().enumerate() {
        let (c, p) = converter.process_block(inputs[i], outputs[i]);
        if i == 0 {
            consumed = c;
            produced = p;
        } else if c != consumed || p != produced {
            return Err(Error::UnalignedConverters { channel: i });
        }
    }
    Ok((consumed, produced))
}

/// Flush planar converters in lockstep. See [`Convert::flush`].
///
/// The caller must pass one output buffer per converter, all of equal length.
/// Like [`Convert::flush`], call again until the returned count is 0 if the
/// buffers may be shorter than the remaining delay. See [`process_planar`]
/// for the error cases.
pub fn flush_planar<C: Convert>(converters: &mut [C], outputs: &mut [&mut [f64]]) -> Result<usize> {
    check_channel_count(converters.len(), outputs.len(), "output")?;
    check_equal_channel_lens(outputs.iter().map(|s| s.len()), "output")?;
    let mut produced = 0;
    for (i, converter) in converters.iter_mut().enumerate() {
        let p = converter.flush(outputs[i]);
        if i == 0 {
            produced = p;
        } else if p != produced {
            return Err(Error::UnalignedConverters { channel: i });
        }
    }
    Ok(produced)
}

fn check_channel_count(expected: usize, actual: usize, what: &'static str) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::MismatchedChannels {
            expected,
            actual,
            what,
        })
    }
}

fn check_equal_channel_lens(
    mut lens: impl Iterator<Item = usize>,
    what: &'static str,
) -> Result<()> {
    let Some(expected) = lens.next() else {
        return Ok(());
    };
    for (i, actual) in lens.enumerate() {
        if actual != expected {
            return Err(Error::MismatchedLength {
                channel: i + 1,
                expected,
                actual,
                what,
            });
        }
    }
    Ok(())
}

pub(crate) fn convert_with<C: Convert>(
    mut converter: C,
    latency: usize,
    ratio: f64,
    input: &[f64],
) -> Vec<f64> {
    let out_len = output_len(ratio, input.len());
    converter
        .process(input.iter().copied().chain(std::iter::repeat(0.0)))
        .skip(latency)
        .take(out_len)
        .collect()
}

pub(crate) fn output_len(ratio: f64, input_len: usize) -> usize {
    (ratio * input_len as f64).round() as usize
}

#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    UnsupportedRatio {
        ratio: f64,
    },
    InvalidParam {
        name: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    MissingParam(&'static str),
    MismatchedChannels {
        expected: usize,
        actual: usize,
        what: &'static str,
    },
    MismatchedLength {
        channel: usize,
        expected: usize,
        actual: usize,
        what: &'static str,
    },
    UnalignedConverters {
        channel: usize,
    },
    FastUnavailable {
        ratio: f64,
        numer: Option<i64>,
    },
}

impl Error {
    pub(crate) fn unsupported(ratio: f64) -> Self {
        Self::UnsupportedRatio { ratio }
    }

    pub(crate) fn invalid(name: &'static str, value: f64, min: f64, max: f64) -> Self {
        Self::InvalidParam {
            name,
            value,
            min,
            max,
        }
    }

    pub(crate) fn missing(name: &'static str) -> Self {
        Self::MissingParam(name)
    }

    pub(crate) fn fast_unavailable(ratio: f64, numer: Option<i64>) -> Self {
        Self::FastUnavailable { ratio, numer }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedRatio { ratio } => {
                write!(f, "unsupported conversion ratio {ratio}")
            }
            Self::InvalidParam {
                name,
                value,
                min,
                max,
            } => write!(
                f,
                "invalid parameter {name}={value}, expected [{min}, {max}]"
            ),
            Self::MissingParam(name) => {
                write!(
                    f,
                    "not enough parameters to build converter, missing {name}"
                )
            }
            Self::MismatchedChannels {
                expected,
                actual,
                what,
            } => write!(
                f,
                "planar {what} count is {actual}, expected {expected} converters"
            ),
            Self::MismatchedLength {
                channel,
                expected,
                actual,
                what,
            } => write!(
                f,
                "planar {what} length mismatch: channel 0 has {expected}, channel {channel} has {actual}"
            ),
            Self::UnalignedConverters { channel } => {
                write!(
                    f,
                    "planar converters are not in lockstep at channel {channel}"
                )
            }
            Self::FastUnavailable { ratio, numer } => match numer {
                Some(numer) => write!(
                    f,
                    "fast polyphase converter is unavailable for ratio {ratio} (numerator {numer} > 1024); use SrcBuilder::sinc_path(SincPath::Generic)"
                ),
                None => write!(
                    f,
                    "fast polyphase converter is unavailable for ratio {ratio}; use SrcBuilder::sinc_path(SincPath::Generic)"
                ),
            },
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Kernel;

    #[test]
    fn with_ratio_is_linear_only() {
        let linear = SrcManager::with_ratio(2.0).unwrap();
        assert_eq!(linear.latency(), 0);
        assert_eq!(linear.order(), None);

        match SrcManager::builder().ratio(2.0).build() {
            Ok(_) => panic!("sinc builder without filter params should fail"),
            Err(e) => {
                let text = e.to_string();
                assert!(text.contains("missing"), "{text}");
            }
        }
    }

    #[test]
    fn error_display() {
        let err = Error::invalid("quantify", 0.0, 1.0, 16384.0);
        let text = err.to_string();
        assert!(text.contains("quantify"));
        assert!(text.contains("0"));
        let fast = Error::fast_unavailable(std::f64::consts::PI, None);
        let text = fast.to_string();
        assert!(text.contains("Generic"));
        assert!(matches!(
            Error::fast_unavailable(1025.0 / 1024.0, Some(1025)),
            Error::FastUnavailable {
                numer: Some(1025),
                ..
            }
        ));
    }

    #[test]
    fn process_block_and_flush_linear() {
        let manager = SrcManager::with_ratio(2.0).unwrap();
        let mut cv = manager.converter();
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 8];
        let (consumed, produced) = cv.process_block(&input, &mut output);
        assert_eq!(consumed, 4);
        assert!(produced <= 8);
        let extra = cv.flush(&mut output[produced..]);
        assert!(extra <= 8 - produced);
        assert!(produced + extra >= 6);
        let mut rest = [0.0; 16];
        let n = cv.flush(&mut rest);
        assert!(n < rest.len());
        assert_eq!(cv.flush(&mut rest), 0);
    }

    #[test]
    fn convert_matches_process_skip() {
        let manager = SrcManager::builder()
            .ratio(2.0)
            .kernel(Kernel::Sinc)
            .generic()
            .attenuation(48.0)
            .quantify(8)
            .trans_width(0.1)
            .build()
            .unwrap();
        let input = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let via_helper = manager.convert(&input);
        let mut cv = manager.converter();
        let via_iter: Vec<_> = cv
            .process(input.iter().copied().chain(std::iter::repeat(0.0)))
            .skip(manager.latency())
            .take(manager.output_len(input.len()))
            .collect();
        assert_eq!(via_helper.len(), via_iter.len());
        for (a, b) in via_helper.iter().zip(via_iter.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn planar_lockstep() {
        let manager = SrcManager::with_ratio(2.0).unwrap();
        let mut converters = [manager.converter(), manager.converter()];
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [4.0, 3.0, 2.0, 1.0];
        let mut out_l = [0.0; 8];
        let mut out_r = [0.0; 8];
        let inputs: [&[f64]; 2] = [&left, &right];
        let mut outputs: [&mut [f64]; 2] = [&mut out_l, &mut out_r];
        let (consumed, produced) = process_planar(&mut converters, &inputs, &mut outputs).unwrap();
        assert_eq!(consumed, 4);
        assert!(produced > 0);
        let mut tail_l = [0.0; 4];
        let mut tail_r = [0.0; 4];
        let mut tails: [&mut [f64]; 2] = [&mut tail_l, &mut tail_r];
        let flushed = flush_planar(&mut converters, &mut tails).unwrap();
        assert!(flushed > 0);
        assert!(flushed <= 4);
        assert_eq!(flush_planar(&mut converters, &mut tails).unwrap(), 0);
    }

    #[test]
    fn process_planar_rejects_channel_count_mismatch() {
        let manager = SrcManager::with_ratio(2.0).unwrap();
        let mut converters = [manager.converter(), manager.converter()];
        let left = [1.0, 2.0];
        let mut out_l = [0.0; 4];
        let mut out_r = [0.0; 4];
        let inputs: [&[f64]; 1] = [&left];
        let mut outputs: [&mut [f64]; 2] = [&mut out_l, &mut out_r];
        let err = process_planar(&mut converters, &inputs, &mut outputs).unwrap_err();
        assert!(matches!(
            err,
            Error::MismatchedChannels {
                expected: 2,
                actual: 1,
                what: "input"
            }
        ));
    }

    #[test]
    fn process_planar_rejects_unequal_input_lens() {
        let manager = SrcManager::with_ratio(2.0).unwrap();
        let mut converters = [manager.converter(), manager.converter()];
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [4.0, 3.0];
        let mut out_l = [0.0; 8];
        let mut out_r = [0.0; 8];
        let inputs: [&[f64]; 2] = [&left, &right];
        let mut outputs: [&mut [f64]; 2] = [&mut out_l, &mut out_r];
        let err = process_planar(&mut converters, &inputs, &mut outputs).unwrap_err();
        assert!(matches!(
            err,
            Error::MismatchedLength {
                channel: 1,
                expected: 4,
                actual: 2,
                what: "input"
            }
        ));
    }
}
