pub(crate) mod builder;
mod filter;

use std::sync::Arc;

use crate::{
    Convert, ConvertMode, Error, Ratio, Rational, Result, convert_with, engine::FirState,
    engine::FirTap, engine::PhaseAccum, engine::fir_next_sample, output_len,
};
use filter::*;

struct GenericFirConverter {
    phase: PhaseAccum,
    state: FirState,
    taps: FirTap,
    filter: Arc<Vec<f64>>,
    quan: f64,
    half_order: f64,
}

impl GenericFirConverter {
    fn new(step: PhaseAccum, order: u32, quan: u32, filter: Arc<Vec<f64>>) -> Self {
        Self {
            phase: step,
            state: FirState::new(),
            taps: FirTap::new((order + 1) as usize),
            filter,
            quan: quan as f64,
            half_order: 0.5 * order as f64,
        }
    }

    fn interpolate(
        phase: &PhaseAccum,
        taps: &FirTap,
        filter: &[f64],
        quan: f64,
        half_order: f64,
    ) -> f64 {
        let coef = phase.pos_float();
        let mut interp = 0.0;
        let pos_max = filter.len() - 1;
        let tap_count = taps.len();
        let iter_count = tap_count / 2;
        let mut left;
        let mut right;
        if tap_count % 2 == 1 {
            let pos = coef * quan;
            let posu = pos as usize;
            let h1 = filter[posu];
            let h2 = filter[posu + 1];
            let h = h1 + (h2 - h1) * (pos - posu as f64);
            interp += taps.get(iter_count) * h;
            left = iter_count - 1;
            right = iter_count + 1;
        } else {
            left = iter_count - 1;
            right = iter_count;
        }
        let coef = coef + half_order;
        for _ in 0..iter_count {
            let pos1 = (coef - left as f64).abs() * quan;
            let pos2 = (coef - right as f64).abs() * quan;
            let pos1u = pos1 as usize;
            let pos2u = pos2 as usize;
            if pos1u < pos_max {
                let h1 = filter[pos1u];
                let h2 = filter[pos1u + 1];
                let h = h1 + (h2 - h1) * (pos1 - pos1u as f64);
                interp += taps.get(left) * h;
            }
            if pos2u < pos_max {
                let h1 = filter[pos2u];
                let h2 = filter[pos2u + 1];
                let h = h1 + (h2 - h1) * (pos2 - pos2u as f64);
                interp += taps.get(right) * h;
            }
            left = left.wrapping_sub(1);
            right = right.wrapping_add(1);
        }
        interp
    }
}

struct RationalFastConverter {
    phase: PhaseAccum,
    state: FirState,
    taps: FirTap,
    lut: Arc<Vec<Vec<f64>>>,
}

impl RationalFastConverter {
    fn new(step: Rational, order: u32, lut: Arc<Vec<Vec<f64>>>) -> Self {
        Self {
            phase: PhaseAccum::rational(step),
            state: FirState::new(),
            taps: FirTap::new((order + 1) as usize),
            lut,
        }
    }

    fn interpolate(phase: &PhaseAccum, taps: &FirTap, lut: &[Vec<f64>]) -> f64 {
        lut[phase.pos_usize()]
            .iter()
            .zip(taps.iter())
            .map(|(h, s)| h * s)
            .sum()
    }
}

enum ConverterKind {
    Generic(GenericFirConverter),
    Fast(RationalFastConverter),
}

/// Opaque sample-rate converter created by [`Backend::converter`].
pub(crate) struct Converter {
    inner: ConverterKind,
}

impl Convert for GenericFirConverter {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
    {
        let filter = &self.filter;
        let quan = self.quan;
        let half_order = self.half_order;
        fir_next_sample(
            &mut self.state,
            &mut self.phase,
            &mut self.taps,
            iter,
            |phase, taps| Self::interpolate(phase, taps, filter, quan, half_order),
        )
    }
}

impl Convert for RationalFastConverter {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        let lut = &self.lut;
        fir_next_sample(
            &mut self.state,
            &mut self.phase,
            &mut self.taps,
            iter,
            |phase, taps| Self::interpolate(phase, taps, lut),
        )
    }
}

impl Converter {
    fn delay_empty(&self) -> bool {
        match &self.inner {
            ConverterKind::Generic(c) => c.taps.is_empty(),
            ConverterKind::Fast(c) => c.taps.is_empty(),
        }
    }
}

impl Convert for Converter {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        match &mut self.inner {
            ConverterKind::Generic(converter) => converter.next_sample(iter),
            ConverterKind::Fast(converter) => converter.next_sample(iter),
        }
    }

    fn flush(&mut self, output: &mut [f64]) -> usize
    where
        Self: Sized,
    {
        // Overrides Convert::flush: stop when the FIR delay is empty instead of
        // filling the whole buffer. Still call until 0 if `output` fills first.
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

fn check_u32(name: &'static str, value: u32, min: u32, max: u32) -> Result<()> {
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(Error::invalid(name, value as f64, min as f64, max as f64))
    }
}

fn check_f64(name: &'static str, value: f64, min: f64, max: f64) -> Result<()> {
    if value.is_finite() && (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(Error::invalid(name, value, min, max))
    }
}

fn trans_width_from_pass_freq(old_sr: u32, new_sr: u32, pass_freq: u32) -> f64 {
    let min_sr = new_sr.min(old_sr);
    min_sr.saturating_sub(pass_freq.saturating_mul(2)) as f64 / min_sr as f64
}

#[derive(Clone)]
enum Lut {
    Generic(Arc<Vec<f64>>),
    Fast(Arc<Vec<Vec<f64>>>),
}

#[derive(Clone)]
pub(crate) struct Backend {
    ratio: Ratio,
    order: u32,
    quan: u32,
    latency: usize,
    lut: Lut,
}

impl Backend {
    pub(crate) fn with_raw_internal(
        ratio: Ratio,
        quan: u32,
        order: u32,
        kaiser_beta: f64,
        cutoff: f64,
    ) -> Result<Self> {
        check_u32("quantify", quan, MIN_QUAN, MAX_QUAN)?;
        check_u32("order", order, MIN_ORDER, MAX_ORDER)?;
        check_f64("kaiser_beta", kaiser_beta, 0.0, 20.0)?;
        check_f64("cutoff", cutoff, 0.01, 1.0)?;
        let filter = generate_filter_table(quan, order, kaiser_beta, cutoff);
        let fratio = ratio.as_float();
        let latency = (fratio * order as f64 * 0.5).round() as usize;
        Ok(Self {
            ratio,
            order,
            quan,
            latency,
            lut: Lut::Generic(Arc::new(filter)),
        })
    }

    pub(crate) fn with_raw_fast_internal(
        ratio: Rational,
        order: u32,
        kaiser_beta: f64,
        cutoff: f64,
    ) -> Result<Self> {
        check_u32("order", order, MIN_ORDER, MAX_ORDER)?;
        check_f64("kaiser_beta", kaiser_beta, 0.0, 20.0)?;
        check_f64("cutoff", cutoff, 0.01, 1.0)?;
        let lut = generate_fast_lut(*ratio.numer() as usize, order, kaiser_beta, cutoff);
        let ratio = Ratio::Rational(ratio);
        let fratio = ratio.as_float();
        let latency = (fratio * order as f64 * 0.5).round() as usize;
        Ok(Self {
            ratio,
            order,
            quan: 0,
            latency,
            lut: Lut::Fast(Arc::new(lut)),
        })
    }

    pub(crate) fn new_internal(
        ratio: Ratio,
        atten: f64,
        quan: u32,
        trans_width: f64,
    ) -> Result<Self> {
        check_f64("attenuation", atten, MIN_ATTEN, MAX_ATTEN)?;
        check_u32("quantify", quan, MIN_QUAN, MAX_QUAN)?;
        check_f64("trans_width", trans_width, 0.01, 1.0)?;
        let kaiser_beta = calc_kaiser_beta(atten);
        let fratio = ratio.as_float();
        let order = calc_order(fratio, atten, trans_width);
        let cutoff = design_cutoff(fratio, trans_width);
        Self::with_raw_internal(ratio, quan, order, kaiser_beta, cutoff)
    }

    pub(crate) fn with_order_internal(
        ratio: Ratio,
        atten: f64,
        quan: u32,
        order: u32,
    ) -> Result<Self> {
        check_f64("attenuation", atten, MIN_ATTEN, MAX_ATTEN)?;
        check_u32("quantify", quan, MIN_QUAN, MAX_QUAN)?;
        check_u32("order", order, MIN_ORDER, MAX_ORDER)?;
        let fratio = ratio.as_float();
        let kaiser_beta = calc_kaiser_beta(atten);
        let trans_width = calc_trans_width(fratio, atten, order);
        let cutoff = design_cutoff(fratio, trans_width);
        Self::with_raw_internal(ratio, quan, order, kaiser_beta, cutoff)
    }

    pub(crate) fn fast_new_internal(ratio: Ratio, atten: f64, trans_width: f64) -> Result<Self> {
        check_f64("attenuation", atten, MIN_ATTEN, MAX_ATTEN)?;
        check_f64("trans_width", trans_width, 0.01, 1.0)?;
        let rational = ratio.require_fast()?;
        let kaiser_beta = calc_kaiser_beta(atten);
        let fratio = ratio.as_float();
        let order = calc_order(fratio, atten, trans_width);
        let cutoff = design_cutoff(fratio, trans_width);
        Self::with_raw_fast_internal(rational, order, kaiser_beta, cutoff)
    }

    pub(crate) fn fast_with_order_internal(ratio: Ratio, atten: f64, order: u32) -> Result<Self> {
        check_f64("attenuation", atten, MIN_ATTEN, MAX_ATTEN)?;
        check_u32("order", order, MIN_ORDER, MAX_ORDER)?;
        let rational = ratio.require_fast()?;
        let fratio = ratio.as_float();
        let kaiser_beta = calc_kaiser_beta(atten);
        let trans_width = calc_trans_width(fratio, atten, order);
        let cutoff = design_cutoff(fratio, trans_width);
        Self::with_raw_fast_internal(rational, order, kaiser_beta, cutoff)
    }

    /// Create a Generic `Manager` with sample rate, attenuation, quantify and pass frequency
    ///
    /// - old_sr: Old sample rate, not 0
    /// - new_sr: New sample rate, not 0
    /// - atten: `[12.0, 180.0]`
    /// - quan: `[1, 16384]`
    /// - pass_freq: pass-band frequency in Hz
    ///
    /// The sample rate ratio should be in `[1/16, 16]`. Always uses half-table
    /// interpolation. For a polyphase LUT, use
    /// [`fast_with_sample_rate`](Self::fast_with_sample_rate).
    #[inline]
    pub(crate) fn with_sample_rate(
        old_sr: u32,
        new_sr: u32,
        atten: f64,
        quan: u32,
        pass_freq: u32,
    ) -> Result<Self> {
        let ratio = Ratio::try_from_integers(new_sr, old_sr)?;
        let trans_width = trans_width_from_pass_freq(old_sr, new_sr, pass_freq);
        Self::new_internal(ratio, atten, quan, trans_width)
    }

    /// Create a Fast polyphase `Manager` from sample rates.
    ///
    /// Typical 44100/48000 conversions should use this with
    /// [`SrcBuilder::quality`](crate::SrcBuilder::quality).
    #[inline]
    pub(crate) fn fast_with_sample_rate(
        old_sr: u32,
        new_sr: u32,
        atten: f64,
        pass_freq: u32,
    ) -> Result<Self> {
        let ratio = Ratio::try_from_integers(new_sr, old_sr)?;
        let trans_width = trans_width_from_pass_freq(old_sr, new_sr, pass_freq);
        Self::fast_new_internal(ratio, atten, trans_width)
    }

    /// Create a `Converter` which actually implement the interpolation.
    #[inline]
    pub(crate) fn converter(&self) -> Converter {
        let inner = match (&self.ratio, &self.lut) {
            (Ratio::Float(ratio), Lut::Generic(filter)) => {
                ConverterKind::Generic(GenericFirConverter::new(
                    PhaseAccum::float(ratio.recip()),
                    self.order,
                    self.quan,
                    filter.clone(),
                ))
            }
            (Ratio::Rational(ratio), Lut::Generic(filter)) => {
                ConverterKind::Generic(GenericFirConverter::new(
                    PhaseAccum::rational(ratio.recip()),
                    self.order,
                    self.quan,
                    filter.clone(),
                ))
            }
            (Ratio::Rational(ratio), Lut::Fast(lut)) => ConverterKind::Fast(
                RationalFastConverter::new(ratio.recip(), self.order, lut.clone()),
            ),
            _ => unreachable!("LUT kind must match ratio representation"),
        };
        Converter { inner }
    }

    /// Get the latency of the FIR filter in output samples.
    #[inline]
    pub(crate) fn latency(&self) -> usize {
        self.latency
    }

    /// Get the order of the FIR filter.
    #[inline]
    pub(crate) fn order(&self) -> u32 {
        self.order
    }

    /// Conversion ratio `fs_new / fs_old` actually in use.
    ///
    /// For float constructors this is the approximated rational when one was
    /// accepted, otherwise the original float. See [`ConvertMode`].
    #[inline]
    pub(crate) fn ratio(&self) -> f64 {
        self.ratio.as_float()
    }

    /// Reduced integer ratio, if a rational mode was selected.
    ///
    /// `None` when the float could not be fit within the bounded continued-
    /// fraction rules (numerator/denominator ? 16384 and relative error
    /// ? `1e-12`). Integer sample-rate APIs always yield `Some`.
    #[inline]
    pub(crate) fn ratio_parts(&self) -> Option<(i64, i64)> {
        self.ratio.parts()
    }

    /// Which interpolation implementation this manager will construct.
    ///
    /// Float ratios become [`ConvertMode::Rational`] / [`ConvertMode::RationalFast`]
    /// only under the bounded approximation rules on [`ConvertMode`]; otherwise
    /// [`ConvertMode::Float`]. Fast LUT constructors still require an eligible
    /// rational and may return [`Error::FastUnavailable`].
    #[inline]
    pub(crate) fn mode(&self) -> ConvertMode {
        match self.lut {
            Lut::Fast(_) => ConvertMode::RationalFast,
            Lut::Generic(_) => match self.ratio {
                Ratio::Float(_) => ConvertMode::Float,
                Ratio::Rational(_) => ConvertMode::Rational,
            },
        }
    }

    /// Coefficient table length.
    ///
    /// Generic is the half Kaiser-sinc table length, including the
    /// interpolation pad. Fast is `numer * (order + 1)`.
    #[inline]
    pub(crate) fn lut_len(&self) -> usize {
        match &self.lut {
            Lut::Generic(filter) => filter.len(),
            Lut::Fast(lut) => lut.len() * (self.order as usize + 1),
        }
    }

    /// Expected output length for a complete input buffer of `input_len` samples.
    #[inline]
    pub(crate) fn output_len(&self, input_len: usize) -> usize {
        output_len(self.ratio(), input_len)
    }

    /// Convert a complete buffer.
    ///
    /// Pads the end with zeros and drops the leading FIR latency so the
    /// returned length is [`Self::output_len`].
    pub(crate) fn convert(&self, input: &[f64]) -> Vec<f64> {
        convert_with(self.converter(), self.latency, self.ratio(), input)
    }
}

#[cfg(test)]
mod tests {
    use super::builder::Builder;
    use super::filter::*;
    use super::*;
    use crate::{ConvertMode, Error, Quality};
    use std::f64::consts::PI;

    fn b() -> Builder {
        Builder::default()
    }

    #[test]
    fn test_manager_with_raw() {
        assert!(
            b().ratio(2.0)
                .quantify(32)
                .order(32)
                .kaiser_beta(5.0)
                .cutoff(0.8)
                .build()
                .is_ok()
        );
        assert!(
            b().ratio(2.0)
                .quantify(0)
                .order(32)
                .kaiser_beta(5.0)
                .cutoff(0.8)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .quantify(32)
                .order(0)
                .kaiser_beta(5.0)
                .cutoff(0.8)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .quantify(32)
                .order(32)
                .kaiser_beta(5.0)
                .cutoff(0.0)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .quantify(32)
                .order(32)
                .kaiser_beta(5.0)
                .cutoff(1.1)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .quantify(32)
                .order(32)
                .kaiser_beta(-0.1)
                .cutoff(0.8)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .quantify(32)
                .order(32)
                .kaiser_beta(20.1)
                .cutoff(0.8)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .fast()
                .order(32)
                .kaiser_beta(5.0)
                .cutoff(0.8)
                .build()
                .is_ok()
        );
    }

    #[test]
    fn test_manager_new() {
        assert!(
            b().ratio(2.0)
                .attenuation(72.0)
                .quantify(32)
                .trans_width(0.1)
                .build()
                .is_ok()
        );
        assert!(
            b().ratio(2.0)
                .attenuation(72.0)
                .quantify(0)
                .trans_width(0.1)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .attenuation(72.0)
                .quantify(32)
                .trans_width(0.0)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .attenuation(72.0)
                .quantify(32)
                .trans_width(1.1)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .attenuation(12.0)
                .quantify(32)
                .trans_width(0.1)
                .build()
                .is_ok()
        );
        assert!(
            b().ratio(2.0)
                .attenuation(11.9)
                .quantify(32)
                .trans_width(0.1)
                .build()
                .is_err()
        );
        let generic = b()
            .ratio(2.0)
            .attenuation(72.0)
            .quantify(32)
            .trans_width(0.1)
            .build()
            .unwrap();
        assert_eq!(generic.mode(), ConvertMode::Rational);
        assert_eq!(generic.lut_len(), generic_table_len(32, generic.order()));
    }

    #[test]
    fn test_manager_fast() {
        let fast = b()
            .ratio(2.0)
            .attenuation(72.0)
            .trans_width(0.1)
            .fast()
            .build()
            .unwrap();
        assert_eq!(fast.mode(), ConvertMode::RationalFast);
        assert_eq!(fast.lut_len(), 2 * (fast.order() as usize + 1));
        let sr = b()
            .sample_rate(44100, 48000)
            .attenuation(72.0)
            .pass_freq(20000)
            .fast()
            .build()
            .unwrap();
        assert_eq!(sr.mode(), ConvertMode::RationalFast);
        assert_eq!(sr.ratio_parts(), Some((160, 147)));
        assert!(matches!(
            b().sample_rate(1024, 1025)
                .attenuation(72.0)
                .pass_freq(400)
                .fast()
                .build(),
            Err(Error::FastUnavailable {
                numer: Some(1025),
                ..
            })
        ));
    }

    #[test]
    fn test_quality_generic_vs_fast() {
        let quality = Quality::Bit8Better;
        let trans_width = 0.1;
        let generic = b()
            .ratio(2.0)
            .quality(quality)
            .trans_width(trans_width)
            .build()
            .unwrap();
        let fast = b()
            .ratio(2.0)
            .quality(quality)
            .trans_width(trans_width)
            .fast()
            .build()
            .unwrap();
        assert_eq!(generic.order(), fast.order());
        assert_eq!(generic.mode(), ConvertMode::Rational);
        assert_eq!(fast.mode(), ConvertMode::RationalFast);
        assert_eq!(
            generic.lut_len(),
            generic_table_len(quality.quantify(), generic.order())
        );
    }

    #[test]
    fn test_manager_with_order() {
        assert!(
            b().ratio(2.0)
                .attenuation(72.0)
                .quantify(32)
                .order(32)
                .build()
                .is_ok()
        );
        assert!(
            b().ratio(2.0)
                .attenuation(72.0)
                .quantify(32)
                .order(0)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .attenuation(72.0)
                .quantify(0)
                .order(32)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .attenuation(12.0)
                .quantify(32)
                .order(32)
                .build()
                .is_ok()
        );
        assert!(
            b().ratio(2.0)
                .attenuation(11.9)
                .quantify(32)
                .order(32)
                .build()
                .is_err()
        );
        assert!(
            b().ratio(2.0)
                .attenuation(72.0)
                .order(32)
                .fast()
                .build()
                .is_ok()
        );
    }

    #[test]
    fn test_builder() {
        assert!(b().build().is_err());
        let manager = b()
            .sample_rate(44100, 48000)
            .quantify(32)
            .attenuation(72)
            .pass_freq(20000)
            .build();
        assert!(manager.is_ok());
        assert_eq!(manager.unwrap().mode(), ConvertMode::Rational);
        let fast = b()
            .sample_rate(44100, 48000)
            .attenuation(72)
            .pass_freq(20000)
            .fast()
            .build();
        assert!(fast.is_ok());
        assert_eq!(fast.unwrap().mode(), ConvertMode::RationalFast);
        let ignored_quan = b()
            .ratio(2.0)
            .quantify(32)
            .attenuation(72)
            .trans_width(0.1)
            .fast()
            .build()
            .unwrap();
        assert_eq!(ignored_quan.mode(), ConvertMode::RationalFast);
        assert!(b().ratio(0.0).quantify(8).build().is_err());
        let preset = b()
            .sample_rate(44100, 48000)
            .quality(Quality::Bit16Better)
            .pass_freq(20000)
            .build();
        assert!(preset.is_ok());
        assert_eq!(preset.as_ref().unwrap().ratio_parts(), Some((160, 147)));
        assert_eq!(preset.unwrap().mode(), ConvertMode::Rational);
    }

    #[test]
    fn inexact_ratio_uses_float_phase() {
        let manager = b()
            .ratio(PI)
            .quality(Quality::Bit8Fast)
            .trans_width(0.2)
            .build()
            .unwrap();
        assert_eq!(manager.mode(), ConvertMode::Float);
        assert_eq!(manager.ratio_parts(), None);
        assert!((manager.ratio() - PI).abs() < 1e-15);
        assert!(matches!(
            b().ratio(PI)
                .attenuation(48.0)
                .trans_width(0.2)
                .fast()
                .build(),
            Err(Error::FastUnavailable { numer: None, .. })
        ));
        let _ = manager.convert(&[1.0, 0.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn odd_order_odd_quantify_covers_half_table() {
        let odd = b()
            .ratio(2.0)
            .attenuation(48.0)
            .quantify(7)
            .order(5)
            .build()
            .unwrap();
        let even = b()
            .ratio(2.0)
            .attenuation(48.0)
            .quantify(8)
            .order(5)
            .build()
            .unwrap();
        assert_eq!(odd.lut_len(), generic_table_len(7, 5));
        assert_eq!(even.lut_len(), generic_table_len(8, 5));
        let input = vec![1.0; 64];
        let odd_out = odd.convert(&input);
        let even_out = even.convert(&input);
        let dc = |m: &Backend, out: &[f64]| {
            let start = m.latency().max(8);
            let end = out.len().saturating_sub(8).max(start + 1);
            let body = &out[start..end];
            body.iter().sum::<f64>() / body.len() as f64
        };
        let odd_dc = dc(&odd, &odd_out);
        let even_dc = dc(&even, &even_out);
        assert!(
            (odd_dc - even_dc).abs() < 0.02,
            "odd dc {odd_dc} vs even dc {even_dc}"
        );
    }

    #[test]
    fn flush_stops_when_delay_empty() {
        let manager = b()
            .ratio(2.0)
            .quality(Quality::Bit8Fast)
            .trans_width(0.2)
            .build()
            .unwrap();
        let mut cv = manager.converter();
        assert_eq!(cv.flush(&mut [0.0; 64]), 0);
        let input: Vec<f64> = (0..32).map(|i| (i as f64).sin()).collect();
        let mut tmp = [0.0; 64];
        let mut pos = 0;
        while pos < input.len() {
            let (c, _) = cv.process_block(&input[pos..], &mut tmp);
            if c == 0 {
                break;
            }
            pos += c;
        }
        let n = cv.flush(&mut [0.0; 4096]);
        assert!(n > 0);
        assert!(n < 4096, "flush should not fill a huge buffer, got {n}");
        assert_eq!(cv.flush(&mut [0.0; 64]), 0);
    }

    #[test]
    fn calc_order_adds_margin_and_is_even() {
        let without_margin = f64::ceil((96.0 - 8.0) / (2.285 * 0.1 * PI * 1.0)) as u32;
        let with_margin = calc_order(1.0, 96.0, 0.1);
        assert!(with_margin > without_margin);
        assert_eq!(with_margin % 2, 0);
        assert!(with_margin <= MAX_ORDER);
        // Explicit order path is unchanged by the margin helper.
        assert_eq!(
            calc_trans_width(1.0, 96.0, without_margin),
            (96.0 - 8.0) / (2.285 * without_margin as f64 * PI)
        );
    }

    #[test]
    fn design_cutoff_puts_transition_below_nyquist() {
        let ratio = 44100.0 / 48000.0;
        let tw = 0.1;
        let c = design_cutoff(ratio, tw);
        let nyq = ratio.min(1.0);
        assert!((c - nyq * (1.0 - tw)).abs() < 1e-15);
        // Stop edge of a Kaiser band centered on cutoff is below Nyquist.
        let stop = c + 0.5 * tw * nyq;
        assert!(stop < nyq + 1e-15);
    }

    #[test]
    fn normalized_dc_gain_is_near_unity() {
        for (ratio, quality, tw) in [
            (2.0, Quality::Bit8Fast, 0.2),
            (0.5, Quality::Bit8Fast, 0.2),
            (2.0, Quality::Bit16Fast, 0.1),
            (48000.0 / 44100.0, Quality::Bit16Fast, 0.1),
        ] {
            let generic = b()
                .ratio(ratio)
                .quality(quality)
                .trans_width(tw)
                .build()
                .unwrap();
            let fast = b()
                .ratio(ratio)
                .quality(quality)
                .trans_width(tw)
                .fast()
                .build()
                .unwrap();
            assert_eq!(generic.order() % 2, 0);
            assert_eq!(fast.order() % 2, 0);
            for (label, m) in [("generic", generic), ("fast", fast)] {
                let out = m.convert(&vec![1.0; 512]);
                let start = m.latency().max(32);
                let end = out.len().saturating_sub(32).max(start + 1);
                let avg = out[start..end].iter().sum::<f64>() / (end - start) as f64;
                assert!(
                    (avg - 1.0).abs() < 1e-3,
                    "{label} ratio={ratio} quality={quality:?} dc={avg}"
                );
            }
        }
    }
}
