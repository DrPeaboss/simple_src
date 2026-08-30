pub(crate) mod builder;
mod dot;
mod fft;
mod filter;
use dot::{DotFn, dot_scalar, select_dot};

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
    /// Flat polyphase row table: `(quan + 1)` rows of `order + 1`
    /// coefficients. One output sample is the lerp of the dot products with
    /// rows `b` and `b + 1`, where `b = floor(frac * quan)` -- the algebraic
    /// transform of per-tap lerped 1-D table lookups, which turns the
    /// scattered lookup work into two dense dot products (shared with the
    /// Fast path kernels).
    rows: Arc<Vec<f64>>,
    stride: usize,
    /// Row count (`quan + 1`).
    phases: usize,
    /// Phases per unit distance as f64 (exact: `quan <= 16384`).
    quan: f64,
    /// Dot-product kernel selected once at build time (AVX2+FMA where
    /// available, portable auto-vectorized fallback otherwise).
    dot: DotFn,
}

impl GenericFirConverter {
    fn new(step: PhaseAccum, order: u32, rows: Arc<Vec<f64>>, dot: DotFn) -> Self {
        let stride = order as usize + 1;
        Self {
            phase: step,
            state: FirState::new(),
            taps: FirTap::new(stride),
            phases: rows.len() / stride,
            stride,
            quan: (rows.len() / stride - 1) as f64,
            rows,
            dot,
        }
    }

    /// One output sample from phase `b` (row index) blended with `t` toward
    /// row `b + 1`; `t` may be exactly 1.0 when `x` rounded up to `phases - 1`.
    #[inline]
    fn interpolate(&self, b: usize, t: f64) -> f64 {
        let (tap_a, tap_b) = self.taps.slices();
        let la = tap_a.len();
        let r0 = b * self.stride;
        let r1 = r0 + self.stride;
        let rows = &self.rows[..];
        // SAFETY: `self.dot` was selected by `select_dot` for this CPU.
        unsafe {
            let d0 = dot2(self.dot, rows, r0, la, self.stride, tap_a, tap_b);
            let d1 = dot2(self.dot, rows, r1, la, self.stride, tap_a, tap_b);
            d0 + (d1 - d0) * t
        }
    }

    /// Streaming batch loop for the Running state: monomorphic sample loop
    /// with the same state transitions and operation order as the per-sample
    /// `fir_next_sample` path (verified bit-identical by an in-crate test).
    pub(crate) fn process_block(&mut self, input: &[f64], output: &mut [f64]) -> (usize, usize) {
        if output.is_empty() {
            // Matches the shared per-sample batch helper: nothing requested,
            // nothing consumed.
            return (0, 0);
        }
        let mut iter = crate::SliceIter {
            data: input,
            pos: 0,
        };
        let mut produced = 0;
        // Resume from a previous input-exhausted suspension with the same
        // behavior as `fir_next_sample`: shift one input sample, then fall
        // through to the Running loop.
        if matches!(self.state, FirState::Suspended) {
            match iter.next() {
                Some(s) => {
                    self.taps.shift(s);
                    self.state = FirState::Running;
                }
                None => return (iter.pos, produced),
            }
        }
        let quan = self.quan;
        let phases = self.phases;
        while produced < output.len() {
            if self.phase.needs_input_advance() {
                self.phase.consume_input_step();
                match iter.next() {
                    Some(s) => self.taps.shift(s),
                    None => {
                        self.state = FirState::Suspended;
                        return (iter.pos, produced);
                    }
                }
                continue;
            }
            let x = self.phase.pos_float() * quan;
            let b = (x as usize).min(phases - 2);
            output[produced] = self.interpolate(b, x - b as f64);
            produced += 1;
            self.phase.advance_output();
        }
        (iter.pos, produced)
    }
}

/// Two dot calls over one row split at the delay-line ring boundary.
///
/// # Safety
/// `dot` must be valid for this CPU (see `select_dot`); the row starting at
/// `base` must have `stride` elements available in `rows`.
#[inline]
unsafe fn dot2(
    dot: DotFn,
    rows: &[f64],
    base: usize,
    la: usize,
    stride: usize,
    tap_a: &[f64],
    tap_b: &[f64],
) -> f64 {
    // SAFETY (edition 2024): caller guarantees the `dot` kernel matches the
    // CPU and that `rows[base..base + stride]` is in bounds.
    unsafe { dot(tap_a, &rows[base..base + la]) + dot(tap_b, &rows[base + la..base + stride]) }
}

struct RationalFastConverter {
    phase: PhaseAccum,
    state: FirState,
    taps: FirTap,
    /// Flat polyphase table: `phases` rows of `stride` coefficients.
    lut: Arc<Vec<f64>>,
    stride: usize,
    /// Dot-product kernel selected once at build time (AVX2+FMA where
    /// available, portable auto-vectorized fallback otherwise).
    dot: DotFn,
}

impl RationalFastConverter {
    fn new(step: Rational, order: u32, lut: Arc<Vec<f64>>, dot: DotFn) -> Self {
        Self {
            phase: PhaseAccum::rational(step),
            state: FirState::new(),
            taps: FirTap::new((order + 1) as usize),
            stride: order as usize + 1,
            lut,
            dot,
        }
    }

    /// One output sample from the current phase; `pos` must be `< phases`.
    #[inline]
    fn interpolate(&self, pos: usize) -> f64 {
        let (tap_a, tap_b) = self.taps.slices();
        let row = &self.lut[pos * self.stride..(pos + 1) * self.stride];
        // SAFETY: `self.dot` was selected by `select_dot` for this CPU.
        unsafe { (self.dot)(tap_a, &row[..tap_a.len()]) + (self.dot)(tap_b, &row[tap_a.len()..]) }
    }

    /// Streaming batch loop for the Running state: the phase arithmetic is
    /// monomorphic (no `PhaseAccum` enum dispatch per output) and each output
    /// is one dot-kernel call. Semantics are bit-identical to `next_sample`
    /// (same state transitions, same operation order).
    pub(crate) fn process_block(&mut self, input: &[f64], output: &mut [f64]) -> (usize, usize) {
        if output.is_empty() {
            // Matches the shared per-sample batch helper: nothing requested,
            // nothing consumed.
            return (0, 0);
        }
        let mut iter = crate::SliceIter {
            data: input,
            pos: 0,
        };
        let mut produced = 0;
        // Resume from a previous input-exhausted suspension with the same
        // behavior as `fir_next_sample`: shift one input sample, then fall
        // through to the Running loop.
        if matches!(self.state, FirState::Suspended) {
            match iter.next() {
                Some(s) => {
                    self.taps.shift(s);
                    self.state = FirState::Running;
                }
                None => return (iter.pos, produced),
            }
        }
        // Running state: copy the phase fields out and run a tight,
        // monomorphic loop (no `PhaseAccum` dispatch per output sample).
        let PhaseAccum::Rational { pos, numer, denom } = &mut self.phase else {
            unreachable!("fast sinc uses the Rational phase");
        };
        let (mut pos, numer, denom) = (*pos, *numer, *denom);
        while produced < output.len() {
            while pos >= denom {
                pos -= denom;
                match iter.next() {
                    Some(s) => self.taps.shift(s),
                    None => {
                        self.state = FirState::Suspended;
                        *pos_ref(&mut self.phase) = pos;
                        return (iter.pos, produced);
                    }
                }
            }
            output[produced] = self.interpolate(pos);
            produced += 1;
            pos += numer;
        }
        *pos_ref(&mut self.phase) = pos;
        (iter.pos, produced)
    }
}

#[inline]
fn pos_ref(phase: &mut PhaseAccum) -> &mut usize {
    match phase {
        PhaseAccum::Rational { pos, .. } => pos,
        _ => unreachable!("fast sinc uses the Rational phase"),
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
        Self: Sized,
    {
        let rows = &self.rows;
        let stride = self.stride;
        let phases = self.phases;
        let quan = self.quan;
        let dot = self.dot;
        fir_next_sample(
            &mut self.state,
            &mut self.phase,
            &mut self.taps,
            iter,
            move |phase, taps| {
                let x = phase.pos_float() * quan;
                let b = (x as usize).min(phases - 2);
                let t = x - b as f64;
                let (tap_a, tap_b) = taps.slices();
                let la = tap_a.len();
                let r0 = b * stride;
                let r1 = r0 + stride;
                // SAFETY: `dot` was selected by `select_dot` for this CPU.
                unsafe {
                    let d0 = dot2(dot, rows, r0, la, stride, tap_a, tap_b);
                    let d1 = dot2(dot, rows, r1, la, stride, tap_a, tap_b);
                    d0 + (d1 - d0) * t
                }
            },
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
        let stride = self.stride;
        let dot = self.dot;
        fir_next_sample(
            &mut self.state,
            &mut self.phase,
            &mut self.taps,
            iter,
            |phase, taps| {
                let (tap_a, tap_b) = taps.slices();
                let pos = phase.pos_usize();
                let row = &lut[pos * stride..(pos + 1) * stride];
                // SAFETY: `dot` was selected by `select_dot` for this CPU.
                unsafe { dot(tap_a, &row[..tap_a.len()]) + dot(tap_b, &row[tap_a.len()..]) }
            },
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

    /// Batch override: the Fast path runs a dedicated monomorphic loop; the
    /// Generic path keeps the shared per-sample batch helper.
    pub(crate) fn process_block(&mut self, input: &[f64], output: &mut [f64]) -> (usize, usize)
    where
        Self: Sized,
    {
        match &mut self.inner {
            ConverterKind::Fast(c) => c.process_block(input, output),
            ConverterKind::Generic(c) => c.process_block(input, output),
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
    /// Flat polyphase table: `phases` rows of `(order + 1)` coefficients.
    Fast(Arc<Vec<f64>>),
}

#[derive(Clone)]
pub(crate) struct Backend {
    ratio: Ratio,
    order: u32,
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
        let rows = generate_generic_rows(quan, order, kaiser_beta, cutoff);
        let fratio = ratio.as_float();
        let latency = (fratio * order as f64 * 0.5).round() as usize;
        Ok(Self {
            ratio,
            order,
            latency,
            lut: Lut::Generic(Arc::new(rows)),
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

    /// Measured-trim variant of [`Self::new_internal`]: replaces the
    /// `+6 dB` order margin and the approximate beta mapping with a search
    /// for the smallest order whose worst polyphase branch meets the
    /// requested stopband. Falls back to the formula design when the search
    /// cannot converge within `MAX_ORDER`.
    pub(crate) fn trimmed_new_internal(
        ratio: Ratio,
        atten: f64,
        quan: u32,
        trans_width: f64,
    ) -> Result<Self> {
        check_f64("attenuation", atten, MIN_ATTEN, MAX_ATTEN)?;
        check_u32("quantify", quan, MIN_QUAN, MAX_QUAN)?;
        check_f64("trans_width", trans_width, 0.01, 1.0)?;
        let fratio = ratio.as_float();
        let cutoff = design_cutoff(fratio, trans_width);
        let (order, kaiser_beta) = match trim_design(fratio, atten, trans_width) {
            Some(design) => design,
            None => (
                calc_order(fratio, atten, trans_width),
                calc_kaiser_beta(atten),
            ),
        };
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

    /// Measured-trim variant of [`Self::fast_new_internal`].
    pub(crate) fn fast_trimmed_new_internal(
        ratio: Ratio,
        atten: f64,
        trans_width: f64,
    ) -> Result<Self> {
        check_f64("attenuation", atten, MIN_ATTEN, MAX_ATTEN)?;
        check_f64("trans_width", trans_width, 0.01, 1.0)?;
        let rational = ratio.require_fast()?;
        let fratio = ratio.as_float();
        let cutoff = design_cutoff(fratio, trans_width);
        let (order, kaiser_beta) = match trim_design(fratio, atten, trans_width) {
            Some(design) => design,
            None => (
                calc_order(fratio, atten, trans_width),
                calc_kaiser_beta(atten),
            ),
        };
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

    /// Measured-trim variant of [`Self::with_sample_rate`].
    #[inline]
    pub(crate) fn trimmed_with_sample_rate(
        old_sr: u32,
        new_sr: u32,
        atten: f64,
        quan: u32,
        pass_freq: u32,
    ) -> Result<Self> {
        let ratio = Ratio::try_from_integers(new_sr, old_sr)?;
        let trans_width = trans_width_from_pass_freq(old_sr, new_sr, pass_freq);
        Self::trimmed_new_internal(ratio, atten, quan, trans_width)
    }

    /// Measured-trim variant of [`Self::fast_with_sample_rate`].
    #[inline]
    pub(crate) fn fast_trimmed_with_sample_rate(
        old_sr: u32,
        new_sr: u32,
        atten: f64,
        pass_freq: u32,
    ) -> Result<Self> {
        let ratio = Ratio::try_from_integers(new_sr, old_sr)?;
        let trans_width = trans_width_from_pass_freq(old_sr, new_sr, pass_freq);
        Self::fast_trimmed_new_internal(ratio, atten, trans_width)
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
        self.converter_forced(false)
    }

    /// Converter with an optionally forced portable dot kernel. Used by tests
    /// and the hidden `internal-bench` benchmark hook to exercise the scalar
    /// fallback end to end on AVX2-capable machines.
    #[inline]
    pub(crate) fn converter_forced(&self, force_scalar: bool) -> Converter {
        let dot = if force_scalar {
            dot_scalar
        } else {
            select_dot()
        };
        let inner = match (&self.ratio, &self.lut) {
            (Ratio::Float(ratio), Lut::Generic(rows)) => {
                ConverterKind::Generic(GenericFirConverter::new(
                    PhaseAccum::float(ratio.recip()),
                    self.order,
                    rows.clone(),
                    dot,
                ))
            }
            (Ratio::Rational(ratio), Lut::Generic(rows)) => {
                ConverterKind::Generic(GenericFirConverter::new(
                    PhaseAccum::rational(ratio.recip()),
                    self.order,
                    rows.clone(),
                    dot,
                ))
            }
            (Ratio::Rational(ratio), Lut::Fast(lut)) => ConverterKind::Fast(
                RationalFastConverter::new(ratio.recip(), self.order, lut.clone(), dot),
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
    /// Generic is `(quan + 1) * (order + 1)`: polyphase rows including the
    /// fractional-phase row. Fast is `numer * (order + 1)`.
    #[inline]
    pub(crate) fn lut_len(&self) -> usize {
        match &self.lut {
            Lut::Generic(rows) => rows.len(),
            Lut::Fast(lut) => lut.len(),
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
        assert_eq!(generic.lut_len(), (32 + 1) * (generic.order() as usize + 1));
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
            (quality.quantify() as usize + 1) * (generic.order() as usize + 1)
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
        assert_eq!(odd.lut_len(), (7 + 1) * 6);
        assert_eq!(even.lut_len(), (8 + 1) * 6);
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

    /// End-to-end check that the scalar dot-kernel fallback produces the
    /// same output as the runtime-selected kernel (AVX2 on this machine,
    /// scalar otherwise) through the full pipeline. On non-AVX2 CPUs both
    /// converters use the same kernel and the test degenerates to a
    /// self-comparison, which is fine.
    ///
    /// Together with the spectral baselines (which run the selected kernel)
    /// this transitively covers the fallback's numerical quality: outputs
    /// agree to float reassociation noise.
    #[test]
    fn forced_scalar_pipeline_matches_selected_kernel() {
        let run = |mgr: &crate::SrcManager, force_scalar: bool| {
            let input: Vec<f64> = (0..5000)
                .map(|i| ((i as f64) * 0.013).sin() + 0.3 * ((i as f64) * 0.107).cos())
                .collect();
            let mut cv = mgr.converter_forced_kernel(force_scalar);
            let mut out = Vec::new();
            let mut pos = 0;
            let mut buf = vec![0.0f64; 1024];
            while pos < input.len() {
                let (consumed, produced) = cv.process_block(&input[pos..], &mut buf);
                if consumed == 0 && produced == 0 {
                    break;
                }
                pos += consumed;
                out.extend_from_slice(&buf[..produced]);
            }
            let mut tail = vec![0.0f64; 8192];
            loop {
                let n = cv.flush(&mut tail);
                if n == 0 {
                    break;
                }
                out.extend_from_slice(&tail[..n]);
            }
            out
        };

        let cases = [
            // (builder, label): Fast (a96), Generic rational, Generic float
            (
                crate::SrcManager::builder()
                    .ratio(48000.0 / 44100.0)
                    .attenuation(96.0)
                    .trans_width(0.05)
                    .fast()
                    .build(),
                "fast",
            ),
            (
                crate::SrcManager::builder()
                    .ratio(48000.0 / 44100.0)
                    .attenuation(72.0)
                    .quantify(32)
                    .trans_width(0.05)
                    .generic()
                    .build(),
                "generic rational",
            ),
            (
                crate::SrcManager::builder()
                    .ratio(f64::sqrt(2.0))
                    .attenuation(72.0)
                    .quantify(32)
                    .trans_width(0.05)
                    .generic()
                    .build(),
                "generic float",
            ),
        ];
        for (built, label) in cases {
            let mgr = built.unwrap();
            let scalar = run(&mgr, true);
            let selected = run(&mgr, false);
            let n = scalar.len().min(selected.len());
            assert!(n > 1000, "{label}: too few samples {n}");
            let maxdiff = scalar[..n]
                .iter()
                .zip(&selected[..n])
                .map(|(a, b)| (a - b).abs() / a.abs().max(1e-12))
                .fold(0.0f64, f64::max);
            assert!(
                maxdiff < 1e-12,
                "{label}: forced scalar vs selected kernel maxrel {maxdiff}"
            );
        }
    }

    /// The forced-scalar converter's batch loop must stay bit-identical to
    /// its own per-sample iterator path (covers the fallback's process_block).
    #[test]
    fn forced_scalar_batch_matches_iterator() {
        let mgr = crate::SrcManager::builder()
            .ratio(48000.0 / 44100.0)
            .attenuation(96.0)
            .trans_width(0.05)
            .fast()
            .build()
            .unwrap();
        let input: Vec<f64> = (0..3000).map(|i| ((i as f64) * 0.021).sin()).collect();
        let iter_out: Vec<f64> = {
            let mut cv = mgr.converter_forced_kernel(true);
            cv.process(input.iter().copied())
                .take(mgr.output_len(input.len()))
                .collect()
        };
        let mut batch_out = Vec::new();
        let mut pos = 0;
        let mut buf = vec![0.0f64; 256];
        let mut cv = mgr.converter_forced_kernel(true);
        while pos < input.len() {
            let (consumed, produced) = cv.process_block(&input[pos..], &mut buf);
            if consumed == 0 && produced == 0 {
                break;
            }
            pos += consumed;
            batch_out.extend_from_slice(&buf[..produced]);
        }
        let lat = mgr.latency();
        let a = &iter_out[lat.min(iter_out.len())..];
        let b = &batch_out[lat.min(batch_out.len())..];
        let n = a.len().min(b.len());
        assert!(n > 500);
        assert_eq!(&a[..n], &b[..n], "forced scalar batch vs iterator");
    }

    /// The polyphase row table must reproduce the per-tap lerped 1-D table
    /// (the old `interpolate`) to floating-point reassociation accuracy, for
    /// odd and even tap counts and random fractional phases.
    #[test]
    fn generic_rows_match_direct_lerp() {
        let beta = 6.0;
        let cutoff = 0.5;
        for (quan, order) in [(8u32, 5u32), (7, 5), (32, 12), (128, 96), (3, 1)] {
            let table = generate_filter_table(quan, order, beta, cutoff);
            let rows = generate_generic_rows(quan, order, beta, cutoff);
            let taps_n = order as usize + 1;
            let half = taps_n / 2;
            let half_order = 0.5 * order as f64;
            let pos_max = table.len() - 1;
            let q = quan as f64;
            let last_real = table.len() - 2;
            for i in 0..64 {
                let frac = (i as f64 * 0.618_033_988_749_894_9) % 1.0;
                let taps: Vec<f64> = (0..taps_n)
                    .map(|k| ((k * 7 + i) as f64 * 0.13).sin())
                    .collect();

                // Direct evaluation replicating the old per-tap algorithm.
                let direct = || {
                    let mut acc = 0.0;
                    for (j, &tap) in taps.iter().enumerate() {
                        let d = if j < half {
                            (half - j) as f64 + frac
                        } else {
                            (j - half) as f64 - frac
                        }
                        .abs();
                        let pos = d * q;
                        let posu = pos as usize;
                        if posu < pos_max || d <= half_order {
                            let h = if posu < pos_max {
                                table[posu] + (table[posu + 1] - table[posu]) * (pos - posu as f64)
                            } else {
                                table[last_real]
                            };
                            acc += tap * h;
                        }
                    }
                    acc
                };

                // Row-table evaluation (what the new converters compute).
                let x = frac * q;
                let b = (x as usize).min(quan as usize);
                let t = x - b as f64;
                let r0 = &rows[b * taps_n..(b + 1) * taps_n];
                let r1 = &rows[(b + 1) * taps_n..(b + 2) * taps_n];
                let got: f64 = taps
                    .iter()
                    .zip(r0)
                    .zip(r1)
                    .map(|((tp, a), bb)| tp * ((1.0 - t) * a + t * bb))
                    .sum();

                let want = direct();
                let scale = want.abs().max(1e-12);
                assert!(
                    (got - want).abs() / scale < 1e-12,
                    "quan {quan} order {order} i {i}: {got} vs {want}"
                );
            }
        }
    }

    /// process_block (batch loop) must be bit-identical to the per-sample
    /// iterator path when both use the same dot kernel.
    #[test]
    fn fast_batch_matches_iterator_bitwise() {
        for (old_sr, new_sr) in [(44100u32, 48000u32), (48000, 44100), (48000, 96000)] {
            let mgr = crate::SrcManager::builder()
                .sample_rate(old_sr, new_sr)
                .fast()
                .quality(crate::Quality::Bit16Fast)
                .trans_width(0.05)
                .build()
                .unwrap();
            let input: Vec<f64> = (0..5000)
                .map(|i| ((i as f64) * 0.013).sin() + 0.3 * ((i as f64) * 0.107).cos())
                .collect();

            // iterator path, one sample at a time
            let mut cv = mgr.converter();
            let iter_out: Vec<f64> = cv
                .process(input.iter().copied())
                .take(mgr.output_len(input.len()))
                .collect();

            // batch path, 10ms-ish chunks with a final flush drain
            let mut cv = mgr.converter();
            let mut batch_out = Vec::new();
            let mut pos = 0;
            let mut buf = vec![0.0f64; 1024];
            while pos < input.len() {
                let (consumed, produced) = cv.process_block(&input[pos..], &mut buf);
                if consumed == 0 && produced == 0 {
                    break;
                }
                pos += consumed;
                batch_out.extend_from_slice(&buf[..produced]);
            }
            let mut tail = vec![0.0f64; 8192];
            loop {
                let n = cv.flush(&mut tail);
                if n == 0 {
                    break;
                }
                batch_out.extend_from_slice(&tail[..n]);
            }

            // convert() drops latency; drop it from the streaming paths too
            let lat = mgr.latency();
            let iter_cmp = &iter_out[lat.min(iter_out.len())..];
            let batch_cmp = &batch_out[lat.min(batch_out.len())..];
            let n = iter_cmp.len().min(batch_cmp.len());
            assert!(
                iter_cmp[..n] == batch_cmp[..n],
                "batch vs iterator mismatch for {old_sr}->{new_sr}"
            );
        }
    }

    /// The flat LUT must still report the expected phase count.
    #[test]
    fn fast_lut_len_is_phase_count() {
        let mgr = crate::SrcManager::builder()
            .sample_rate(44100, 48000)
            .fast()
            .quality(crate::Quality::Bit16Fast)
            .trans_width(0.05)
            .build()
            .unwrap();
        let order = mgr.order().unwrap() as usize;
        assert_eq!(mgr.lut_len().unwrap(), 160 * (order + 1));
    }

    /// process_block (batch loop) must be bit-identical to the per-sample
    /// iterator path for the Generic converter, for Float and Rational
    /// phases alike (both use the same dot kernels and state transitions).
    #[test]
    fn generic_batch_matches_iterator_bitwise() {
        // 2.0 exercises the Rational phase; sqrt2 (approximation rejected by
        // the bounded continued-fraction rules) exercises the Float phase.
        for ratio in [2.0, f64::sqrt(2.0)] {
            let mgr = crate::SrcManager::builder()
                .ratio(ratio)
                .attenuation(72.0)
                .quantify(32)
                .trans_width(0.1)
                .generic()
                .build()
                .unwrap();
            let input: Vec<f64> = (0..4000)
                .map(|i| ((i as f64) * 0.013).sin() + 0.3 * ((i as f64) * 0.107).cos())
                .collect();

            // Iterator path, one sample at a time.
            let mut cv = mgr.converter();
            let iter_out: Vec<f64> = cv
                .process(input.iter().copied())
                .take(mgr.output_len(input.len()))
                .collect();

            // Batch path, chunks plus a flush drain.
            let mut cv = mgr.converter();
            let mut batch_out = Vec::new();
            let mut pos = 0;
            let mut buf = vec![0.0f64; 512];
            while pos < input.len() {
                let (consumed, produced) = cv.process_block(&input[pos..], &mut buf);
                if consumed == 0 && produced == 0 {
                    break;
                }
                pos += consumed;
                batch_out.extend_from_slice(&buf[..produced]);
            }
            let mut tail = vec![0.0f64; 8192];
            loop {
                let n = cv.flush(&mut tail);
                if n == 0 {
                    break;
                }
                batch_out.extend_from_slice(&tail[..n]);
            }

            // `convert` drops latency; drop it from the streaming paths too.
            let lat = mgr.latency();
            let iter_cmp = &iter_out[lat.min(iter_out.len())..];
            let batch_cmp = &batch_out[lat.min(batch_out.len())..];
            let n = iter_cmp.len().min(batch_cmp.len());
            assert!(
                iter_cmp[..n] == batch_cmp[..n],
                "batch vs iterator mismatch for ratio {ratio}"
            );
        }
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
