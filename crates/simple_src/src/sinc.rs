//! Sinc interpolation converter
//!
//! ## Simple way
//!
//! ```
//! use simple_src::{sinc, Convert};
//!
//! let samples = vec![1.0, 2.0, 3.0, 4.0];
//! let manager = sinc::Manager::new(2.0, 48.0, 8, 0.1).unwrap();
//! let mut converter = manager.converter();
//! for s in converter.process(samples.into_iter()) {
//!     println!("{s}");
//! }
//! ```
//!
//! Generic constructors (`new`, `with_quality`, `with_sample_rate`) always use
//! half-table interpolation. For a polyphase LUT, call `fast` /
//! `fast_with_quality` / `fast_with_sample_rate` (or builder `.fast()`).
//!
//! ## Builder way
//!
//! ```
//! use simple_src::{sinc, Convert};
//!
//! let samples = vec![1.0, 2.0, 3.0, 4.0];
//! let manager = sinc::Manager::builder()
//!     .ratio(2.0)
//!     .attenuation(48.0)
//!     .quantify(8)
//!     .pass_width(0.9)
//!     .build()
//!     .unwrap();
//! let mut converter = manager.converter();
//! for s in converter.process(samples.into_iter()) {
//!     println!("{s}");
//! }
//! ```

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::Arc;

use super::{
    Convert, ConvertMode, Error, Quality, Ratio, Rational, Result, convert_with, output_len,
};

#[inline]
fn sinc_c(x: f64, cutoff: f64) -> f64 {
    if x != 0.0 {
        (PI * x * cutoff).sin() / (PI * x)
    } else {
        cutoff
    }
}

#[inline]
fn bessel_i0(x: f64) -> f64 {
    let mut y = 1.0;
    let mut t = 1.0;
    for k in 1..32 {
        t *= (x / (2.0 * k as f64)).powi(2);
        y += t;
        if t < 1e-10 {
            break;
        }
    }
    y
}

#[inline]
fn generate_filter_table(quan: u32, order: u32, beta: f64, cutoff: f64) -> Vec<f64> {
    let len = order * quan / 2;
    let i0_beta = bessel_i0(beta);
    let half_order = order as f64 * 0.5;
    let mut filter = Vec::with_capacity(len as usize + 1);
    for i in 0..len {
        let pos = i as f64 / quan as f64;
        let i0_1 = bessel_i0(beta * (1.0 - (pos / half_order).powi(2)).sqrt());
        let coef = sinc_c(pos, cutoff) * (i0_1 / i0_beta);
        filter.push(coef);
    }
    filter.push(0.0);
    filter
}

#[inline]
fn generate_fast_lut(len: usize, order: u32, beta: f64, cutoff: f64) -> Vec<Vec<f64>> {
    let mut lut = Vec::with_capacity(len);
    let i0_beta = bessel_i0(beta);
    let half_order = order as f64 * 0.5;
    let taps = order + 1;
    for i in 0..len {
        let pos = i as f64 / len as f64;
        let mut coef_pos = Vec::with_capacity(taps as usize);
        for j in (0..taps).rev() {
            let pos = pos + j as f64 - half_order;
            let coef = if (-half_order..=half_order).contains(&pos) {
                let i0_1 = bessel_i0(beta * (1.0 - (pos / half_order).powi(2)).sqrt());
                sinc_c(pos, cutoff) * (i0_1 / i0_beta)
            } else {
                0.0
            };
            coef_pos.push(coef);
        }
        lut.push(coef_pos);
    }
    lut
}

#[inline]
fn calc_kaiser_beta(atten: f64) -> f64 {
    if atten > 50.0 {
        0.1102 * (atten - 8.7)
    } else if atten >= 21.0 {
        0.5842 * (atten - 21.0).powf(0.4) + 0.07886 * (atten - 21.0)
    } else {
        0.0
    }
}

#[inline]
fn calc_trans_width(ratio: f64, atten: f64, order: u32) -> f64 {
    (atten - 8.0) / (2.285 * order as f64 * PI * ratio.min(1.0))
}

#[inline]
fn calc_order(ratio: f64, atten: f64, trans_width: f64) -> u32 {
    f64::ceil((atten - 8.0) / (2.285 * trans_width * PI * ratio.min(1.0))) as u32
}

enum State {
    Normal,
    Suspend,
}

pub(crate) struct FloatConverter {
    state: State,
    buf: VecDeque<f64>,
    filter: Arc<Vec<f64>>,
    quan: f64,
    half_order: f64,
    step: f64,
    pos: f64,
}

pub(crate) struct RationalConverter {
    state: State,
    buf: VecDeque<f64>,
    filter: Arc<Vec<f64>>,
    quan: f64,
    half_order: f64,
    pos: usize,
    numer: usize,
    denom: usize,
    coefs: Vec<f64>,
}

pub(crate) struct RationalFastConverter {
    state: State,
    buf: VecDeque<f64>,
    pos: usize,
    numer: usize,
    denom: usize,
    lut: Arc<Vec<Vec<f64>>>,
}

enum ConverterKind {
    Float(FloatConverter),
    Rational(RationalConverter),
    RationalFast(RationalFastConverter),
}

/// Opaque sample-rate converter created by [`Manager::converter`].
pub struct Converter {
    inner: ConverterKind,
}

impl FloatConverter {
    fn new(step: f64, order: u32, quan: u32, filter: Arc<Vec<f64>>) -> Self {
        let taps = (order + 1) as usize;
        let mut buf = VecDeque::with_capacity(taps);
        buf.extend(std::iter::repeat_n(0.0, taps));
        Self {
            state: State::Normal,
            buf,
            filter,
            quan: quan as f64,
            half_order: 0.5 * order as f64,
            pos: 0.0,
            step,
        }
    }

    fn interpolate(&self) -> f64 {
        let coef = self.pos;
        let mut interp = 0.0;
        let pos_max = self.filter.len() - 1;
        let taps = self.buf.len();
        let iter_count = taps / 2;
        let mut left;
        let mut right;
        if taps % 2 == 1 {
            let pos = coef * self.quan;
            let posu = pos as usize;
            let h1 = self.filter[posu];
            let h2 = self.filter[posu + 1];
            let h = h1 + (h2 - h1) * (pos - posu as f64);
            interp += self.buf[iter_count] * h;
            left = iter_count - 1;
            right = iter_count + 1;
        } else {
            left = iter_count - 1;
            right = iter_count;
        }
        let coef = coef + self.half_order;
        for _ in 0..iter_count {
            let pos1 = (coef - left as f64).abs() * self.quan;
            let pos2 = (coef - right as f64).abs() * self.quan;
            let pos1u = pos1 as usize;
            let pos2u = pos2 as usize;
            if pos1u < pos_max {
                let h1 = self.filter[pos1u];
                let h2 = self.filter[pos1u + 1];
                let h = h1 + (h2 - h1) * (pos1 - pos1u as f64);
                interp += self.buf[left] * h;
            }
            if pos2u < pos_max {
                let h1 = self.filter[pos2u];
                let h2 = self.filter[pos2u + 1];
                let h = h1 + (h2 - h1) * (pos2 - pos2u as f64);
                interp += self.buf[right] * h;
            }
            left = left.wrapping_sub(1);
            right = right.wrapping_add(1);
        }
        interp
    }
}

impl RationalConverter {
    fn new(step: Rational, order: u32, quan: u32, filter: Arc<Vec<f64>>) -> Self {
        let numer = *step.numer() as usize;
        let denom = *step.denom() as usize;
        let mut coefs = Vec::with_capacity(denom);
        for i in 0..denom {
            coefs.push(i as f64 / denom as f64);
        }
        let taps = (order + 1) as usize;
        let mut buf = VecDeque::with_capacity(taps);
        buf.extend(std::iter::repeat_n(0.0, taps));
        Self {
            state: State::Normal,
            buf,
            filter,
            quan: quan as f64,
            half_order: 0.5 * order as f64,
            pos: 0,
            numer,
            denom,
            coefs,
        }
    }

    fn interpolate(&self) -> f64 {
        let coef = self.coefs[self.pos];
        let mut interp = 0.0;
        let pos_max = self.filter.len() - 1;
        let taps = self.buf.len();
        let iter_count = taps / 2;
        let mut left;
        let mut right;
        if taps % 2 == 1 {
            let pos = coef * self.quan;
            let posu = pos as usize;
            let h1 = self.filter[posu];
            let h2 = self.filter[posu + 1];
            let h = h1 + (h2 - h1) * (pos - posu as f64);
            interp += self.buf[iter_count] * h;
            left = iter_count - 1;
            right = iter_count + 1;
        } else {
            left = iter_count - 1;
            right = iter_count;
        }
        let coef = coef + self.half_order;
        for _ in 0..iter_count {
            let pos1 = (coef - left as f64).abs() * self.quan;
            let pos2 = (coef - right as f64).abs() * self.quan;
            let pos1u = pos1 as usize;
            let pos2u = pos2 as usize;
            if pos1u < pos_max {
                let h1 = self.filter[pos1u];
                let h2 = self.filter[pos1u + 1];
                let h = h1 + (h2 - h1) * (pos1 - pos1u as f64);
                interp += self.buf[left] * h;
            }
            if pos2u < pos_max {
                let h1 = self.filter[pos2u];
                let h2 = self.filter[pos2u + 1];
                let h = h1 + (h2 - h1) * (pos2 - pos2u as f64);
                interp += self.buf[right] * h;
            }
            left = left.wrapping_sub(1);
            right = right.wrapping_add(1);
        }
        interp
    }
}

impl RationalFastConverter {
    fn new(step: Rational, order: u32, lut: Arc<Vec<Vec<f64>>>) -> Self {
        let taps = (order + 1) as usize;
        let mut buf = VecDeque::with_capacity(taps);
        buf.extend(std::iter::repeat_n(0.0, taps));
        Self {
            state: State::Normal,
            buf,
            pos: 0,
            numer: *step.numer() as usize,
            denom: *step.denom() as usize,
            lut,
        }
    }

    fn interpolate(&self) -> f64 {
        self.lut[self.pos]
            .iter()
            .zip(self.buf.iter())
            .map(|(h, s)| h * s)
            .sum()
    }
}

impl Convert for FloatConverter {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
    {
        loop {
            match self.state {
                State::Normal => {
                    while self.pos >= 1.0 {
                        self.pos -= 1.0;
                        if let Some(s) = iter.next() {
                            self.buf.pop_front();
                            self.buf.push_back(s);
                        } else {
                            self.state = State::Suspend;
                            return None;
                        }
                    }
                    let interp = self.interpolate();
                    self.pos += self.step;
                    return Some(interp);
                }
                State::Suspend => {
                    let s = iter.next()?;
                    self.buf.pop_front();
                    self.buf.push_back(s);
                    self.state = State::Normal;
                }
            }
        }
    }
}

impl Convert for RationalConverter {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
    {
        loop {
            match self.state {
                State::Normal => {
                    while self.pos >= self.denom {
                        self.pos -= self.denom;
                        if let Some(s) = iter.next() {
                            self.buf.pop_front();
                            self.buf.push_back(s);
                        } else {
                            self.state = State::Suspend;
                            return None;
                        }
                    }
                    let interp = self.interpolate();
                    self.pos += self.numer;
                    return Some(interp);
                }
                State::Suspend => {
                    let s = iter.next()?;
                    self.buf.pop_front();
                    self.buf.push_back(s);
                    self.state = State::Normal;
                }
            }
        }
    }
}

impl Convert for RationalFastConverter {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        loop {
            match self.state {
                State::Normal => {
                    while self.pos >= self.denom {
                        self.pos -= self.denom;
                        if let Some(s) = iter.next() {
                            self.buf.pop_front();
                            self.buf.push_back(s);
                        } else {
                            self.state = State::Suspend;
                            return None;
                        }
                    }
                    let interp = self.interpolate();
                    self.pos += self.numer;
                    return Some(interp);
                }
                State::Suspend => {
                    let s = iter.next()?;
                    self.buf.pop_front();
                    self.buf.push_back(s);
                    self.state = State::Normal;
                }
            }
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
            ConverterKind::Float(float_converter) => float_converter.next_sample(iter),
            ConverterKind::Rational(rational_converter) => rational_converter.next_sample(iter),
            ConverterKind::RationalFast(rational_fast_converter) => {
                rational_fast_converter.next_sample(iter)
            }
        }
    }
}

const MIN_ORDER: u32 = 1;
const MAX_ORDER: u32 = 2048;
const MIN_QUAN: u32 = 1;
const MAX_QUAN: u32 = 16384;
const MIN_ATTEN: f64 = 12.0;
const MAX_ATTEN: f64 = 180.0;

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
pub struct Manager {
    ratio: Ratio,
    order: u32,
    quan: u32,
    latency: usize,
    lut: Lut,
}

impl Manager {
    fn with_raw_internal(
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

    fn with_raw_fast_internal(
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

    fn new_internal(ratio: Ratio, atten: f64, quan: u32, trans_width: f64) -> Result<Self> {
        check_f64("attenuation", atten, MIN_ATTEN, MAX_ATTEN)?;
        check_u32("quantify", quan, MIN_QUAN, MAX_QUAN)?;
        check_f64("trans_width", trans_width, 0.01, 1.0)?;
        let kaiser_beta = calc_kaiser_beta(atten);
        let fratio = ratio.as_float();
        let order = calc_order(fratio, atten, trans_width);
        let cutoff = fratio.min(1.0) * (1.0 - 0.5 * trans_width);
        Self::with_raw_internal(ratio, quan, order, kaiser_beta, cutoff)
    }

    fn with_order_internal(ratio: Ratio, atten: f64, quan: u32, order: u32) -> Result<Self> {
        check_f64("attenuation", atten, MIN_ATTEN, MAX_ATTEN)?;
        check_u32("quantify", quan, MIN_QUAN, MAX_QUAN)?;
        check_u32("order", order, MIN_ORDER, MAX_ORDER)?;
        let fratio = ratio.as_float();
        let kaiser_beta = calc_kaiser_beta(atten);
        let trans_width = calc_trans_width(fratio, atten, order);
        let cutoff = fratio.min(1.0) * (1.0 - 0.5 * trans_width);
        Self::with_raw_internal(ratio, quan, order, kaiser_beta, cutoff)
    }

    fn fast_new_internal(ratio: Ratio, atten: f64, trans_width: f64) -> Result<Self> {
        check_f64("attenuation", atten, MIN_ATTEN, MAX_ATTEN)?;
        check_f64("trans_width", trans_width, 0.01, 1.0)?;
        let rational = ratio.require_fast()?;
        let kaiser_beta = calc_kaiser_beta(atten);
        let fratio = ratio.as_float();
        let order = calc_order(fratio, atten, trans_width);
        let cutoff = fratio.min(1.0) * (1.0 - 0.5 * trans_width);
        Self::with_raw_fast_internal(rational, order, kaiser_beta, cutoff)
    }

    fn fast_with_order_internal(ratio: Ratio, atten: f64, order: u32) -> Result<Self> {
        check_f64("attenuation", atten, MIN_ATTEN, MAX_ATTEN)?;
        check_u32("order", order, MIN_ORDER, MAX_ORDER)?;
        let rational = ratio.require_fast()?;
        let fratio = ratio.as_float();
        let kaiser_beta = calc_kaiser_beta(atten);
        let trans_width = calc_trans_width(fratio, atten, order);
        let cutoff = fratio.min(1.0) * (1.0 - 0.5 * trans_width);
        Self::with_raw_fast_internal(rational, order, kaiser_beta, cutoff)
    }

    /// Create a Generic `Manager` with raw parameters, that means all of these
    /// should be calculated in advance.
    ///
    /// Always uses half-table interpolation; `quantify` is required. For a
    /// polyphase LUT, use [`fast_with_raw`](Self::fast_with_raw).
    ///
    /// - ratio: the conversion ratio, fs_new / fs_old, support `[1/16, 16]`
    /// - quan: the quantify number, usually power of 2, support `[1, 16384]`
    /// - order: the order of interpolation FIR filter, support `[1, 2048]`
    /// - kaiser_beta: the beta parameter of kaiser window method, support `[0.0, 20.0]`
    /// - cutoff: the cutoff of FIR filter, according to target sample rate, in `[0.01, 1.0]`
    pub fn with_raw(
        ratio: f64,
        quan: u32,
        order: u32,
        kaiser_beta: f64,
        cutoff: f64,
    ) -> Result<Self> {
        let ratio = Ratio::try_from_float(ratio)?;
        Self::with_raw_internal(ratio, quan, order, kaiser_beta, cutoff)
    }

    /// Create a Generic `Manager` with attenuation, quantify and transition band width.
    ///
    /// That means the order will be calculated. Always uses half-table
    /// interpolation; `quantify` is required. For a polyphase LUT, use
    /// [`fast`](Self::fast).
    ///
    /// - ratio: the conversion ratio, fs_new / fs_old, support `[1/16, 16]`
    /// - atten: the attenuation in dB, support `[12.0, 180.0]`
    /// - quan: the quantify number, usually power of 2, support `[1, 16384]`
    /// - trans_width: the transition band width in `[0.01, 1.0]`
    #[inline]
    pub fn new(ratio: f64, atten: f64, quan: u32, trans_width: f64) -> Result<Self> {
        let ratio = Ratio::try_from_float(ratio)?;
        Self::new_internal(ratio, atten, quan, trans_width)
    }

    /// Create a Generic `Manager` with attenuation, quantify and order
    ///
    /// That means the transition band will be calculated.
    ///
    /// - ratio: `[1/16, 16]`
    /// - atten: `[12.0, 180.0]`
    /// - quan: `[1, 16384]`
    /// - order: `[1, 2048]`
    #[inline]
    pub fn with_order(ratio: f64, atten: f64, quan: u32, order: u32) -> Result<Self> {
        let ratio = Ratio::try_from_float(ratio)?;
        Self::with_order_internal(ratio, atten, quan, order)
    }

    /// Create a Generic `Manager` with a [`Quality`] preset.
    ///
    /// Uses both [`Quality::attenuation`] and [`Quality::quantify`].
    #[inline]
    pub fn with_quality(ratio: f64, quality: Quality, trans_width: f64) -> Result<Self> {
        Self::new(
            ratio,
            quality.attenuation(),
            quality.quantify(),
            trans_width,
        )
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
    pub fn with_sample_rate(
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

    /// Create a Generic `Manager` from sample rates and a [`Quality`] preset.
    #[inline]
    pub fn with_sample_rate_quality(
        old_sr: u32,
        new_sr: u32,
        quality: Quality,
        pass_freq: u32,
    ) -> Result<Self> {
        Self::with_sample_rate(
            old_sr,
            new_sr,
            quality.attenuation(),
            quality.quantify(),
            pass_freq,
        )
    }

    /// Create a Fast polyphase `Manager`.
    ///
    /// Requires a rational ratio whose reduced numerator is ≤ 1024; otherwise
    /// returns [`Error::FastUnavailable`]. Does not take `quantify`.
    ///
    /// - ratio: `[1/16, 16]`
    /// - atten: `[12.0, 180.0]`
    /// - trans_width: `[0.01, 1.0]`
    #[inline]
    pub fn fast(ratio: f64, atten: f64, trans_width: f64) -> Result<Self> {
        let ratio = Ratio::try_from_float(ratio)?;
        Self::fast_new_internal(ratio, atten, trans_width)
    }

    /// Create a Fast polyphase `Manager` from attenuation and order.
    #[inline]
    pub fn fast_with_order(ratio: f64, atten: f64, order: u32) -> Result<Self> {
        let ratio = Ratio::try_from_float(ratio)?;
        Self::fast_with_order_internal(ratio, atten, order)
    }

    /// Create a Fast polyphase `Manager` from raw filter parameters.
    ///
    /// Does not take `quantify`. Fails with [`Error::FastUnavailable`] if the
    /// ratio is not eligible.
    pub fn fast_with_raw(ratio: f64, order: u32, kaiser_beta: f64, cutoff: f64) -> Result<Self> {
        let ratio = Ratio::try_from_float(ratio)?;
        let rational = ratio.require_fast()?;
        Self::with_raw_fast_internal(rational, order, kaiser_beta, cutoff)
    }

    /// Create a Fast polyphase `Manager` from a [`Quality`] preset.
    ///
    /// Only [`Quality::attenuation`] is used to compute β and order.
    /// [`Quality::quantify`] is ignored.
    #[inline]
    pub fn fast_with_quality(ratio: f64, quality: Quality, trans_width: f64) -> Result<Self> {
        Self::fast(ratio, quality.attenuation(), trans_width)
    }

    /// Create a Fast polyphase `Manager` from sample rates.
    ///
    /// Typical 44100/48000 conversions should use this (or
    /// [`fast_with_sample_rate_quality`](Self::fast_with_sample_rate_quality)).
    #[inline]
    pub fn fast_with_sample_rate(
        old_sr: u32,
        new_sr: u32,
        atten: f64,
        pass_freq: u32,
    ) -> Result<Self> {
        let ratio = Ratio::try_from_integers(new_sr, old_sr)?;
        let trans_width = trans_width_from_pass_freq(old_sr, new_sr, pass_freq);
        Self::fast_new_internal(ratio, atten, trans_width)
    }

    /// Create a Fast polyphase `Manager` from sample rates and a [`Quality`] preset.
    ///
    /// Only [`Quality::attenuation`] is used; [`Quality::quantify`] is ignored.
    #[inline]
    pub fn fast_with_sample_rate_quality(
        old_sr: u32,
        new_sr: u32,
        quality: Quality,
        pass_freq: u32,
    ) -> Result<Self> {
        Self::fast_with_sample_rate(old_sr, new_sr, quality.attenuation(), pass_freq)
    }

    /// Create a `Converter` which actually implement the interpolation.
    #[inline]
    pub fn converter(&self) -> Converter {
        let inner = match (&self.ratio, &self.lut) {
            (Ratio::Float(ratio), Lut::Generic(filter)) => ConverterKind::Float(
                FloatConverter::new(ratio.recip(), self.order, self.quan, filter.clone()),
            ),
            (Ratio::Rational(ratio), Lut::Generic(filter)) => ConverterKind::Rational(
                RationalConverter::new(ratio.recip(), self.order, self.quan, filter.clone()),
            ),
            (Ratio::Rational(ratio), Lut::Fast(lut)) => ConverterKind::RationalFast(
                RationalFastConverter::new(ratio.recip(), self.order, lut.clone()),
            ),
            _ => unreachable!("LUT kind must match ratio representation"),
        };
        Converter { inner }
    }

    /// Get the latency of the FIR filter in output samples.
    #[inline]
    pub fn latency(&self) -> usize {
        self.latency
    }

    /// Get the order of the FIR filter.
    #[inline]
    pub fn order(&self) -> u32 {
        self.order
    }

    /// Conversion ratio `fs_new / fs_old` actually in use.
    #[inline]
    pub fn ratio(&self) -> f64 {
        self.ratio.as_float()
    }

    /// Reduced integer ratio, if a rational mode was selected.
    #[inline]
    pub fn ratio_parts(&self) -> Option<(i64, i64)> {
        self.ratio.parts()
    }

    /// Which interpolation implementation this manager will construct.
    #[inline]
    pub fn mode(&self) -> ConvertMode {
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
    /// Generic is the half Kaiser-sinc table length. Fast is
    /// `numer * (order + 1)`.
    #[inline]
    pub fn lut_len(&self) -> usize {
        match &self.lut {
            Lut::Generic(filter) => filter.len(),
            Lut::Fast(lut) => lut.len() * (self.order as usize + 1),
        }
    }

    /// Expected output length for a complete input buffer of `input_len` samples.
    #[inline]
    pub fn output_len(&self, input_len: usize) -> usize {
        output_len(self.ratio(), input_len)
    }

    /// Convert a complete buffer.
    ///
    /// Pads the end with zeros and drops the leading FIR latency so the
    /// returned length is [`Self::output_len`].
    pub fn convert(&self, input: &[f64]) -> Vec<f64> {
        convert_with(self.converter(), self.latency, self.ratio(), input)
    }

    /// Create a `Builder` to build `Manager`
    #[inline]
    pub fn builder() -> Builder {
        Builder::default()
    }
}

/// The Builder to build `Manager`
///
/// Defaults to Generic interpolation (`quantify` is required). Call
/// [`.fast()`](Builder::fast) for a polyphase LUT; then `quantify` is ignored
/// and an ineligible ratio returns [`Error::FastUnavailable`].
///
/// ```
/// use simple_src::sinc;
///
/// let manager = sinc::Manager::builder()
///     .sample_rate(44100, 48000)
///     .quantify(32)
///     .attenuation(72)
///     .pass_freq(20000)
///     .build();
/// assert!(manager.is_ok());
/// ```
#[derive(Default)]
pub struct Builder {
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
}

impl Builder {
    /// Set `ratio` in `[1/16, 16]`.
    pub fn ratio(mut self, ratio: f64) -> Self {
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
    pub fn sample_rate(mut self, old_sr: u32, new_sr: u32) -> Self {
        self.old_sr = Some(old_sr);
        self.new_sr = Some(new_sr);
        self
    }

    /// Set quantify number in `[1, 16384]`.
    ///
    /// Required for Generic. Ignored after [`.fast()`](Self::fast).
    pub fn quantify(mut self, quan: u32) -> Self {
        self.quan = Some(quan);
        self
    }

    /// Set order of filter in `[1, 2048]`
    pub fn order(mut self, order: u32) -> Self {
        self.order = Some(order);
        self
    }

    /// Set beta of kaiser window function in `[0, 20]`
    pub fn kaiser_beta<B: Into<f64>>(mut self, beta: B) -> Self {
        self.kaiser_beta = Some(beta.into());
        self
    }

    /// Set cutoff of filter in `[0.01, 1.0]`
    pub fn cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = Some(cutoff);
        self
    }

    /// Set attenuation of stop band in `[12, 180]`
    pub fn attenuation<A: Into<f64>>(mut self, atten: A) -> Self {
        self.atten = Some(atten.into());
        self
    }

    /// Set transition band width in `[0.01, 1.0]`
    pub fn trans_width(mut self, width: f64) -> Self {
        self.trans_width = Some(width);
        self
    }

    /// Set pass band width in `[0, 0.99]`
    pub fn pass_width(mut self, width: f64) -> Self {
        self.trans_width = Some(1.0 - width);
        self
    }

    /// Set pass band frequency in Hz, the calculated transition band width
    /// should not less than 0.01
    pub fn pass_freq(mut self, freq: u32) -> Self {
        self.pass_freq = Some(freq);
        self
    }

    /// Set attenuation and quantify from a [`Quality`] preset.
    ///
    /// After [`.fast()`](Self::fast), only attenuation is used; quantify is
    /// ignored.
    pub fn quality(mut self, quality: Quality) -> Self {
        self.atten = Some(quality.attenuation());
        self.quan = Some(quality.quantify());
        self
    }

    /// Build a Fast polyphase LUT. `quantify` is not required and is ignored
    /// if set. Ineligible ratios return [`Error::FastUnavailable`].
    pub fn fast(mut self) -> Self {
        self.use_fast = true;
        self
    }

    /// Build Generic half-table interpolation (the default). `quantify` is
    /// required.
    pub fn generic(mut self) -> Self {
        self.use_fast = false;
        self
    }

    fn resolved_ratio(&self) -> Result<Ratio> {
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
    /// use simple_src::sinc;
    ///
    /// let manager = sinc::Builder::default()
    ///     .ratio(0.5)
    ///     .quantify(32)
    ///     .order(32)
    ///     .kaiser_beta(7.0)
    ///     .cutoff(0.8)
    ///     .build();
    /// assert!(manager.is_ok());
    /// ```
    pub fn build(self) -> Result<Manager> {
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
                Manager::with_raw_internal(ratio, quan, order, kaiser_beta, cutoff)
            }
            (_, _, _, Some(atten), Some(trans_width), _, _, _) => {
                Manager::new_internal(ratio, atten, quan, trans_width)
            }
            (Some(order), _, _, Some(atten), _, _, _, _) => {
                Manager::with_order_internal(ratio, atten, quan, order)
            }
            (_, _, _, Some(atten), _, Some(old_sr), Some(new_sr), Some(pass_freq)) => {
                Manager::with_sample_rate(old_sr, new_sr, atten, quan, pass_freq)
            }
            _ => Err(Error::missing(
                "attenuation with trans_width/order/pass_freq, or raw cutoff",
            )),
        }
    }

    fn build_fast(self) -> Result<Manager> {
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
                Manager::with_raw_fast_internal(rational, order, kaiser_beta, cutoff)
            }
            (_, _, _, Some(atten), Some(trans_width), _, _, _) => {
                Manager::fast_new_internal(ratio, atten, trans_width)
            }
            (Some(order), _, _, Some(atten), _, _, _, _) => {
                Manager::fast_with_order_internal(ratio, atten, order)
            }
            (_, _, _, Some(atten), _, Some(old_sr), Some(new_sr), Some(pass_freq)) => {
                Manager::fast_with_sample_rate(old_sr, new_sr, atten, pass_freq)
            }
            _ => Err(Error::missing(
                "attenuation with trans_width/order/pass_freq, or raw cutoff",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_with_raw() {
        assert!(Manager::with_raw(2.0, 32, 32, 5.0, 0.8).is_ok());
        assert!(Manager::with_raw(2.0, 0, 32, 5.0, 0.8).is_err());
        assert!(Manager::with_raw(2.0, 32, 0, 5.0, 0.8).is_err());
        assert!(Manager::with_raw(2.0, 32, 32, 5.0, 0.0).is_err());
        assert!(Manager::with_raw(2.0, 32, 32, 5.0, 1.1).is_err());
        assert!(Manager::with_raw(2.0, 32, 32, -0.1, 0.8).is_err());
        assert!(Manager::with_raw(2.0, 32, 32, 20.1, 0.8).is_err());
        assert!(Manager::fast_with_raw(2.0, 32, 5.0, 0.8).is_ok());
    }

    #[test]
    fn test_manager_new() {
        assert!(Manager::new(2.0, 72.0, 32, 0.1).is_ok());
        assert!(Manager::new(2.0, 72.0, 0, 0.1).is_err());
        assert!(Manager::new(2.0, 72.0, 32, 0.0).is_err());
        assert!(Manager::new(2.0, 72.0, 32, 1.1).is_err());
        assert!(Manager::new(2.0, 12.0, 32, 0.1).is_ok());
        assert!(Manager::new(2.0, 11.9, 32, 0.1).is_err());
        let generic = Manager::new(2.0, 72.0, 32, 0.1).unwrap();
        assert_eq!(generic.mode(), ConvertMode::Rational);
        assert_eq!(generic.lut_len(), (generic.order() * 32 / 2 + 1) as usize);
    }

    #[test]
    fn test_manager_fast() {
        let fast = Manager::fast(2.0, 72.0, 0.1).unwrap();
        assert_eq!(fast.mode(), ConvertMode::RationalFast);
        assert_eq!(fast.lut_len(), 2 * (fast.order() as usize + 1));
        let sr = Manager::fast_with_sample_rate(44100, 48000, 72.0, 20000).unwrap();
        assert_eq!(sr.mode(), ConvertMode::RationalFast);
        assert_eq!(sr.ratio_parts(), Some((160, 147)));
        assert!(matches!(
            Manager::fast_with_sample_rate(1024, 1025, 72.0, 400),
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
        let generic = Manager::with_quality(2.0, quality, trans_width).unwrap();
        let fast = Manager::fast_with_quality(2.0, quality, trans_width).unwrap();
        assert_eq!(generic.order(), fast.order());
        assert_eq!(generic.mode(), ConvertMode::Rational);
        assert_eq!(fast.mode(), ConvertMode::RationalFast);
        assert_eq!(
            generic.lut_len(),
            (generic.order() * quality.quantify() / 2 + 1) as usize
        );
    }

    #[test]
    fn test_manager_with_order() {
        assert!(Manager::with_order(2.0, 72.0, 32, 32).is_ok());
        assert!(Manager::with_order(2.0, 72.0, 32, 0).is_err());
        assert!(Manager::with_order(2.0, 72.0, 0, 32).is_err());
        assert!(Manager::with_order(2.0, 12.0, 32, 32).is_ok());
        assert!(Manager::with_order(2.0, 11.9, 32, 32).is_err());
        assert!(Manager::fast_with_order(2.0, 72.0, 32).is_ok());
    }

    #[test]
    fn test_builder() {
        assert!(Manager::builder().build().is_err());
        let manager = Manager::builder()
            .sample_rate(44100, 48000)
            .quantify(32)
            .attenuation(72)
            .pass_freq(20000)
            .build();
        assert!(manager.is_ok());
        assert_eq!(manager.unwrap().mode(), ConvertMode::Rational);
        let fast = Manager::builder()
            .sample_rate(44100, 48000)
            .attenuation(72)
            .pass_freq(20000)
            .fast()
            .build();
        assert!(fast.is_ok());
        assert_eq!(fast.unwrap().mode(), ConvertMode::RationalFast);
        let ignored_quan = Manager::builder()
            .ratio(2.0)
            .quantify(32)
            .attenuation(72)
            .trans_width(0.1)
            .fast()
            .build()
            .unwrap();
        assert_eq!(ignored_quan.mode(), ConvertMode::RationalFast);
        assert!(Manager::builder().ratio(0.0).quantify(8).build().is_err());
        let preset = Manager::with_sample_rate_quality(44100, 48000, Quality::Bit16Better, 20000);
        assert!(preset.is_ok());
        assert_eq!(preset.as_ref().unwrap().ratio_parts(), Some((160, 147)));
        assert_eq!(preset.unwrap().mode(), ConvertMode::Rational);
    }
}
