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

use num_rational::Rational64;

use crate::supported_ratio;

use super::{Convert, Error, Result};

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
#[allow(dead_code)]
fn kaiser(x: f64, order: u32, beta: f64) -> f64 {
    let half = order as f64 * 0.5;
    if (x < -half) || (x > half) {
        return 0.0;
    }
    bessel_i0(beta * (1.0 - (x / half).powi(2)).sqrt()) / bessel_i0(beta)
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

pub struct Converter {
    numer: usize,
    denom: usize,
    pos: usize,
    coefs: Vec<f64>,
    half_order: f64,
    quan: f64,
    filter: Arc<Vec<f64>>,
    buf: VecDeque<f64>,
    state: State,
}

impl Converter {
    #[inline]
    fn new(step: Rational64, order: u32, quan: u32, filter: Arc<Vec<f64>>) -> Self {
        let numer = *step.numer() as usize;
        let denom = *step.denom() as usize;
        let mut coefs = Vec::with_capacity(denom);
        for i in 0..denom {
            coefs.push(i as f64 / denom as f64);
        }
        let taps = (order + 1) as usize;
        let mut buf = VecDeque::with_capacity(taps);
        buf.extend(std::iter::repeat(0.0).take(taps));
        Self {
            numer,
            denom,
            pos: 0,
            coefs,
            half_order: 0.5 * order as f64,
            quan: quan as f64,
            filter,
            buf,
            state: State::Normal,
        }
    }

    #[inline]
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

impl Convert for Converter {
    #[inline]
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
                    if let Some(s) = iter.next() {
                        self.buf.pop_front();
                        self.buf.push_back(s);
                        self.state = State::Normal;
                    } else {
                        return None;
                    }
                }
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

#[derive(Clone)]
pub struct Manager {
    ratio: Rational64,
    order: u32,
    quan: u32,
    latency: usize,
    filter: Arc<Vec<f64>>,
}

impl Manager {
    fn with_raw_internal(
        ratio: Rational64,
        quan: u32,
        order: u32,
        kaiser_beta: f64,
        cutoff: f64,
    ) -> Result<Self> {
        if !supported_ratio(ratio) {
            return Err(Error::UnsupportedRatio);
        }
        if !(MIN_QUAN..=MAX_QUAN).contains(&quan)
            || !(MIN_ORDER..=MAX_ORDER).contains(&order)
            || !(0.0..=20.0).contains(&kaiser_beta)
            || !(0.01..=1.0).contains(&cutoff)
        {
            return Err(Error::InvalidParam);
        }
        let filter = generate_filter_table(quan, order, kaiser_beta, cutoff);
        let fratio = *ratio.numer() as f64 / *ratio.denom() as f64;
        let latency = (fratio * order as f64 * 0.5).round() as usize;
        Ok(Self {
            ratio,
            order,
            quan,
            latency,
            filter: Arc::new(filter),
        })
    }

    fn new_internal(ratio: Rational64, atten: f64, quan: u32, trans_width: f64) -> Result<Self> {
        if !supported_ratio(ratio) {
            return Err(Error::UnsupportedRatio);
        }
        if !(MIN_ATTEN..=MAX_ATTEN).contains(&atten)
            || !(MIN_QUAN..=MAX_QUAN).contains(&quan)
            || !(0.01..=1.0).contains(&trans_width)
        {
            return Err(Error::InvalidParam);
        }
        let kaiser_beta = calc_kaiser_beta(atten);
        let fratio = *ratio.numer() as f64 / *ratio.denom() as f64;
        let order = calc_order(fratio, atten, trans_width);
        let cutoff = fratio.min(1.0) * (1.0 - 0.5 * trans_width);
        Self::with_raw_internal(ratio, quan, order, kaiser_beta, cutoff)
    }

    fn with_order_internal(ratio: Rational64, atten: f64, quan: u32, order: u32) -> Result<Self> {
        if !supported_ratio(ratio) {
            return Err(Error::UnsupportedRatio);
        }
        if !(MIN_ATTEN..=MAX_ATTEN).contains(&atten)
            || !(MIN_QUAN..=MAX_QUAN).contains(&quan)
            || !(MIN_ORDER..=MAX_ORDER).contains(&order)
        {
            return Err(Error::InvalidParam);
        }
        let fratio = *ratio.numer() as f64 / *ratio.denom() as f64;
        let kaiser_beta = calc_kaiser_beta(atten);
        let trans_width = calc_trans_width(fratio, atten, order);
        let cutoff = fratio.min(1.0) * (1.0 - 0.5 * trans_width);
        Self::with_raw_internal(ratio, quan, order, kaiser_beta, cutoff)
    }

    /// Create a `Manager` with raw parameters, that means all of these should
    /// be calculated in advance.
    ///
    /// - ratio: the conversion ratio, fs_new / fs_old, support `[1/16, 16]`,
    ///   the numerator after reduction should <= 1024
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
        let ratio = Rational64::approximate_float(ratio).unwrap_or_default();
        Self::with_raw_internal(ratio, quan, order, kaiser_beta, cutoff)
    }

    /// Create a `Manager` with attenuation, quantify and transition band width.
    ///
    /// That means the order will be calculated.
    ///
    /// - ratio: the conversion ratio, fs_new / fs_old, support `[1/16, 16]`,
    ///   the numerator after reduction should <= 1024
    /// - atten: the attenuation in dB, support `[12.0, 180.0]`
    /// - quan: the quantify number, usually power of 2, support `[1, 16384]`
    /// - trans_width: the transition band width in `[0.01, 1.0]`
    #[inline]
    pub fn new(ratio: f64, atten: f64, quan: u32, trans_width: f64) -> Result<Self> {
        let ratio = Rational64::approximate_float(ratio).unwrap_or_default();
        Self::new_internal(ratio, atten, quan, trans_width)
    }

    /// Create a `Manager` with attenuation, quantify and order
    ///
    /// That means the transition band will be calculated.
    ///
    /// - ratio: `[1/16, 16]`
    /// - atten: `[12.0, 180.0]`
    /// - quan: `[1, 16384]`
    /// - order: `[1, 2048]`
    #[inline]
    pub fn with_order(ratio: f64, atten: f64, quan: u32, order: u32) -> Result<Self> {
        let ratio = Rational64::approximate_float(ratio).unwrap_or_default();
        Self::with_order_internal(ratio, atten, quan, order)
    }

    /// Create a `Manager` with sample rate, attenuation, quantify and pass frequency
    ///
    /// - old_sr: Old sample rate, not 0
    /// - new_sr: New sample rate, not 0
    /// - atten: `[12.0, 180.0]`
    /// - quan: `[1, 16384]`
    /// - order: `[1, 2048]`
    ///
    /// The sample rate ratio should in `[1/16, 16]` and the numerator after
    /// reduction cannot be greater than 1024
    #[inline]
    pub fn with_sample_rate(
        old_sr: u32,
        new_sr: u32,
        atten: f64,
        quan: u32,
        pass_freq: u32,
    ) -> Result<Self> {
        if old_sr == 0 || new_sr == 0 {
            return Err(Error::InvalidParam);
        }
        let ratio = Rational64::new(new_sr.into(), old_sr.into());
        if !supported_ratio(ratio) {
            return Err(Error::UnsupportedRatio);
        }
        let min_sr = new_sr.min(old_sr);
        let trans_width = min_sr.saturating_sub(pass_freq.saturating_mul(2)) as f64 / min_sr as f64;
        Self::new_internal(ratio, atten, quan, trans_width)
    }

    /// Create a `Converter` which actually implement the interpolation.
    #[inline]
    pub fn converter(&self) -> Converter {
        Converter::new(
            self.ratio.recip(),
            self.order,
            self.quan,
            self.filter.clone(),
        )
    }

    /// Get the latency of the FIR filter.
    #[inline]
    pub fn latency(&self) -> usize {
        self.latency
    }

    /// Get the order of the FIR filter.
    #[inline]
    pub fn order(&self) -> u32 {
        self.order
    }

    /// Create a `Builder` to build `Manager`
    #[inline]
    pub fn builder() -> Builder {
        Builder::default()
    }
}

/// The Builder to build `Manager`
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
    ratio: Option<Rational64>,
    order: Option<u32>,
    quan: Option<u32>,
    kaiser_beta: Option<f64>,
    cutoff: Option<f64>,
    atten: Option<f64>,
    trans_width: Option<f64>,
    old_sr: Option<u32>,
    new_sr: Option<u32>,
    pass_freq: Option<u32>,
}

impl Builder {
    /// Set `ratio`, in `[1/16, 16]`, the numerator after reduction should <= 1024
    pub fn ratio(mut self, ratio: f64) -> Self {
        self.ratio = Some(Rational64::approximate_float(ratio).unwrap_or_default());
        self
    }

    /// Set old sample rate and new sample rate
    pub fn sample_rate(mut self, old_sr: u32, new_sr: u32) -> Self {
        self.old_sr = Some(old_sr);
        self.new_sr = Some(new_sr);
        self
    }

    /// Set quantify number in `[1, 16384]`
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

    /// Build the `Manager`, there are the following combinations in order:
    ///
    /// - ratio, quantify, order, kaiser_beta, cutoff
    /// - ratio, attenuation, quantify, trans_width or pass_width
    /// - ratio, attenuation, quantify, order
    /// - sample_rate, attenuation, quantify, pass_freq
    ///
    /// For example, this is the first situation:
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
        let (ratio, quan) = match (self.ratio, self.quan, self.old_sr, self.new_sr) {
            (Some(ratio), Some(quan), _, _) => (ratio, quan),
            (_, Some(quan), Some(old_sr), Some(new_sr)) => {
                if old_sr == 0 || new_sr == 0 {
                    return Err(Error::InvalidParam);
                }
                (Rational64::new(new_sr.into(), old_sr.into()), quan)
            }
            _ => return Err(Error::NotEnoughParam),
        };
        if !supported_ratio(ratio) {
            return Err(Error::UnsupportedRatio);
        }
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
            _ => Err(Error::NotEnoughParam),
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
    }

    #[test]
    fn test_manager_new() {
        assert!(Manager::new(2.0, 72.0, 32, 0.1).is_ok());
        assert!(Manager::new(2.0, 72.0, 0, 0.1).is_err());
        assert!(Manager::new(2.0, 72.0, 32, 0.0).is_err());
        assert!(Manager::new(2.0, 72.0, 32, 1.1).is_err());
        assert!(Manager::new(2.0, 12.0, 32, 0.1).is_ok());
        assert!(Manager::new(2.0, 11.9, 32, 0.1).is_err());
    }

    #[test]
    fn test_manager_with_order() {
        assert!(Manager::with_order(2.0, 72.0, 32, 32).is_ok());
        assert!(Manager::with_order(2.0, 72.0, 32, 0).is_err());
        assert!(Manager::with_order(2.0, 72.0, 0, 32).is_err());
        assert!(Manager::with_order(2.0, 12.0, 32, 32).is_ok());
        assert!(Manager::with_order(2.0, 11.9, 32, 32).is_err());
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
    }
}
