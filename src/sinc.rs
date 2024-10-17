//! Sinc interpolation converter

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
        for (i, s) in self.buf.iter().enumerate() {
            let index = i as f64 - self.half_order;
            let pos = (coef - index).abs() * self.quan;
            let pos_n = pos as usize;
            if pos_n < pos_max {
                let h1 = self.filter[pos_n];
                let h2 = self.filter[pos_n + 1];
                let h = h1 + (h2 - h1) * (pos - pos_n as f64);
                interp += s * h;
            }
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

    /// Create a `Manager` with raw parameters, that means all of these should
    /// be calculated in advance.
    ///
    /// - ratio: the conversion ratio, fs_new / fs_old, support `[0.1, 10.0]`
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
    /// - ratio: the conversion ratio, fs_new / fs_old, support `[0.1, 10.0]`
    /// - atten: the attenuation in dB, support `[12.0, 180.0]`
    /// - quan: the quantify number, usually power of 2, support `[1, 16384]`
    /// - trans_width: the transition band width in `[0.01, 1.0]`
    #[inline]
    pub fn new(ratio: f64, atten: f64, quan: u32, trans_width: f64) -> Result<Self> {
        let ratio_i64 = Rational64::approximate_float(ratio).unwrap_or_default();
        if !supported_ratio(ratio_i64) {
            return Err(Error::UnsupportedRatio);
        }
        if !(MIN_ATTEN..=MAX_ATTEN).contains(&atten)
            || !(MIN_QUAN..=MAX_QUAN).contains(&quan)
            || !(0.01..=1.0).contains(&trans_width)
        {
            return Err(Error::InvalidParam);
        }
        let kaiser_beta = calc_kaiser_beta(atten);
        let order = calc_order(ratio, atten, trans_width);
        let cutoff = ratio.min(1.0) * (1.0 - 0.5 * trans_width);
        Self::with_raw_internal(ratio_i64, quan, order, kaiser_beta, cutoff)
    }

    /// Create a `Manager` with attenuation, quantify and order
    ///
    /// That means the transition band will be calculated.
    ///
    /// - ratio: `[0.1, 10.0]`
    /// - atten: `[12.0, 180.0]`
    /// - quan: `[1, 16384]`
    /// - order: `[1, 2048]`
    #[inline]
    pub fn with_order(ratio: f64, atten: f64, quan: u32, order: u32) -> Result<Self> {
        let ratio_i64 = Rational64::approximate_float(ratio).unwrap_or_default();
        if !supported_ratio(ratio_i64) {
            return Err(Error::UnsupportedRatio);
        }
        if !(MIN_ATTEN..=MAX_ATTEN).contains(&atten)
            || !(MIN_QUAN..=MAX_QUAN).contains(&quan)
            || !(MIN_ORDER..=MAX_ORDER).contains(&order)
        {
            return Err(Error::InvalidParam);
        }
        let kaiser_beta = calc_kaiser_beta(atten);
        let trans_width = calc_trans_width(ratio, atten, order);
        let cutoff = ratio.min(1.0) * (1.0 - 0.5 * trans_width);
        Self::with_raw_internal(ratio_i64, quan, order, kaiser_beta, cutoff)
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
}
