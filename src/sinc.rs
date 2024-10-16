//! Sinc interpolation converter

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::Arc;

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
    step: f64,
    pos: f64,
    half_order: f64,
    quan: f64,
    filter: Arc<Vec<f64>>,
    buf: VecDeque<f64>,
    state: State,
}

impl Converter {
    #[inline]
    fn new(step: f64, order: u32, quan: u32, filter: Arc<Vec<f64>>) -> Self {
        let taps = (order + 1) as usize;
        let mut buf = VecDeque::with_capacity(taps);
        buf.extend(std::iter::repeat(0.0).take(taps));
        Self {
            step,
            pos: 0.0,
            half_order: 0.5 * order as f64,
            quan: quan as f64,
            filter,
            buf,
            state: State::Normal,
        }
    }

    #[inline]
    fn interpolate(&self) -> f64 {
        let mut interp = 0.0;
        let pos_max = self.filter.len() - 1;
        let taps = self.buf.len();
        let iter_count = taps / 2;
        let mut left;
        let mut right;
        if taps % 2 == 1 {
            let pos = self.pos * self.quan;
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
        let coef = self.pos + self.half_order;
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
                    self.pos += self.step;
                    let interp = self.interpolate();
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

use super::{MAX_RATIO, MIN_RATIO};

const MIN_ORDER: u32 = 1;
const MAX_ORDER: u32 = 2048;
const MIN_QUAN: u32 = 1;
const MAX_QUAN: u32 = 16384;
const MIN_CUTOFF: f64 = 0.01;
const MAX_CUTOFF: f64 = 1.0;
const MIN_ATTEN: f64 = 12.0;
const MAX_ATTEN: f64 = 180.0;

#[derive(Clone)]
pub struct Manager {
    ratio: f64,
    order: u32,
    quan: u32,
    latency: usize,
    filter: Arc<Vec<f64>>,
}

impl Manager {
    /// Create a `Manager` with raw parameters, that means all of these should
    /// be calculated in advance.
    ///
    /// - ratio: the conversion ratio, fs_new / fs_old, support [0.1, 100.0]
    /// - quan: the quantify number, usually power of 2, support [1, 16384]
    /// - order: the order of interpolation FIR filter, support [1, 2048]
    /// - kaiser_beta: the beta parameter of kaiser window method, support [0.0, 20.0]
    /// - cutoff: the cutoff of FIR filter, according to target sample rate, in [0.01, 1.0]
    pub fn with_raw(
        ratio: f64,
        quan: u32,
        order: u32,
        kaiser_beta: f64,
        cutoff: f64,
    ) -> Result<Self> {
        if !(MIN_RATIO..=MAX_RATIO).contains(&ratio) {
            return Err(Error::InvalidRatio);
        }
        if !(MIN_QUAN..=MAX_QUAN).contains(&quan)
            || !(MIN_ORDER..=MAX_ORDER).contains(&order)
            || !(0.0..=20.0).contains(&kaiser_beta)
            || !(MIN_CUTOFF..=MAX_CUTOFF).contains(&cutoff)
        {
            return Err(Error::InvalidParam);
        }
        let filter = generate_filter_table(quan, order, kaiser_beta, cutoff);
        let latency = (ratio * order as f64 * 0.5).round() as usize;
        Ok(Self {
            ratio,
            order,
            quan,
            latency,
            filter: Arc::new(filter),
        })
    }

    /// Create a `Manager` with attenuation, quantify and transition band width.
    ///
    /// That means the order will be calculated.
    ///
    /// - ratio: the conversion ratio, fs_new / fs_old, support [0.1, 100.0]
    /// - atten: the attenuation in dB, support [12.0, 180.0]
    /// - quan: the quantify number, usually power of 2, support [1, 16384]
    /// - trans_width: the transition band width in [0.01, 1.0]
    #[inline]
    pub fn new(ratio: f64, atten: f64, quan: u32, trans_width: f64) -> Result<Self> {
        if !(MIN_RATIO..=MAX_RATIO).contains(&ratio) {
            return Err(Error::InvalidRatio);
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
        Self::with_raw(ratio, quan, order, kaiser_beta, cutoff)
    }

    /// Create a `Manager` with attenuation, quantify and order
    ///
    /// That means the transition band will be calculated.
    ///
    /// - ratio: [0.1, 100.0]
    /// - atten: [12.0, 180.0]
    /// - quan: [1, 16384]
    /// - order: [1, 2048]
    #[inline]
    pub fn with_order(ratio: f64, atten: f64, quan: u32, order: u32) -> Result<Self> {
        if !(MIN_RATIO..=MAX_RATIO).contains(&ratio) {
            return Err(Error::InvalidRatio);
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
        Self::with_raw(ratio, quan, order, kaiser_beta, cutoff)
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
        assert!(Manager::with_raw(0.01, 32, 32, 5.0, 0.8).is_ok());
        assert!(Manager::with_raw(100.0, 32, 32, 5.0, 0.8).is_ok());
        assert!(Manager::with_raw(0.009, 32, 32, 5.0, 0.8).is_err());
        assert!(Manager::with_raw(100.1, 32, 32, 5.0, 0.8).is_err());
        assert!(Manager::with_raw(2.0, 0, 32, 5.0, 0.8).is_err());
        assert!(Manager::with_raw(2.0, 32, 0, 5.0, 0.8).is_err());
        assert!(Manager::with_raw(2.0, 32, 32, 5.0, -0.1).is_err());
        assert!(Manager::with_raw(2.0, 32, 32, 5.0, 1.1).is_err());
        assert!(Manager::with_raw(2.0, 32, 32, -0.1, 0.8).is_err());
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
