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
        let coef = self.pos;
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
        if ratio < 0.01 || ratio > 100.0 {
            return Err(Error::InvalidRatio);
        }
        if quan == 0
            || quan > 16384
            || order == 0
            || order > 2048
            || kaiser_beta < 0.0
            || kaiser_beta > 20.0
            || cutoff < 0.01
            || cutoff > 1.0
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
        if ratio < 0.01 || ratio > 100.0 {
            return Err(Error::InvalidRatio);
        }
        if atten < 12.0
            || atten > 180.0
            || quan == 0
            || quan > 16384
            || trans_width < 0.01
            || trans_width > 1.0
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
        if ratio < 0.01 || ratio > 100.0 {
            return Err(Error::InvalidRatio);
        }
        if atten < 12.0 || atten > 180.0 || quan == 0 || quan > 16384 || order == 0 || order > 2048
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
        assert!(Manager::with_raw(2.0, 48, 32, 5.0, 0.8).is_ok());
        assert!(Manager::with_raw(0.01, 48, 32, 5.0, 0.8).is_ok());
        assert!(Manager::with_raw(100.0, 48, 32, 5.0, 0.8).is_ok());
        assert!(Manager::with_raw(0.009, 48, 32, 5.0, 0.8).is_err());
        assert!(Manager::with_raw(100.1, 48, 32, 5.0, 0.8).is_err());
        assert!(Manager::with_raw(2.0, 0, 32, 5.0, 0.8).is_err());
        assert!(Manager::with_raw(2.0, 48, 0, 5.0, 0.8).is_err());
        assert!(Manager::with_raw(2.0, 48, 32, 5.0, -0.1).is_err());
        assert!(Manager::with_raw(2.0, 48, 32, 5.0, 1.1).is_err());
        assert!(Manager::with_raw(2.0, 48, 32, -0.1, 0.8).is_err());
    }

    #[test]
    fn test_manager_new() {
        assert!(Manager::new(2.0, 96.0, 128, 0.1).is_ok());
        assert!(Manager::new(2.0, 96.0, 0, 0.1).is_err());
        assert!(Manager::new(2.0, 96.0, 128, 0.0).is_err());
        assert!(Manager::new(2.0, 96.0, 128, 1.1).is_err());
        assert!(Manager::new(2.0, 4.0, 128, 0.1).is_err());
        assert!(Manager::new(2.0, 8.0, 128, 0.1).is_err());
        assert!(Manager::new(2.0, 8.1, 128, 0.1).is_ok());
    }

    #[test]
    fn test_manager_with_order() {
        assert!(Manager::with_order(2.0, 96.0, 128, 128).is_ok());
        assert!(Manager::with_order(2.0, 96.0, 128, 0).is_err());
        assert!(Manager::with_order(2.0, 96.0, 0, 128).is_err());
        assert!(Manager::with_order(2.0, 8.0, 128, 128).is_err());
        assert!(Manager::with_order(2.0, 8.1, 128, 128).is_ok());
    }
}
