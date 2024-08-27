use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::Arc;

use super::NextSample;

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
fn kaiser(x: f64, order: u32, beta: f64) -> f64 {
    let half = order as f64 * 0.5;
    if (x < -half) || (x > half) {
        return 0.0;
    }
    bessel_i0(beta * (1.0 - (x / half).powi(2)).sqrt()) / bessel_i0(beta)
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
    pub fn new(step: f64, order: u32, quan: u32, filter: Arc<Vec<f64>>) -> Self {
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
                let h = h1 + (h2 - h1) * pos.fract();
                interp += s * h;
            }
        }
        interp
    }
}

impl NextSample for Converter {
    #[inline]
    fn next_sample<F>(&mut self, f: &mut F) -> Option<f64>
    where
        F: FnMut() -> Option<f64>,
    {
        loop {
            match self.state {
                State::Normal => {
                    while self.pos >= 1.0 {
                        self.pos -= 1.0;
                        if let Some(s) = f() {
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
                    if let Some(s) = f() {
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

pub struct Manager {
    ratio: f64,
    order: u32,
    quan: u32,
    latency: usize,
    filter: Arc<Vec<f64>>,
}

impl Manager {
    pub fn with_raw(ratio: f64, quan: u32, order: u32, kaiser_beta: f64, cutoff: f64) -> Self {
        let half = order as f64 * 0.5;
        let h_len = (quan as f64 * half).ceil() as usize;
        let mut filter = Vec::with_capacity(h_len);
        for i in 0..h_len {
            let pos = i as f64 / quan as f64;
            let coef = sinc_c(pos, cutoff) * kaiser(pos, order, kaiser_beta);
            filter.push(coef);
        }
        let latency = (ratio * half).round() as usize;
        Self {
            ratio,
            order,
            quan,
            latency,
            filter: Arc::new(filter),
        }
    }

    #[inline]
    pub fn new(ratio: f64, atten: f64, quan: u32, trans_width: f64) -> Self {
        let kaiser_beta = calc_kaiser_beta(atten);
        let order = calc_order(ratio, atten, trans_width);
        let cutoff = ratio.min(1.0) * (1.0 - 0.5 * trans_width);
        Self::with_raw(ratio, quan, order, kaiser_beta, cutoff)
    }

    #[inline]
    pub fn with_order(ratio: f64, atten: f64, quan: u32, order: u32) -> Self {
        let kaiser_beta = calc_kaiser_beta(atten);
        let trans_width = calc_trans_width(ratio, atten, order);
        let cutoff = ratio.min(1.0) * (1.0 - 0.5 * trans_width);
        Self::with_raw(ratio, quan, order, kaiser_beta, cutoff)
    }

    #[inline]
    pub fn converter(&self) -> Converter {
        Converter::new(
            self.ratio.recip(),
            self.order,
            self.quan,
            self.filter.clone(),
        )
    }

    #[inline]
    pub fn latency(&self) -> usize {
        self.latency
    }

    #[inline]
    pub fn order(&self) -> u32 {
        self.order
    }
}
