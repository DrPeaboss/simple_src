//! Linear interpolation converter
//!
//! ```
//! use simple_src::{linear, Convert};
//!
//! let samples = vec![1.0, 2.0, 3.0, 4.0];
//! let manager = linear::Manager::new(2.0).unwrap();
//! let mut converter = manager.converter();
//! for s in converter.process(samples.into_iter()) {
//!     println!("{s}");
//! }
//! ```

use super::{Convert, Ratio, Rational, Result};

enum State {
    First,
    Normal,
    Suspend,
}

pub struct FloatConverter {
    state: State,
    last_in: [f64; 2],
    step: f64,
    pos: f64,
}

pub struct RationalConverter {
    state: State,
    last_in: [f64; 2],
    numer: usize,
    denom: usize,
    pos: usize,
    recip: f64,
}

pub struct RationalFastConverter {
    state: State,
    last_in: [f64; 2],
    numer: usize,
    denom: usize,
    pos: usize,
    coef: Vec<f64>,
}

pub enum Converter {
    Float(FloatConverter),
    Rational(RationalConverter),
    RationalFast(RationalFastConverter),
}

impl FloatConverter {
    fn new(step: f64) -> Self {
        Self {
            step,
            pos: 0.0,
            last_in: [0.0; 2],
            state: State::First,
        }
    }
}

impl Convert for FloatConverter {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        loop {
            match self.state {
                State::First => {
                    if let Some(s) = iter.next() {
                        self.last_in[1] = s;
                        self.pos = 1.0;
                        self.state = State::Normal;
                    } else {
                        return None;
                    }
                }
                State::Normal => {
                    while self.pos >= 1.0 {
                        self.pos -= 1.0;
                        self.last_in[0] = self.last_in[1];
                        if let Some(s) = iter.next() {
                            self.last_in[1] = s;
                        } else {
                            self.state = State::Suspend;
                            return None;
                        }
                    }
                    let coef = self.pos;
                    let interp = self.last_in[0] + (self.last_in[1] - self.last_in[0]) * coef;
                    self.pos += self.step;
                    return Some(interp);
                }
                State::Suspend => {
                    if let Some(s) = iter.next() {
                        self.last_in[1] = s;
                        self.state = State::Normal;
                    } else {
                        return None;
                    }
                }
            }
        }
    }
}

impl RationalConverter {
    fn new(step: Rational) -> Self {
        let numer = *step.numer() as usize;
        let denom = *step.denom() as usize;
        Self {
            state: State::First,
            last_in: [0.0; 2],
            numer,
            denom,
            pos: 0,
            recip: (denom as f64).recip(),
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
                State::First => {
                    if let Some(s) = iter.next() {
                        self.last_in[1] = s;
                        self.pos = self.numer;
                        self.state = State::Normal;
                    } else {
                        return None;
                    }
                }
                State::Normal => {
                    while self.pos >= self.denom {
                        self.pos -= self.denom;
                        self.last_in[0] = self.last_in[1];
                        if let Some(s) = iter.next() {
                            self.last_in[1] = s;
                        } else {
                            self.state = State::Suspend;
                            return None;
                        }
                    }
                    let coef = self.pos as f64 * self.recip;
                    let interp = self.last_in[0] + (self.last_in[1] - self.last_in[0]) * coef;
                    self.pos += self.numer;
                    return Some(interp);
                }
                State::Suspend => {
                    if let Some(s) = iter.next() {
                        self.last_in[1] = s;
                        self.state = State::Normal;
                    } else {
                        return None;
                    }
                }
            }
        }
    }
}

impl RationalFastConverter {
    fn new(step: Rational) -> Self {
        let numer = *step.numer() as usize;
        let denom = *step.denom() as usize;
        let coef = (0..denom).map(|i| i as f64 / denom as f64).collect();
        Self {
            numer,
            denom,
            pos: 0,
            coef,
            last_in: [0.0; 2],
            state: State::First,
        }
    }
}

impl Convert for RationalFastConverter {
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
    {
        loop {
            match self.state {
                State::First => {
                    if let Some(s) = iter.next() {
                        self.last_in[1] = s;
                        self.pos = self.numer;
                        self.state = State::Normal;
                    } else {
                        return None;
                    }
                }
                State::Normal => {
                    while self.pos >= self.denom {
                        self.pos -= self.denom;
                        self.last_in[0] = self.last_in[1];
                        if let Some(s) = iter.next() {
                            self.last_in[1] = s;
                        } else {
                            self.state = State::Suspend;
                            return None;
                        }
                    }
                    let coef = self.coef[self.pos];
                    let interp = self.last_in[0] + (self.last_in[1] - self.last_in[0]) * coef;
                    self.pos += self.numer;
                    return Some(interp);
                }
                State::Suspend => {
                    if let Some(s) = iter.next() {
                        self.last_in[1] = s;
                        self.state = State::Normal;
                    } else {
                        return None;
                    }
                }
            }
        }
    }
}

impl Converter {
    fn new(ratio: Ratio) -> Self {
        match ratio {
            Ratio::Float(ratio) => Self::Float(FloatConverter::new(ratio.recip())),
            Ratio::Rational(ratio) => {
                if *ratio.numer() <= 16384 {
                    Self::RationalFast(RationalFastConverter::new(ratio.recip()))
                } else {
                    Self::Rational(RationalConverter::new(ratio.recip()))
                }
            }
        }
    }
}

impl Convert for Converter {
    #[inline]
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        match self {
            Converter::Float(converter) => converter.next_sample(iter),
            Converter::Rational(converter) => converter.next_sample(iter),
            Converter::RationalFast(converter) => converter.next_sample(iter),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Manager {
    ratio: Ratio,
}

impl Manager {
    #[inline]
    pub fn new(ratio: f64) -> Result<Self> {
        let ratio = Ratio::try_from_float(ratio)?;
        Ok(Self { ratio })
    }

    #[inline]
    pub fn converter(&self) -> Converter {
        Converter::new(self.ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_ok() {
        let ratio_ok = vec![0.0625, 0.063, 1.0, 15.9, 16.0, 0.123456];
        for ratio in ratio_ok {
            assert!(Manager::new(ratio).is_ok());
        }
    }

    #[test]
    fn test_manager_err() {
        let ratio_err = vec![
            -1.0,
            0.0,
            0.0624,
            16.01,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for ratio in ratio_err {
            assert!(Manager::new(ratio).is_err());
        }
    }
}
