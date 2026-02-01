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

use num_rational::Rational64;

use crate::supported_ratio;

use super::{Convert, Error, Result};

enum State {
    First,
    Normal,
    Suspend,
}

pub struct Converter {
    numer: usize,
    denom: usize,
    pos: usize,
    coefs: Vec<f64>,
    last_in: [f64; 2],
    state: State,
}

impl Converter {
    #[inline]
    fn new(step: Rational64) -> Self {
        let numer = *step.numer() as usize;
        let denom = *step.denom() as usize;
        let mut coefs = Vec::with_capacity(denom);
        for i in 0..denom {
            coefs.push(i as f64 / denom as f64);
        }
        Self {
            numer,
            denom,
            pos: 0,
            coefs,
            last_in: [0.0; 2],
            state: State::First,
        }
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
                    let coef = self.coefs[self.pos];
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

#[derive(Clone, Copy)]
pub struct Manager {
    ratio: Rational64,
}

impl Manager {
    #[inline]
    pub fn new(ratio: f64) -> Result<Self> {
        let ratio = Rational64::approximate_float(ratio).unwrap_or_default();
        if supported_ratio(ratio) {
            Ok(Self { ratio })
        } else {
            Err(Error::UnsupportedRatio)
        }
    }

    #[inline]
    pub fn converter(&self) -> Converter {
        Converter::new(self.ratio.recip())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_ok() {
        let ratio_ok = vec![0.0625, 0.063, 1.0, 15.9, 16.0];
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
            0.123456,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for ratio in ratio_err {
            assert!(Manager::new(ratio).is_err());
        }
    }
}
