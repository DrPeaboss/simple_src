//! Linear converter

use super::{Convert, Error, Result};

enum State {
    First,
    Normal,
    Suspend,
}

pub struct Converter {
    step: f64,
    pos: f64,
    last_in: [f64; 2],
    state: State,
}

impl Converter {
    #[inline]
    fn new(step: f64) -> Self {
        Self {
            step,
            pos: 0.0,
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
                    let interp = self.last_in[0] + (self.last_in[1] - self.last_in[0]) * self.pos;
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

#[derive(Clone, Copy)]
pub struct Manager {
    ratio: f64,
}

impl Manager {
    #[inline]
    pub fn new(ratio: f64) -> Result<Self> {
        if ratio >= 0.01 && ratio <= 100.0 {
            Ok(Self { ratio })
        } else {
            Err(Error::InvalidRatio)
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
        let ratio_ok = vec![0.01, 1.0, 10.0, 99.99, 100.0];
        for ratio in ratio_ok {
            assert!(Manager::new(ratio).is_ok());
        }
    }

    #[test]
    fn test_manager_err() {
        let ratio_err = vec![
            -1.0,
            0.0,
            100.01,
            1000.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for ratio in ratio_err {
            assert!(Manager::new(ratio).is_err());
        }
    }
}
