use super::{Error, Result};

pub type Rational = num_rational::Rational64;

#[derive(Clone, Copy)]
pub enum Ratio {
    Float(f64),
    Rational(Rational),
}

impl Ratio {
    pub fn is_supported(&self) -> bool {
        match self {
            Ratio::Float(ratio) => ((1.0 / 16.0)..=16.0).contains(ratio),
            Ratio::Rational(ratio) => {
                *ratio > Rational::default()
                    && ratio.ceil().to_integer() <= 16
                    && ratio.recip().ceil().to_integer() <= 16
            }
        }
    }

    pub fn try_from_float(float_ratio: f64) -> Result<Self> {
        if Self::is_supported(&Self::Float(float_ratio)) {
            let ratio = Rational::approximate_float(float_ratio).unwrap_or_default();
            if Self::is_supported(&Self::Rational(ratio)) {
                Ok(Self::Rational(ratio))
            } else {
                Ok(Self::Float(float_ratio))
            }
        } else {
            Err(Error::UnsupportedRatio)
        }
    }

    pub fn try_from_integers<T: Into<i64>>(numer: T, denom: T) -> Result<Self> {
        let numer = numer.into();
        let denom = denom.into();
        if numer == 0 || denom == 0 {
            return Err(Error::InvalidParam);
        }
        let ratio = Rational::new(numer, denom);
        if Self::is_supported(&Self::Rational(ratio)) {
            Ok(Self::Rational(ratio))
        } else {
            Err(Error::UnsupportedRatio)
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            Ratio::Float(f) => *f,
            Ratio::Rational(r) => *r.numer() as f64 / *r.denom() as f64,
        }
    }
}

impl Default for Ratio {
    fn default() -> Self {
        Self::Float(f64::default())
    }
}
