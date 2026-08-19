use super::{ConvertMode, Error, Result};

pub type Rational = num_rational::Rational64;

const LINEAR_FAST_NUMER_MAX: i64 = 16384;
const SINC_FAST_NUMER_MAX: i64 = 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ratio {
    Float(f64),
    Rational(Rational),
}

impl Ratio {
    pub fn is_supported(&self) -> bool {
        match self {
            Ratio::Float(ratio) => ratio.is_finite() && ((1.0 / 16.0)..=16.0).contains(ratio),
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
            Err(Error::unsupported(float_ratio))
        }
    }

    pub fn try_from_integers<T: Into<i64>>(numer: T, denom: T) -> Result<Self> {
        let numer = numer.into();
        let denom = denom.into();
        if numer == 0 || denom == 0 {
            return Err(Error::invalid("sample_rate", 0.0, 1.0, i32::MAX as f64));
        }
        let ratio = Rational::new(numer, denom);
        if Self::is_supported(&Self::Rational(ratio)) {
            Ok(Self::Rational(ratio))
        } else {
            Err(Error::unsupported(numer as f64 / denom as f64))
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            Ratio::Float(f) => *f,
            Ratio::Rational(r) => *r.numer() as f64 / *r.denom() as f64,
        }
    }

    pub fn parts(&self) -> Option<(i64, i64)> {
        match self {
            Ratio::Float(_) => None,
            Ratio::Rational(r) => Some((*r.numer(), *r.denom())),
        }
    }

    pub fn linear_mode(&self) -> ConvertMode {
        match self {
            Ratio::Float(_) => ConvertMode::Float,
            Ratio::Rational(r) if *r.numer() <= LINEAR_FAST_NUMER_MAX => ConvertMode::RationalFast,
            Ratio::Rational(_) => ConvertMode::Rational,
        }
    }

    pub fn require_fast(&self) -> Result<Rational> {
        match self {
            Ratio::Rational(r) if *r.numer() <= SINC_FAST_NUMER_MAX => Ok(*r),
            Ratio::Rational(r) => Err(Error::fast_unavailable(self.as_float(), Some(*r.numer()))),
            Ratio::Float(f) => Err(Error::fast_unavailable(*f, None)),
        }
    }
}

impl Default for Ratio {
    fn default() -> Self {
        Self::Float(f64::default())
    }
}
