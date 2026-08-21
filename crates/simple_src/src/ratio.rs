use super::{ConvertMode, Error, Result};

pub type Rational = num_rational::Rational64;

const LINEAR_FAST_NUMER_MAX: i64 = 16384;
const SINC_FAST_NUMER_MAX: i64 = 1024;
/// Max numerator or denominator accepted from a float approximation.
/// Documented for callers via [`ConvertMode`] and Manager docs.
const GENERIC_RATIONAL_TERM_MAX: i64 = 16384;
/// Relative error above this keeps a float phase accumulator.
const RATIONAL_REL_ERR: f64 = 1e-12;

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
        if !Self::is_supported(&Self::Float(float_ratio)) {
            return Err(Error::unsupported(float_ratio));
        }
        if let Some(ratio) = approximate_bounded(float_ratio, GENERIC_RATIONAL_TERM_MAX) {
            let approx = *ratio.numer() as f64 / *ratio.denom() as f64;
            let rel = (approx - float_ratio).abs() / float_ratio.abs();
            if rel <= RATIONAL_REL_ERR && Self::is_supported(&Self::Rational(ratio)) {
                return Ok(Self::Rational(ratio));
            }
        }
        Ok(Self::Float(float_ratio))
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

/// Best continued-fraction convergent with numer and denom ≤ `max_term`.
fn approximate_bounded(x: f64, max_term: i64) -> Option<Rational> {
    if !x.is_finite() || x <= 0.0 || max_term < 1 {
        return None;
    }

    let mut n0: i64 = 0;
    let mut d0: i64 = 1;
    let mut n1: i64 = 1;
    let mut d1: i64 = 0;
    let mut rest = x;

    for _ in 0..64 {
        if !rest.is_finite() || rest > i64::MAX as f64 {
            break;
        }
        let a = rest.floor() as i64;
        if a < 0 {
            break;
        }
        match next_convergent(a, n0, d0, n1, d1, max_term) {
            Some((n2, d2)) => {
                n0 = n1;
                d0 = d1;
                n1 = n2;
                d1 = d2;
            }
            None => {
                if let Some((n2, d2)) = best_semiconvergent(a, n0, d0, n1, d1, max_term) {
                    n1 = n2;
                    d1 = d2;
                }
                break;
            }
        }
        if n1 > 0 && d1 > 0 {
            let approx = n1 as f64 / d1 as f64;
            if (approx - x).abs() <= RATIONAL_REL_ERR * x.abs() {
                break;
            }
        }
        let frac = rest - a as f64;
        if frac <= (rest.abs() + 1.0) * 1e-18 {
            break;
        }
        rest = 1.0 / frac;
    }

    if d1 <= 0 || n1 <= 0 {
        return None;
    }
    Some(Rational::new(n1, d1))
}

fn next_convergent(
    a: i64,
    n0: i64,
    d0: i64,
    n1: i64,
    d1: i64,
    max_term: i64,
) -> Option<(i64, i64)> {
    let n2 = a.checked_mul(n1)?.checked_add(n0)?;
    let d2 = a.checked_mul(d1)?.checked_add(d0)?;
    if n2 > max_term || d2 > max_term || d2 <= 0 {
        None
    } else {
        Some((n2, d2))
    }
}

fn best_semiconvergent(
    a: i64,
    n0: i64,
    d0: i64,
    n1: i64,
    d1: i64,
    max_term: i64,
) -> Option<(i64, i64)> {
    if a <= 1 {
        return None;
    }
    let mut t = a - 1;
    if n1 > 0 {
        t = t.min(max_term.saturating_sub(n0) / n1);
    }
    if d1 > 0 {
        t = t.min(max_term.saturating_sub(d0) / d1);
    }
    if t < 1 {
        return None;
    }
    next_convergent(t, n0, d0, n1, d1, max_term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{E, PI};

    fn parts_of(x: f64) -> Option<(i64, i64)> {
        Ratio::try_from_float(x).unwrap().parts()
    }

    #[test]
    fn simple_floats_are_rational() {
        assert_eq!(parts_of(2.0), Some((2, 1)));
        assert_eq!(parts_of(0.5), Some((1, 2)));
        assert_eq!(parts_of(0.7), Some((7, 10)));
        assert_eq!(parts_of(1.0 / 3.0), Some((1, 3)));
        assert_eq!(parts_of(48000.0 / 44100.0), Some((160, 147)));
        assert_eq!(parts_of(44100.0 / 48000.0), Some((147, 160)));
    }

    #[test]
    fn inexact_floats_stay_float() {
        let pi = Ratio::try_from_float(PI).unwrap();
        assert_eq!(pi, Ratio::Float(PI));
        assert_eq!(pi.as_float(), PI);
        assert!(matches!(
            pi.require_fast(),
            Err(Error::FastUnavailable { numer: None, .. })
        ));

        let e = Ratio::try_from_float(E).unwrap();
        assert!(matches!(e, Ratio::Float(_)));

        let messy = Ratio::try_from_float(1.23456789).unwrap();
        assert!(matches!(messy, Ratio::Float(_)));
    }

    #[test]
    fn integers_keep_large_rationals() {
        let r = Ratio::try_from_integers(1025i64, 1024i64).unwrap();
        assert_eq!(r.parts(), Some((1025, 1024)));
        assert!(matches!(
            r.require_fast(),
            Err(Error::FastUnavailable {
                numer: Some(1025),
                ..
            })
        ));
    }
}
