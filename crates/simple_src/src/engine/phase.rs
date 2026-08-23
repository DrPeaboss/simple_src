use crate::Rational;

#[derive(Clone, Debug)]
pub(crate) enum PhaseAccum {
    Float {
        pos: f64,
        step: f64,
    },
    Rational {
        pos: usize,
        numer: usize,
        denom: usize,
        recip: f64,
    },
    RationalFast {
        pos: usize,
        numer: usize,
        denom: usize,
        coef: Vec<f64>,
    },
}

impl PhaseAccum {
    #[inline]
    pub(crate) fn float(step: f64) -> Self {
        Self::Float { pos: 0.0, step }
    }

    #[inline]
    pub(crate) fn rational(step: Rational) -> Self {
        let numer = *step.numer() as usize;
        let denom = *step.denom() as usize;
        Self::Rational {
            pos: 0,
            numer,
            denom,
            recip: (denom as f64).recip(),
        }
    }

    /// Rational fast phase for linear interpolation (precomputed fractional coefs).
    #[inline]
    pub(crate) fn rational_fast_linear(step: Rational) -> Self {
        let numer = *step.numer() as usize;
        let denom = *step.denom() as usize;
        let coef = (0..denom).map(|i| i as f64 / denom as f64).collect();
        Self::RationalFast {
            pos: 0,
            numer,
            denom,
            coef,
        }
    }

    /// After the first input sample in linear converters.
    #[inline]
    pub(crate) fn prepare_linear_priming(&mut self) {
        match self {
            Self::Float { pos, .. } => *pos = 1.0,
            Self::Rational { pos, numer, .. } | Self::RationalFast { pos, numer, .. } => {
                *pos = *numer;
            }
        }
    }

    #[inline]
    pub(crate) fn coef(&self) -> f64 {
        match self {
            Self::Float { pos, .. } => *pos,
            Self::Rational { pos, recip, .. } => *pos as f64 * recip,
            Self::RationalFast { pos, coef, .. } => coef[*pos],
        }
    }

    #[inline]
    pub(crate) fn pos_float(&self) -> f64 {
        match self {
            Self::Float { pos, .. } => *pos,
            Self::Rational { pos, denom, .. } => *pos as f64 / *denom as f64,
            Self::RationalFast { pos, .. } => *pos as f64,
        }
    }

    #[inline]
    pub(crate) fn pos_usize(&self) -> usize {
        match self {
            Self::Float { .. } => unreachable!("pos_usize on float phase"),
            Self::Rational { pos, .. } | Self::RationalFast { pos, .. } => *pos,
        }
    }

    #[inline]
    pub(crate) fn needs_input_advance(&self) -> bool {
        match self {
            Self::Float { pos, .. } => *pos >= 1.0,
            Self::Rational { pos, denom, .. } | Self::RationalFast { pos, denom, .. } => {
                *pos >= *denom
            }
        }
    }

    #[inline]
    pub(crate) fn consume_input_step(&mut self) {
        match self {
            Self::Float { pos, .. } => *pos -= 1.0,
            Self::Rational { pos, denom, .. } | Self::RationalFast { pos, denom, .. } => {
                *pos -= *denom;
            }
        }
    }

    #[inline]
    pub(crate) fn advance_output(&mut self) {
        match self {
            Self::Float { pos, step } => *pos += *step,
            Self::Rational { pos, numer, .. } | Self::RationalFast { pos, numer, .. } => {
                *pos += *numer;
            }
        }
    }
}
