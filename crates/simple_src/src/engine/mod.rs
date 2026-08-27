mod buffer;
mod fir;
mod phase;
mod polynomial;
mod state;

pub(crate) use buffer::FirTap;
pub(crate) use fir::fir_next_sample;
pub(crate) use phase::{
    FloatPhase, PhaseAccum, PhaseFor, PolynomialPhase, RationalFastPhase, RationalPhase,
    polynomial_phase,
};
pub(crate) use polynomial::{FourTap, PolynomialKind, polynomial_next_sample};
pub(crate) use state::{FirState, LinearState};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rational;

    #[test]
    fn phase_selects_concrete_type() {
        assert!(matches!(
            polynomial_phase(&crate::Ratio::Float(2.0)),
            PhaseFor::Float(_)
        ));
        assert!(matches!(
            polynomial_phase(&crate::Ratio::Rational(Rational::new(160, 147))),
            PhaseFor::RationalFast(_)
        ));
        assert!(matches!(
            polynomial_phase(&crate::Ratio::Rational(Rational::new(20000, 19999))),
            PhaseFor::Rational(_)
        ));
    }

    #[test]
    fn linear_state_starts_priming_fir_does_not() {
        assert!(LinearState::new_cubic().is_priming());
        assert_eq!(FirState::new(), FirState::Running);
    }
}
