mod buffer;
mod fir;
mod phase;
mod polynomial;
mod state;

pub(crate) use buffer::{FirTap, TwoTap};
pub(crate) use fir::fir_next_sample;
pub(crate) use phase::PhaseAccum;
pub(crate) use polynomial::{FourTap, PolynomialKind, polynomial_next_sample};
pub(crate) use state::{FirState, LinearState};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rational;

    #[test]
    fn phase_float_advance_cycle() {
        let mut phase = PhaseAccum::float(0.5);
        assert!(!phase.needs_input_advance());
        assert_eq!(phase.coef(), 0.0);
        phase.advance_output();
        assert_eq!(phase.coef(), 0.5);
        assert!(!phase.needs_input_advance());
        phase.advance_output();
        assert!(phase.needs_input_advance());
        phase.consume_input_step();
        assert_eq!(phase.coef(), 0.0);
    }

    #[test]
    fn phase_rational_fast_linear_coef_table() {
        let step = Rational::new(1, 2);
        let phase = PhaseAccum::rational_fast_linear(step);
        match phase {
            PhaseAccum::RationalFast { denom, coef, .. } => {
                assert_eq!(denom, 2);
                assert_eq!(coef, vec![0.0, 0.5]);
            }
            _ => panic!("expected RationalFast"),
        }
    }

    #[test]
    fn phase_rational_has_no_coef_table() {
        let step = Rational::new(1, 2);
        match PhaseAccum::rational(step) {
            PhaseAccum::Rational { denom, .. } => assert_eq!(denom, 2),
            _ => panic!("expected Rational"),
        }
    }

    #[test]
    fn two_tap_advance_left_on_suspend() {
        let mut taps = TwoTap::new();
        taps.shift(1.0);
        taps.shift(2.0);
        taps.advance_left();
        assert_eq!(taps.interpolate(0.0), 2.0);
        assert_eq!(taps.interpolate(1.0), 2.0);
        taps.set_second(3.0);
        assert_eq!(taps.interpolate(0.0), 2.0);
        assert_eq!(taps.interpolate(1.0), 3.0);
    }

    #[test]
    fn linear_state_starts_priming_fir_does_not() {
        assert!(LinearState::new_linear().is_priming());
        assert_eq!(FirState::new(), FirState::Running);
    }
}
