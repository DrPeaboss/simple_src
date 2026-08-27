use crate::Rational;

/// Monomorphic phase accumulator for polynomial interpolation.
///
/// Splitting the former `PhaseAccum` enum into one concrete type per mode
/// keeps the hot loop free of per-operation tag dispatch: the mode is chosen
/// once at converter construction, then every phase operation is a plain
/// field access that the optimizer can keep in registers.
pub(crate) trait PolynomialPhase {
    /// Fractional interpolation coefficient for the pending output sample.
    fn coef(&self) -> f64;
    /// Whether the next output sample needs a new input sample shifted in.
    fn needs_input_advance(&self) -> bool;
    /// Drop one input interval (consume a step of input position).
    fn consume_input_step(&mut self);
    /// Move to the next output sample position.
    fn advance_output(&mut self);
    /// Prepare after four-tap priming: first output sits at phase zero.
    fn prepare_four_tap_priming(&mut self);
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FloatPhase {
    pos: f64,
    step: f64,
}

impl FloatPhase {
    #[inline]
    pub(crate) fn new(step: f64) -> Self {
        Self { pos: 0.0, step }
    }
}

impl PolynomialPhase for FloatPhase {
    #[inline]
    fn coef(&self) -> f64 {
        self.pos
    }

    #[inline]
    fn needs_input_advance(&self) -> bool {
        self.pos >= 1.0
    }

    #[inline]
    fn consume_input_step(&mut self) {
        self.pos -= 1.0;
    }

    #[inline]
    fn advance_output(&mut self) {
        self.pos += self.step;
    }

    #[inline]
    fn prepare_four_tap_priming(&mut self) {
        self.pos = 0.0;
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RationalPhase {
    pos: usize,
    numer: usize,
    denom: usize,
    recip: f64,
}

impl RationalPhase {
    #[inline]
    pub(crate) fn new(step: Rational) -> Self {
        Self {
            pos: 0,
            numer: *step.numer() as usize,
            denom: *step.denom() as usize,
            recip: (*step.denom() as f64).recip(),
        }
    }
}

impl PolynomialPhase for RationalPhase {
    #[inline]
    fn coef(&self) -> f64 {
        self.pos as f64 * self.recip
    }

    #[inline]
    fn needs_input_advance(&self) -> bool {
        self.pos >= self.denom
    }

    #[inline]
    fn consume_input_step(&mut self) {
        self.pos -= self.denom;
    }

    #[inline]
    fn advance_output(&mut self) {
        self.pos += self.numer;
    }

    #[inline]
    fn prepare_four_tap_priming(&mut self) {
        self.pos = 0;
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RationalFastPhase {
    pos: usize,
    numer: usize,
    denom: usize,
    coef: Vec<f64>,
}

impl RationalFastPhase {
    /// Phase with a precomputed fractional coefficient table (linear path).
    #[inline]
    pub(crate) fn new(step: Rational) -> Self {
        let numer = *step.numer() as usize;
        let denom = *step.denom() as usize;
        let coef = (0..denom).map(|i| i as f64 / denom as f64).collect();
        Self {
            pos: 0,
            numer,
            denom,
            coef,
        }
    }
}

impl PolynomialPhase for RationalFastPhase {
    #[inline]
    fn coef(&self) -> f64 {
        // SAFETY-free variant: pos < denom is an invariant maintained by
        // consume_input_step/advance_output; keep the checked index for now.
        self.coef[self.pos]
    }

    #[inline]
    fn needs_input_advance(&self) -> bool {
        self.pos >= self.denom
    }

    #[inline]
    fn consume_input_step(&mut self) {
        self.pos -= self.denom;
    }

    #[inline]
    fn advance_output(&mut self) {
        self.pos += self.numer;
    }

    #[inline]
    fn prepare_four_tap_priming(&mut self) {
        self.pos = 0;
    }
}

/// Select the concrete phase for `ratio`, mirroring the former enum layout.
#[inline]
pub(crate) fn polynomial_phase(ratio: &crate::Ratio) -> PhaseFor {
    match ratio {
        crate::Ratio::Float(ratio) => PhaseFor::Float(FloatPhase::new(ratio.recip())),
        crate::Ratio::Rational(ratio) => {
            if *ratio.numer() <= crate::ratio::LINEAR_FAST_NUMER_MAX {
                PhaseFor::RationalFast(RationalFastPhase::new(ratio.recip()))
            } else {
                PhaseFor::Rational(RationalPhase::new(ratio.recip()))
            }
        }
    }
}

/// Owned concrete phase, dispatched once per converter construction.
pub(crate) enum PhaseFor {
    Float(FloatPhase),
    Rational(RationalPhase),
    RationalFast(RationalFastPhase),
}

/// Tagged phase accumulator retained for the sinc FIR path.
///
/// Each output sample runs a full FIR convolution, so the per-operation tag
/// dispatch is amortized there; the polynomial (linear/cubic) path uses the
/// monomorphic [`PolynomialPhase`] implementations above instead.
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
        }
    }

    #[inline]
    pub(crate) fn pos_float(&self) -> f64 {
        match self {
            Self::Float { pos, .. } => *pos,
            Self::Rational { pos, denom, .. } => *pos as f64 / *denom as f64,
        }
    }

    #[inline]
    pub(crate) fn pos_usize(&self) -> usize {
        match self {
            Self::Float { .. } => unreachable!("pos_usize on float phase"),
            Self::Rational { pos, .. } => *pos,
        }
    }

    #[inline]
    pub(crate) fn needs_input_advance(&self) -> bool {
        match self {
            Self::Float { pos, .. } => *pos >= 1.0,
            Self::Rational { pos, denom, .. } => *pos >= *denom,
        }
    }

    #[inline]
    pub(crate) fn consume_input_step(&mut self) {
        match self {
            Self::Float { pos, .. } => *pos -= 1.0,
            Self::Rational { pos, denom, .. } => *pos -= *denom,
        }
    }

    #[inline]
    pub(crate) fn advance_output(&mut self) {
        match self {
            Self::Float { pos, step } => *pos += *step,
            Self::Rational { pos, numer, .. } => *pos += *numer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_advance_cycle() {
        let mut phase = FloatPhase::new(0.5);
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
    fn rational_fast_coef_table() {
        let phase = RationalFastPhase::new(Rational::new(1, 2));
        assert_eq!(phase.coef, vec![0.0, 0.5]);
        assert_eq!(phase.denom, 2);
    }

    #[test]
    fn rational_has_no_coef_table() {
        let phase = RationalPhase::new(Rational::new(1, 2));
        assert_eq!(phase.denom, 2);
    }

    #[test]
    fn four_tap_priming_starts_at_phase_zero() {
        let mut phase = RationalFastPhase::new(Rational::new(1, 2));
        phase.prepare_four_tap_priming();
        assert!(!phase.needs_input_advance());
        assert_eq!(phase.coef(), 0.0);

        let mut float_phase = FloatPhase::new(0.5);
        float_phase.prepare_four_tap_priming();
        assert!(!float_phase.needs_input_advance());
        assert_eq!(float_phase.coef(), 0.0);
    }
}
