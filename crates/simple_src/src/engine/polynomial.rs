use super::{LinearState, PolynomialPhase};

pub(crate) trait PolynomialTap {
    fn push_priming(&mut self, sample: f64);
    fn push_resume(&mut self, sample: f64);
    fn shift(&mut self, sample: f64);
    fn advance_left(&mut self);
    fn interpolate(&self, coef: f64) -> f64;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FourTap {
    data: [f64; 4],
}

impl FourTap {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub(crate) fn shift(&mut self, sample: f64) {
        self.data[0] = self.data[1];
        self.data[1] = self.data[2];
        self.data[2] = self.data[3];
        self.data[3] = sample;
    }

    #[inline]
    pub(crate) fn advance_left(&mut self) {
        self.data[0] = self.data[1];
        self.data[1] = self.data[2];
        self.data[2] = self.data[3];
    }

    #[inline]
    pub(crate) fn set_last(&mut self, sample: f64) {
        self.data[3] = sample;
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.data.iter().all(|&x| x == 0.0)
    }

    #[inline]
    pub(crate) fn catmull_rom(&self, t: f64) -> f64 {
        let [y0, y1, y2, y3] = self.data;
        let t2 = t * t;
        let t3 = t2 * t;
        0.5 * ((2.0 * y1)
            + (-y0 + y2) * t
            + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2
            + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3)
    }
}

impl PolynomialTap for FourTap {
    #[inline]
    fn push_priming(&mut self, sample: f64) {
        self.shift(sample);
    }

    #[inline]
    fn push_resume(&mut self, sample: f64) {
        self.set_last(sample);
    }

    #[inline]
    fn shift(&mut self, sample: f64) {
        FourTap::shift(self, sample);
    }

    #[inline]
    fn advance_left(&mut self) {
        FourTap::advance_left(self);
    }

    #[inline]
    fn interpolate(&self, coef: f64) -> f64 {
        self.catmull_rom(coef)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PolynomialKind {
    FourTap,
}

#[inline]
pub(crate) fn polynomial_next_sample<I, P: PolynomialPhase, T: PolynomialTap>(
    kind: PolynomialKind,
    state: &mut LinearState,
    phase: &mut P,
    taps: &mut T,
    iter: &mut I,
) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    loop {
        match *state {
            LinearState::Priming { filled, need } => {
                let s = iter.next()?;
                taps.push_priming(s);
                let filled = filled + 1;
                if filled >= need {
                    let _ = kind;
                    phase.prepare_four_tap_priming();
                    *state = LinearState::Running;
                } else {
                    *state = LinearState::Priming { filled, need };
                }
            }
            LinearState::Running => {
                while phase.needs_input_advance() {
                    phase.consume_input_step();
                    if let Some(s) = iter.next() {
                        taps.shift(s);
                    } else {
                        taps.advance_left();
                        *state = state.on_input_exhausted();
                        return None;
                    }
                }
                let interp = taps.interpolate(phase.coef());
                phase.advance_output();
                return Some(interp);
            }
            LinearState::Suspended => {
                let s = iter.next()?;
                taps.push_resume(s);
                *state = state.on_input_resumed();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rational;
    use crate::engine::{FloatPhase, RationalFastPhase};

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

    #[test]
    fn four_tap_catmull_rom_endpoints() {
        let taps = FourTap {
            data: [0.0, 1.0, 4.0, 9.0],
        };
        assert!((taps.catmull_rom(0.0) - 1.0).abs() < 1e-12);
        assert!((taps.catmull_rom(1.0) - 4.0).abs() < 1e-12);
    }
}
