use super::{FirState, FirTap, PhaseAccum};

pub(crate) fn fir_next_sample<I>(
    state: &mut FirState,
    phase: &mut PhaseAccum,
    taps: &mut FirTap,
    iter: &mut I,
    mut interpolate: impl FnMut(&PhaseAccum, &FirTap) -> f64,
) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    loop {
        match *state {
            FirState::Running => {
                while phase.needs_input_advance() {
                    phase.consume_input_step();
                    if let Some(s) = iter.next() {
                        taps.shift(s);
                    } else {
                        *state = state.on_input_exhausted();
                        return None;
                    }
                }
                let interp = interpolate(phase, taps);
                phase.advance_output();
                return Some(interp);
            }
            FirState::Suspended => {
                let s = iter.next()?;
                taps.shift(s);
                *state = state.on_input_resumed();
            }
        }
    }
}
