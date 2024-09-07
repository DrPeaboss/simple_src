use super::NextSample;

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
    pub fn new(step: f64) -> Self {
        Self {
            step,
            pos: 0.0,
            last_in: [0.0; 2],
            state: State::First,
        }
    }
}

impl NextSample for Converter {
    #[inline]
    fn next_sample<F>(&mut self, f: &mut F) -> Option<f64>
    where
        F: FnMut() -> Option<f64>,
    {
        loop {
            match self.state {
                State::First => {
                    if let Some(s) = f() {
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
                        if let Some(s) = f() {
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
                    if let Some(s) = f() {
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
    pub fn new(ratio: f64) -> Self {
        Self { ratio }
    }

    #[inline]
    pub fn converter(&self) -> Converter {
        Converter::new(self.ratio.recip())
    }
}
