#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearState {
    Priming { filled: u8, need: u8 },
    Running,
    Suspended,
}

impl LinearState {
    #[inline]
    pub(crate) fn new_linear() -> Self {
        Self::Priming { filled: 0, need: 1 }
    }

    #[inline]
    pub(crate) fn new_cubic() -> Self {
        Self::Priming { filled: 0, need: 3 }
    }

    #[inline]
    pub(crate) fn is_priming(self) -> bool {
        matches!(self, Self::Priming { .. })
    }

    #[inline]
    pub(crate) fn on_input_exhausted(self) -> Self {
        Self::Suspended
    }

    #[inline]
    pub(crate) fn on_input_resumed(self) -> Self {
        Self::Running
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FirState {
    Running,
    Suspended,
}

impl FirState {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::Running
    }

    #[inline]
    pub(crate) fn on_input_exhausted(self) -> Self {
        Self::Suspended
    }

    #[inline]
    pub(crate) fn on_input_resumed(self) -> Self {
        Self::Running
    }
}
