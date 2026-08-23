#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearState {
    Priming,
    Running,
    Suspended,
}

impl LinearState {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::Priming
    }

    #[inline]
    pub(crate) fn is_priming(self) -> bool {
        matches!(self, Self::Priming)
    }

    #[inline]
    pub(crate) fn on_input_exhausted(self) -> Self {
        Self::Suspended
    }

    #[inline]
    pub(crate) fn on_input_resumed(self) -> Self {
        Self::Running
    }

    #[inline]
    pub(crate) fn finish_priming(self) -> Self {
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
