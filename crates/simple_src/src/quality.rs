/// Recommended sinc quality presets.
///
/// Stop-band attenuation `A` is used by both Generic and Fast constructors.
/// LUT quantify `Q` is used only by Generic constructors (`new`, `with_quality`,
/// and the default builder path). Fast constructors (`fast`, `fast_with_quality`)
/// ignore `quantify` and build a polyphase table instead.
///
/// The pairing of stop-band attenuation and LUT quantify follows
/// `Q ≈ 2^(A/12 - 1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Bit8Fast,
    Bit8Medium,
    Bit8Better,
    Bit16Lower,
    Bit16Fast,
    Bit16Medium,
    Bit16Better,
    Bit24Lower,
    Bit24Fast,
    Bit24Medium,
    Bit24Better,
}

impl Quality {
    /// Stop-band attenuation in dB. Used by both Generic and Fast sinc paths.
    pub fn attenuation(self) -> f64 {
        match self {
            Self::Bit8Fast => 48.0,
            Self::Bit8Medium => 60.0,
            Self::Bit8Better => 72.0,
            Self::Bit16Lower => 84.0,
            Self::Bit16Fast => 96.0,
            Self::Bit16Medium => 108.0,
            Self::Bit16Better => 120.0,
            Self::Bit24Lower => 132.0,
            Self::Bit24Fast => 144.0,
            Self::Bit24Medium => 156.0,
            Self::Bit24Better => 168.0,
        }
    }

    /// Filter LUT quantify number. Used only by Generic sinc constructors.
    pub fn quantify(self) -> u32 {
        match self {
            Self::Bit8Fast => 8,
            Self::Bit8Medium => 16,
            Self::Bit8Better => 32,
            Self::Bit16Lower => 64,
            Self::Bit16Fast => 128,
            Self::Bit16Medium => 256,
            Self::Bit16Better => 512,
            Self::Bit24Lower => 1024,
            Self::Bit24Fast => 2048,
            Self::Bit24Medium => 4096,
            Self::Bit24Better => 8192,
        }
    }
}
