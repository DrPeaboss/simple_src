//! Error-path coverage: what the public API rejects and why.
//!
//! P0 hardening. Every `Result`-returning public constructor is exercised
//! with invalid inputs, every [`Error`] variant's `Display` text is pinned,
//! the `Quality` preset table is verified against the documented values, and
//! the `SincPath::Auto` selection logic is pinned.

use std::f64::consts::PI;

use simple_src::{
    Convert, ConvertMode, Error, Kernel, Quality, SincPath, SrcManager, process_planar,
};

fn build_err(b: simple_src::SrcBuilder) -> Error {
    b.build().err().unwrap()
}

fn generic(ratio: f64, atten: f64, quan: u32, tw: f64) -> simple_src::SrcBuilder {
    SrcManager::builder()
        .ratio(ratio)
        .generic()
        .attenuation(atten)
        .quantify(quan)
        .trans_width(tw)
}

fn same_f64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

// ---------------------------------------------------------------------------
// Ratio bounds
// ---------------------------------------------------------------------------

#[test]
fn ratio_bounds_accept_edges() {
    SrcManager::with_ratio(1.0 / 16.0).unwrap();
    SrcManager::with_ratio(16.0).unwrap();
}

#[test]
fn ratio_bounds_reject_out_of_range() {
    for bad in [
        0.0624,
        16.01,
        0.0,
        -2.0,
        32.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let err = SrcManager::with_ratio(bad).err().unwrap();
        assert!(
            matches!(err, Error::UnsupportedRatio { ratio } if same_f64(ratio, bad)),
            "ratio {bad} produced {err:?}"
        );
    }
}

#[test]
fn sample_rate_rejects_zero_and_out_of_range_ratio() {
    for (old, new) in [(0u32, 48000u32), (48000, 0)] {
        let err = SrcManager::builder()
            .sample_rate(old, new)
            .kernel(Kernel::Linear)
            .build()
            .err()
            .unwrap();
        assert!(
            matches!(
                err,
                Error::InvalidParam {
                    name: "sample_rate",
                    ..
                }
            ),
            "{err:?}"
        );
    }
    // 800000/48000 = 16.67 > 16.
    let err = SrcManager::builder()
        .sample_rate(48000, 800_000)
        .kernel(Kernel::Linear)
        .build()
        .err()
        .unwrap();
    assert!(matches!(err, Error::UnsupportedRatio { .. }), "{err:?}");
}

// ---------------------------------------------------------------------------
// Missing parameter combinations
// ---------------------------------------------------------------------------

#[test]
fn missing_parameter_combos() {
    // No ratio and no sample rate.
    let err = SrcManager::builder().build().err().unwrap();
    assert!(
        matches!(err, Error::MissingParam(name) if name.contains("ratio")),
        "{err:?}"
    );

    // Sinc generic without quantify.
    let err = SrcManager::builder()
        .ratio(2.0)
        .generic()
        .attenuation(48.0)
        .trans_width(0.2)
        .build()
        .err()
        .unwrap();
    assert!(matches!(err, Error::MissingParam("quantify")), "{err:?}");

    // Ratio + quantify, but no filter parameters at all.
    let err = SrcManager::builder()
        .ratio(2.0)
        .generic()
        .quantify(128)
        .build()
        .err()
        .unwrap();
    assert!(matches!(err, Error::MissingParam(_)), "{err:?}");
    assert!(err.to_string().contains("attenuation"), "{err:?}");

    // Raw filter spec incomplete: order without beta/cutoff and no atten.
    let err = SrcManager::builder()
        .ratio(2.0)
        .generic()
        .quantify(128)
        .order(64)
        .build()
        .err()
        .unwrap();
    assert!(matches!(err, Error::MissingParam(_)), "{err:?}");

    // Fast path also requires a filter spec.
    let err = SrcManager::builder()
        .ratio(2.0)
        .fast()
        .quantify(128)
        .build()
        .err()
        .unwrap();
    assert!(matches!(err, Error::MissingParam(_)), "{err:?}");
}

// ---------------------------------------------------------------------------
// Invalid parameter ranges
// ---------------------------------------------------------------------------

#[test]
fn invalid_parameter_ranges() {
    // quantify outside [1, 16384]
    for bad in [0u32, 16_385] {
        let err = build_err(generic(2.0, 48.0, bad, 0.2));
        assert!(
            matches!(
                err,
                Error::InvalidParam {
                    name: "quantify",
                    ..
                }
            ),
            "{err:?}"
        );
    }
    // order outside [1, 2048]
    for bad in [0u32, 2049] {
        let b = SrcManager::builder()
            .ratio(2.0)
            .generic()
            .quantify(8)
            .order(bad)
            .kaiser_beta(7.0)
            .cutoff(0.8);
        let err = build_err(b);
        assert!(
            matches!(err, Error::InvalidParam { name: "order", .. }),
            "{err:?}"
        );
    }
    // kaiser_beta outside [0, 20]
    for bad in [-0.01, 20.01, f64::NAN] {
        let b = SrcManager::builder()
            .ratio(2.0)
            .generic()
            .quantify(8)
            .order(8)
            .kaiser_beta(bad)
            .cutoff(0.8);
        let err = build_err(b);
        assert!(
            matches!(
                err,
                Error::InvalidParam {
                    name: "kaiser_beta",
                    ..
                }
            ),
            "{err:?}"
        );
    }
    // cutoff outside [0.01, 1.0]
    for bad in [0.0, 1.01, f64::NAN] {
        let b = SrcManager::builder()
            .ratio(2.0)
            .generic()
            .quantify(8)
            .order(8)
            .kaiser_beta(7.0)
            .cutoff(bad);
        let err = build_err(b);
        assert!(
            matches!(err, Error::InvalidParam { name: "cutoff", .. }),
            "{err:?}"
        );
    }
    // attenuation outside [12, 180]
    for bad in [11.0, 181.0, f64::NAN] {
        let err = build_err(generic(2.0, bad, 8, 0.2));
        assert!(
            matches!(
                err,
                Error::InvalidParam {
                    name: "attenuation",
                    ..
                }
            ),
            "{err:?}"
        );
    }
    // trans_width outside [0.01, 1.0]
    for bad in [0.0, 1.01, f64::NAN] {
        let err = build_err(generic(2.0, 48.0, 8, bad));
        assert!(
            matches!(
                err,
                Error::InvalidParam {
                    name: "trans_width",
                    ..
                }
            ),
            "{err:?}"
        );
    }
    // pass_width maps to trans_width = 1 - width and is validated the same way.
    let err = SrcManager::builder()
        .ratio(2.0)
        .generic()
        .attenuation(48.0)
        .quantify(8)
        .pass_width(1.0)
        .build()
        .err()
        .unwrap();
    assert!(
        matches!(
            err,
            Error::InvalidParam {
                name: "trans_width",
                ..
            }
        ),
        "{err:?}"
    );
    // pass_freq at Nyquist drives trans_width to 0 -> the same error.
    let err = SrcManager::builder()
        .sample_rate(44100, 48000)
        .generic()
        .attenuation(48.0)
        .quantify(8)
        .pass_freq(22050)
        .build()
        .err()
        .unwrap();
    assert!(
        matches!(
            err,
            Error::InvalidParam {
                name: "trans_width",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn parameter_boundaries_are_accepted() {
    // quantify boundary with a tiny raw filter (order 1) keeps the table cheap.
    for quan in [1u32, 16_384] {
        SrcManager::builder()
            .ratio(2.0)
            .generic()
            .quantify(quan)
            .order(1)
            .kaiser_beta(7.0)
            .cutoff(0.8)
            .build()
            .unwrap();
    }
    // order boundary with quantify 1 (2 rows only).
    for order in [1u32, 2048] {
        SrcManager::builder()
            .ratio(2.0)
            .generic()
            .quantify(1)
            .order(order)
            .kaiser_beta(7.0)
            .cutoff(0.8)
            .build()
            .unwrap();
    }
    // attenuation boundary (quan 8 keeps rows tiny; tw 0.2 bounds the order).
    for atten in [12.0, 180.0] {
        generic(2.0, atten, 8, 0.2).build().unwrap();
    }
    // trans_width boundary with low atten keeps the order small.
    for tw in [0.01, 1.0] {
        generic(2.0, 12.0, 8, tw).build().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Fast path availability
// ---------------------------------------------------------------------------

#[test]
fn fast_unavailable_cases() {
    // Irrational/float ratio: no numerator to report.
    let err = SrcManager::builder()
        .ratio(PI)
        .fast()
        .attenuation(48.0)
        .trans_width(0.2)
        .build()
        .err()
        .unwrap();
    assert!(
        matches!(err, Error::FastUnavailable { numer: None, .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("Generic"), "{err:?}");

    // Rational with numerator > 1024: the numerator is reported.
    let err = SrcManager::builder()
        .sample_rate(1024, 1025)
        .fast()
        .attenuation(48.0)
        .trans_width(0.2)
        .build()
        .err()
        .unwrap();
    assert!(
        matches!(
            err,
            Error::FastUnavailable {
                numer: Some(1025),
                ..
            }
        ),
        "{err:?}"
    );
    assert!(err.to_string().contains("1025"), "{err:?}");
}

// ---------------------------------------------------------------------------
// SincPath::Auto selection
// ---------------------------------------------------------------------------

#[test]
fn auto_path_picks_fast_for_eligible_ratio() {
    let m = SrcManager::builder()
        .sample_rate(44100, 48000)
        .quality(Quality::Bit16Fast)
        .pass_freq(20000)
        .build()
        .unwrap();
    assert_eq!(m.mode(), ConvertMode::RationalFast);
    assert_eq!(m.ratio_parts(), Some((160, 147)));
}

#[test]
fn auto_path_falls_back_to_generic_for_large_numerator() {
    // 1025/1024 is exactly representable, so it stays rational, but its
    // numerator exceeds the fast-path limit of 1024.
    let m = SrcManager::builder()
        .ratio(1025.0 / 1024.0)
        .attenuation(96.0)
        .quantify(128)
        .trans_width(0.2)
        .build()
        .unwrap();
    assert_eq!(m.mode(), ConvertMode::Rational);
    assert_eq!(m.ratio_parts(), Some((1025, 1024)));
    let order = m.order().unwrap();
    assert_eq!(
        m.lut_len(),
        Some((128 + 1) * (order as usize + 1)),
        "generic lut = (quan + 1) * (order + 1)"
    );
}

#[test]
fn auto_path_uses_float_phase_for_irrational() {
    let m = SrcManager::builder()
        .ratio(PI)
        .attenuation(96.0)
        .quantify(128)
        .trans_width(0.2)
        .build()
        .unwrap();
    assert_eq!(m.mode(), ConvertMode::Float);
    assert_eq!(m.ratio_parts(), None);
}

#[test]
fn auto_matches_explicit_fast_for_44100_48000() {
    let auto = SrcManager::builder()
        .sample_rate(44100, 48000)
        .quality(Quality::Bit16Fast)
        .pass_freq(20000)
        .build()
        .unwrap();
    let fast = SrcManager::builder()
        .sample_rate(44100, 48000)
        .quality(Quality::Bit16Fast)
        .pass_freq(20000)
        .sinc_path(SincPath::Fast)
        .build()
        .unwrap();
    assert_eq!(auto.mode(), fast.mode());
    assert_eq!(auto.order(), fast.order());
    assert_eq!(auto.latency(), fast.latency());
    let input: Vec<f64> = (0..256).map(|i| (i as f64 * 0.13).sin()).collect();
    let a = auto.convert(&input);
    let f = fast.convert(&input);
    assert_eq!(a.len(), f.len());
    for (x, y) in a.iter().zip(f.iter()) {
        assert!((x - y).abs() < 1e-12);
    }
}

#[test]
fn sinc_path_is_ignored_for_linear_and_cubic() {
    let l = SrcManager::builder()
        .kernel(Kernel::Linear)
        .sinc_path(SincPath::Fast)
        .ratio(2.0)
        .build()
        .unwrap();
    assert_eq!(l.latency(), 0);
    assert_eq!(l.order(), None);

    let c = SrcManager::builder()
        .kernel(Kernel::Cubic)
        .sinc_path(SincPath::Generic)
        .ratio(2.0)
        .build()
        .unwrap();
    assert_eq!(c.order(), None);
    assert_eq!(c.lut_len(), None);
}

// ---------------------------------------------------------------------------
// Quality preset table
// ---------------------------------------------------------------------------

#[test]
fn quality_preset_table_matches_documented_values() {
    let table = [
        (Quality::Bit8Fast, 48.0, 8),
        (Quality::Bit8Medium, 60.0, 16),
        (Quality::Bit8Better, 72.0, 32),
        (Quality::Bit16Lower, 84.0, 64),
        (Quality::Bit16Fast, 96.0, 128),
        (Quality::Bit16Medium, 108.0, 256),
        (Quality::Bit16Better, 120.0, 512),
        (Quality::Bit24Lower, 132.0, 1024),
        (Quality::Bit24Fast, 144.0, 2048),
        (Quality::Bit24Medium, 156.0, 4096),
        (Quality::Bit24Better, 168.0, 8192),
    ];
    for (q, atten, quan) in table {
        assert_eq!(q.attenuation(), atten, "{q:?}");
        assert_eq!(q.quantify(), quan, "{q:?}");
    }
}

#[test]
fn every_quality_preset_builds_a_generic_sinc() {
    let presets = [
        Quality::Bit8Fast,
        Quality::Bit8Medium,
        Quality::Bit8Better,
        Quality::Bit16Lower,
        Quality::Bit16Fast,
        Quality::Bit16Medium,
        Quality::Bit16Better,
        Quality::Bit24Lower,
        Quality::Bit24Fast,
        Quality::Bit24Medium,
        Quality::Bit24Better,
    ];
    for q in presets {
        let m = SrcManager::builder()
            .ratio(2.0)
            .generic()
            .quality(q)
            .trans_width(0.2)
            .build()
            .unwrap();
        let order = m.order().unwrap();
        assert_eq!(
            m.lut_len(),
            Some((q.quantify() as usize + 1) * (order as usize + 1)),
            "{q:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Error Display text
// ---------------------------------------------------------------------------

#[test]
fn error_display_pins_all_variants() {
    let unsupported = SrcManager::with_ratio(32.0).err().unwrap().to_string();
    assert!(
        unsupported.contains("unsupported conversion ratio 32"),
        "{unsupported}"
    );

    let invalid = build_err(generic(2.0, 48.0, 0, 0.2)).to_string();
    assert!(
        invalid.contains("quantify") && invalid.contains("[1, 16384]"),
        "{invalid}"
    );

    let missing = SrcManager::builder()
        .ratio(2.0)
        .generic()
        .quantify(128)
        .build()
        .err()
        .unwrap()
        .to_string();
    assert!(missing.contains("missing"), "{missing}");

    let manager = SrcManager::with_ratio(2.0).unwrap();
    let mut cvs = [manager.converter(), manager.converter()];
    let left = [1.0, 2.0];
    let mut out_l = [0.0; 4];
    let inputs: [&[f64]; 1] = [&left];
    let mut outputs: [&mut [f64]; 1] = [&mut out_l];
    let bad_channels = process_planar(&mut cvs, &inputs, &mut outputs)
        .err()
        .unwrap()
        .to_string();
    assert!(
        bad_channels.contains("planar input count is 1, expected 2 converters"),
        "{bad_channels}"
    );

    let right = [1.0];
    let mut out_r = [0.0; 2];
    let inputs2: [&[f64]; 2] = [&left, &right];
    let mut outputs2: [&mut [f64]; 2] = [&mut out_l, &mut out_r];
    let bad_len = process_planar(&mut cvs, &inputs2, &mut outputs2)
        .err()
        .unwrap()
        .to_string();
    assert!(bad_len.contains("length mismatch"), "{bad_len}");

    let mut fresh = [manager.converter(), manager.converter()];
    // Drift channel 1 ahead of channel 0 *before* the call; the input
    // lengths stay equal so only state drift can produce the mismatch.
    let mut tmp = [0.0; 8];
    fresh[1].process_block(&[1.0; 4], &mut tmp);
    let eq = [1.0; 8];
    let mut ol = [0.0; 16];
    let mut os = [0.0; 16];
    let ins: [&[f64]; 2] = [&eq, &eq];
    let mut outs: [&mut [f64]; 2] = [&mut ol, &mut os];
    let drift = process_planar(&mut fresh, &ins, &mut outs)
        .err()
        .unwrap()
        .to_string();
    assert!(drift.contains("not in lockstep"), "{drift}");

    let fast = SrcManager::builder()
        .ratio(PI)
        .fast()
        .attenuation(48.0)
        .trans_width(0.2)
        .build()
        .err()
        .unwrap()
        .to_string();
    assert!(
        fast.contains("fast polyphase") && fast.contains("Generic"),
        "{fast}"
    );
}
