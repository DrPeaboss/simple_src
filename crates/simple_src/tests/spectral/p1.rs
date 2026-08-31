//! P1 hardening: rate pairs and interpolation paths the original baselines
//! never exercised.
//!
//! - 96k -> 44.1k / 48k downsampling (ratio ~0.46-0.5): different stopband
//!   geometry than the near-unity 44100/48000 baselines.
//! - 44.1k -> 96k (ratio 2.177): >2x upsample polyphase regime.
//! - Irrational ratio pi on the generic path: the float-phase interpolation
//!   code path, which had zero spectral coverage.
//! - All 11 quality presets get a generic-path THD+N red line; the
//!   documented ~0.15 dB generic ripple at quantify = 8 is pinned against the
//!   flat fast path.

use super::*;
use simple_src::{ConvertMode, Quality, SrcManager};

/// Alias residue for an arbitrary rate pair: `tone` above the output Nyquist
/// folds back to `fs_out - tone`; the residue is peak-detected in [lo, hi].
fn alias_at(
    manager: &SrcManager,
    fs_in: f64,
    fs_out: f64,
    tone: f64,
    lo: f64,
    hi: f64,
    name: &str,
) -> f64 {
    let (_, input) = binned_tone(fs_in, fs_in, tone, TONE_AMP, input_len_for(fs_in, fs_out));
    let (db, bin_hz) = analyze(manager, &input, fs_out, name, &[(tone, "> Nyquist".into())]);
    let (residue, at) = peak_in(&db, bin_hz, lo, hi);
    println!(
        "{name:<26} residue {residue:>7.2} dBFS at {:.2} kHz",
        at / 1000.0
    );
    residue
}

#[test]
fn high_ratio_alias_96k_down() {
    // 96k -> 44.1k: 25 kHz tone folds to 19.1 kHz (output Nyquist 22.05 kHz).
    // 96k -> 48k:   26 kHz tone folds to 22.0 kHz (output Nyquist 24 kHz).
    let sinc_at = |fs_in: u32, fs_out: u32, q: Quality, fast: bool| -> SrcManager {
        let mut b = SrcManager::builder()
            .sample_rate(fs_in, fs_out)
            .quality(q)
            .trans_width(TW);
        b = if fast { b.fast() } else { b.generic() };
        b.build().unwrap()
    };

    // Red lines initial pass (measured values in comments; headroom 6+ dB).
    for (fs_in, fs_out, tone, lo, hi) in [
        (96_000.0, 44_100.0, 25_000.0, 15_000.0, 21_500.0),
        (96_000.0, 48_000.0, 26_000.0, 17_000.0, 23_500.0),
    ] {
        let g96 = sinc_at(fs_in as u32, fs_out as u32, Quality::Bit16Fast, false);
        let f96 = sinc_at(fs_in as u32, fs_out as u32, Quality::Bit16Fast, true);
        let f144 = sinc_at(fs_in as u32, fs_out as u32, Quality::Bit24Fast, true);
        let tag = (fs_in / 1000.0) as u32;
        // Measured residues (tone @ -6 dBFS): g96/f96 -120.8, f144 -167.8
        // (44.1k) and -164.4 (48k). Lines carry >= 6 dB headroom.
        assert!(
            alias_at(
                &g96,
                fs_in,
                fs_out,
                tone,
                lo,
                hi,
                &format!("{tag}k_down_g96")
            ) < -110.0,
            "96k down generic stopband regressed"
        );
        assert!(
            alias_at(
                &f96,
                fs_in,
                fs_out,
                tone,
                lo,
                hi,
                &format!("{tag}k_down_f96")
            ) < -110.0,
            "96k down fast stopband regressed"
        );
        assert!(
            alias_at(
                &f144,
                fs_in,
                fs_out,
                tone,
                lo,
                hi,
                &format!("{tag}k_down_f144")
            ) < -150.0,
            "96k down 144 dB stopband regressed"
        );
    }

    // Cheap kernels do not anti-alias; sanity windows only.
    let lin = SrcManager::with_sample_rate(96_000, 44_100).unwrap();
    let r = alias_at(
        &lin,
        96_000.0,
        44_100.0,
        25_000.0,
        15_000.0,
        21_500.0,
        "96k_down_linear",
    );
    assert!(
        r > -25.0 && r < 0.0,
        "linear alias residue moved: {r:.2} dBFS"
    );
    let cub = SrcManager::builder()
        .sample_rate(96_000, 44_100)
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();
    let r = alias_at(
        &cub,
        96_000.0,
        44_100.0,
        25_000.0,
        15_000.0,
        21_500.0,
        "96k_down_cubic",
    );
    assert!(r < -6.0, "cubic alias residue moved: {r:.2} dBFS");
}

/// THD+N for an arbitrary rate pair (997 Hz @ -6 dBFS).
fn thdn_at(manager: &SrcManager, fs_in: f64, fs_out: f64, name: &str) -> analysis::ThdnMetrics {
    let (f, input) = binned_tone(fs_in, fs_out, 997.0, TONE_AMP, input_len_for(fs_in, fs_out));
    let (db, bin_hz) = analyze(manager, &input, fs_out, name, &[(f, "997 Hz".into())]);
    thdn(&db, bin_hz, f)
}

#[test]
fn high_ratio_thdn_96k_up() {
    // 44.1k -> 96k (ratio 2.177): a >2x upsample is a different polyphase
    // regime than the 44100/48000 baselines and deserves its own lines.
    let cases: Vec<(SrcManager, &str)> = vec![
        (
            SrcManager::builder()
                .sample_rate(44_100, 96_000)
                .quality(Quality::Bit16Fast)
                .trans_width(TW)
                .fast()
                .build()
                .unwrap(),
            "sinc_f96_up96k",
        ),
        (
            SrcManager::builder()
                .sample_rate(44_100, 96_000)
                .quality(Quality::Bit24Fast)
                .trans_width(TW)
                .fast()
                .build()
                .unwrap(),
            "sinc_f144_up96k",
        ),
    ];
    for (m, name) in cases {
        let mt = thdn_at(&m, 44_100.0, 96_000.0, name);
        println!(
            "{name:<16} THD+N {:.2} dB, spur {:.2} dB",
            mt.thd_plus_n_db, mt.max_spur_dbfs
        );
        assert!(
            mt.fundamental_dbfs > -6.0 - FUNDAMENTAL_TOLERANCE_DB
                && mt.fundamental_dbfs < -6.0 + FUNDAMENTAL_TOLERANCE_DB,
            "{name}: fundamental {:.3}",
            mt.fundamental_dbfs
        );
        assert!(
            mt.thd_plus_n_db < -120.0,
            "{name}: THD+N {}",
            mt.thd_plus_n_db
        );
    }
}

/// Parameterized flatness (tones given in the *input* domain, levels taken
/// from the output spectrum).
fn flatness_at(
    manager: &SrcManager,
    fs_in: f64,
    fs_out: f64,
    tones: &[f64],
    name: &str,
) -> Vec<f64> {
    let amp = 10.0f64.powf(FLAT_AMP_DBFS / 20.0);
    let bin_hz = fs_out / FFT_N as f64;
    let freqs: Vec<f64> = tones
        .iter()
        .map(|&f| (f / bin_hz).round() * bin_hz)
        .collect();
    let input = multi_tone(fs_in, &freqs, amp, input_len_for(fs_in, fs_out));
    let markers: Vec<(f64, String)> = freqs.iter().map(|&f| (f, format!("{:.0} Hz", f))).collect();
    let (db, bin_hz_out) = analyze(manager, &input, fs_out, name, &markers);
    freqs
        .iter()
        .map(|&f| {
            let (level, at) = peak_in(&db, bin_hz_out, f * 0.99, f * 1.01);
            assert!((at - f).abs() < f * 0.01, "{name}: no tone at {f:.0} Hz");
            level - FLAT_AMP_DBFS
        })
        .collect()
}

#[test]
fn float_ratio_pi_baseline() {
    let m = SrcManager::builder()
        .ratio(std::f64::consts::PI)
        .generic()
        .quality(Quality::Bit16Fast)
        .trans_width(TW)
        .build()
        .unwrap();
    assert_eq!(m.mode(), ConvertMode::Float);
    let fs_out = std::f64::consts::PI * 44_100.0;

    let mt = thdn_at(&m, 44_100.0, fs_out, "sinc_pi_thdn");
    println!(
        "sinc_pi: fund {:.2} dBFS, THD+N {:.2} dB, spur {:.2} dB at {:.2} kHz",
        mt.fundamental_dbfs,
        mt.thd_plus_n_db,
        mt.max_spur_dbfs,
        mt.spur_hz / 1000.0
    );
    assert!(
        mt.fundamental_dbfs > -6.0 - FUNDAMENTAL_TOLERANCE_DB
            && mt.fundamental_dbfs < -6.0 + FUNDAMENTAL_TOLERANCE_DB,
        "pi fundamental dropped: {:.3} dBFS",
        mt.fundamental_dbfs
    );
    assert!(mt.thd_plus_n_db < -120.0, "pi THD+N {}", mt.thd_plus_n_db); // measured -134.4

    // Flatness: passband is ~0.95 * fs_out/2 ~ 65 kHz, tones well inside.
    let errs = flatness_at(
        &m,
        44_100.0,
        fs_out,
        &[997.0, 5000.0, 10000.0, 15000.0, 19000.0],
        "sinc_pi_flat",
    );
    let worst = errs.iter().fold(0.0f64, |a, e| a.max(e.abs()));
    println!("sinc_pi flatness worst: {worst:.3} dB");
    assert!(worst < 0.1, "pi passband ripple {worst:.3} dB > 0.1"); // measured 0.000
}

// ---------------------------------------------------------------------------
// Full quality ladder: all 11 presets get a generic-path THD+N red line
// (quantify is honored there). tw = 0.2 keeps the largest tables
// (Bit24Better = 8193 rows x ~118 taps) cheap to build. Red lines are
// per-tier with >= 6 dB headroom below measured values.
// ---------------------------------------------------------------------------

#[test]
fn all_quality_tiers_thdn() {
    // Lines carry >= 6 dB headroom below the measured values (tw 0.2):
    // Bit8Fast -66.6, Bit8Medium -83.9, Bit8Better -96.6, Bit16Lower -103.0,
    // Bit16Fast -110.3, Bit16Medium -124.5, Bit16Better -141.5,
    // Bit24Lower -153.5, Bit24Fast/Medium/Better hit the -180 f64 floor.
    let tiers: [(Quality, f64); 11] = [
        (Quality::Bit8Fast, -62.0),
        (Quality::Bit8Medium, -77.0),
        (Quality::Bit8Better, -90.0),
        (Quality::Bit16Lower, -96.0),
        (Quality::Bit16Fast, -104.0),
        (Quality::Bit16Medium, -118.0),
        (Quality::Bit16Better, -135.0),
        (Quality::Bit24Lower, -147.0),
        (Quality::Bit24Fast, -170.0),
        (Quality::Bit24Medium, -170.0),
        (Quality::Bit24Better, -170.0),
    ];
    for (tier, line) in tiers {
        let m = SrcManager::builder()
            .sample_rate(44_100, 48_000)
            .quality(tier)
            .trans_width(0.2)
            .generic()
            .build()
            .unwrap();
        let mt = thdn_at(&m, 44_100.0, 48_000.0, "tier");
        println!(
            "{tier:?}: THD+N {:.2} dB (line {line}), spur {:.2} dB",
            mt.thd_plus_n_db, mt.max_spur_dbfs
        );
        assert!(
            mt.thd_plus_n_db < line,
            "{tier:?}: THD+N {:.2} dB exceeds line {line}",
            mt.thd_plus_n_db
        );
    }
}

#[test]
fn generic_quantize_ripple_visible_at_q8() {
    // README documents ~0.15 dB generic passband ripple at quantify = 8 from
    // phase interpolation; Fast (quantify ignored) is flat. Pin both facts:
    // the ripple must be measurable beyond the fast path and stay small.
    let g = SrcManager::builder()
        .sample_rate(44_100, 48_000)
        .quality(Quality::Bit8Fast)
        .trans_width(TW)
        .generic()
        .build()
        .unwrap();
    let f = SrcManager::builder()
        .sample_rate(44_100, 48_000)
        .quality(Quality::Bit8Fast)
        .trans_width(TW)
        .fast()
        .build()
        .unwrap();
    let ge = flatness_case(&g, "q8_generic_flat");
    let fe = flatness_case(&f, "q8_fast_flat");
    let gw = ge.iter().fold(0.0f64, |a, e| a.max(e.abs()));
    let fw = fe.iter().fold(0.0f64, |a, e| a.max(e.abs()));
    println!("q8 generic worst {gw:.3} dB, fast worst {fw:.3} dB");
    assert!(
        gw - fw > 0.02,
        "generic q8 ripple should exceed fast: {gw} vs {fw}"
    );
    assert!(gw < 0.8, "generic q8 ripple too large: {gw}");
}
