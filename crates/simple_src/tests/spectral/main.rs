//! Spectral quality baselines (FFT-based), pinned with hard red lines.
//!
//! These tests measure what the bench suite cannot: how much noise,
//! distortion, aliasing, and passband ripple each kernel actually adds.
//! Measured artifacts (CSV + SVG spectra, raw sweeps) land in
//! `$CARGO_TARGET_TMPDIR/quality/` — open the SVGs in a browser, or feed
//! the raw sweep to `plots.py` for a spectrogram.
//!
//! Run with:
//! ```text
//! cargo test -p simple_src --test quality -- --nocapture
//! ```
//!
//! Thresholds carry ~6 dB headroom below measured values to absorb
//! cross-platform libm differences; tighten deliberately, never silently.

#[path = "../../tooling/report/analysis.rs"]
mod analysis;

mod p1;

use analysis::*;
use simple_src::{Kernel, Quality, SrcManager};

/// Transition width used for every sinc case (fraction of Nyquist).
const TW: f64 = 0.05;
/// Reference tone level: -6 dBFS.
const TONE_AMP: f64 = 0.5;
const FUNDAMENTAL_TOLERANCE_DB: f64 = 0.2;

fn out_samples_needed() -> usize {
    FFT_N + 2 * EDGE_TRIM
}

fn input_len_for(old_fs: f64, new_fs: f64) -> usize {
    (out_samples_needed() as f64 * old_fs / new_fs).ceil() as usize + 64
}

fn analyze(
    manager: &SrcManager,
    input: &[f64],
    new_fs: f64,
    name: &str,
    markers: &[(f64, String)],
) -> (Vec<f64>, f64) {
    let out = manager.convert(input);
    assert!(
        out.len() >= out_samples_needed(),
        "{name}: short output {}",
        out.len()
    );
    let core = &out[EDGE_TRIM..EDGE_TRIM + FFT_N];
    let (db, bin_hz) = spectrum_db(core, new_fs);
    let dir = artifact_dir(".");
    write_csv(&dir.join(format!("{name}.csv")), &db, bin_hz, 4);
    write_svg(
        &dir.join(format!("{name}.svg")),
        &db,
        name,
        new_fs / 2.0,
        markers,
    );
    (db, bin_hz)
}

// ---------------------------------------------------------------------------
// THD+N baselines (44100 -> 48000, 997 Hz tone @ -6 dBFS)
// ---------------------------------------------------------------------------

fn sinc_thdn_case(quality: Quality, fast: bool, name: &str) -> ThdnMetrics {
    let mut b = SrcManager::builder()
        .sample_rate(44100, 48000)
        .quality(quality)
        .trans_width(TW);
    b = if fast { b.fast() } else { b.generic() };
    let m = b.build().unwrap();
    let (f, input) = binned_tone(
        44100.0,
        48000.0,
        997.0,
        TONE_AMP,
        input_len_for(44100.0, 48000.0),
    );
    let (db, bin_hz) = analyze(&m, &input, 48000.0, name, &[(f, "997 Hz".into())]);
    thdn(&db, bin_hz, f)
}

#[test]
fn sinc_thdn_baseline() {
    let cases: [(&str, Quality, bool); 3] = [
        ("sinc_g96_thdn", Quality::Bit16Fast, false),
        ("sinc_f96_thdn", Quality::Bit16Fast, true),
        ("sinc_f144_thdn", Quality::Bit24Fast, true),
    ];
    println!(
        "{:<16} {:>10} {:>8} {:>10} {:>9} {:>9}",
        "case", "fund dBFS", "THD dB", "THD+N dB", "spur dB", "spur kHz"
    );
    for (name, q, fast) in cases {
        let m = sinc_thdn_case(q, fast, name);
        println!(
            "{:<16} {:>10.2} {:>8.2} {:>10.2} {:>9.2} {:>9.2}",
            name,
            m.fundamental_dbfs,
            m.thd_db,
            m.thd_plus_n_db,
            m.max_spur_dbfs,
            m.spur_hz / 1000.0
        );
        // Red lines (see table below for the measured values behind them).
        assert!(
            m.fundamental_dbfs > -6.0 - FUNDAMENTAL_TOLERANCE_DB
                && m.fundamental_dbfs < -6.0 + FUNDAMENTAL_TOLERANCE_DB,
            "{name}: fundamental level drifted: {:.3} dBFS",
            m.fundamental_dbfs
        );
        assert!(
            m.thd_plus_n_db < thdn_red_line(name),
            "{name}: THD+N {:.2} dB exceeds red line {}",
            m.thd_plus_n_db,
            thdn_red_line(name)
        );
        assert!(
            m.max_spur_dbfs < spur_red_line(name),
            "{name}: max spur {:.2} dBFS exceeds red line {}",
            m.max_spur_dbfs,
            spur_red_line(name)
        );
    }
}

#[test]
fn linear_cubic_thdn_baseline() {
    let linear = SrcManager::with_sample_rate(44100, 48000).unwrap();
    let cubic = SrcManager::builder()
        .sample_rate(44100, 48000)
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();
    let (f, input) = binned_tone(
        44100.0,
        48000.0,
        997.0,
        TONE_AMP,
        input_len_for(44100.0, 48000.0),
    );
    println!(
        "{:<16} {:>10} {:>8} {:>10} {:>9} {:>9}",
        "case", "fund dBFS", "THD dB", "THD+N dB", "spur dB", "spur kHz"
    );
    for (name, m_mgr) in [("linear_thdn", &linear), ("cubic_thdn", &cubic)] {
        let (db, bin_hz) = analyze(m_mgr, &input, 48000.0, name, &[(f, "997 Hz".into())]);
        let m = thdn(&db, bin_hz, f);
        println!(
            "{:<16} {:>10.2} {:>8.2} {:>10.2} {:>9.2} {:>9.2}",
            name,
            m.fundamental_dbfs,
            m.thd_db,
            m.thd_plus_n_db,
            m.max_spur_dbfs,
            m.spur_hz / 1000.0
        );
        // Cheap-kernel sanity lines: they are allowed to distort, but a
        // regression beyond these means the interpolation math changed.
        assert!(
            m.thd_plus_n_db < cheap_thdn_red_line(name),
            "{name}: THD+N {:.2} dB exceeds red line {}",
            m.thd_plus_n_db,
            cheap_thdn_red_line(name)
        );
    }
}

// ---------------------------------------------------------------------------
// Alias baselines (48000 -> 44100, 23 kHz tone @ -6 dBFS)
//
// The tone sits above the 44.1 kHz Nyquist, so a converter without
// sufficient stopband rejection folds it into the audible band
// (~21.1 kHz). The residue level directly reads out the realized
// stopband attenuation.
// ---------------------------------------------------------------------------

fn alias_case(manager: &SrcManager, name: &str) -> f64 {
    let (f, input) = binned_tone(
        48000.0,
        48000.0,
        23000.0,
        TONE_AMP,
        input_len_for(48000.0, 44100.0),
    );
    let (db, bin_hz) = analyze(
        manager,
        &input,
        44100.0,
        name,
        &[(f, "23 kHz -> folds".into())],
    );
    // Residue anywhere in 10k..22k above the noise floor.
    let (residue, at) = peak_in(&db, bin_hz, 10_000.0, 22_000.0);
    println!(
        "{name:<20} residue {residue:>7.2} dBFS at {:.2} kHz",
        at / 1000.0
    );
    residue
}

#[test]
fn alias_baseline() {
    let g96 = SrcManager::builder()
        .sample_rate(48000, 44100)
        .quality(Quality::Bit16Fast)
        .trans_width(TW)
        .generic()
        .build()
        .unwrap();
    let f96 = SrcManager::builder()
        .sample_rate(48000, 44100)
        .quality(Quality::Bit16Fast)
        .trans_width(TW)
        .fast()
        .build()
        .unwrap();
    let f144 = SrcManager::builder()
        .sample_rate(48000, 44100)
        .quality(Quality::Bit24Fast)
        .trans_width(TW)
        .fast()
        .build()
        .unwrap();
    let linear = SrcManager::with_sample_rate(48000, 44100).unwrap();
    let cubic = SrcManager::builder()
        .sample_rate(48000, 44100)
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();

    // Stopband red lines; measured residues in comments (23 kHz @ -6 dBFS
    // folds to 21.1 kHz). Headroom >= 6 dB.
    assert!(alias_case(&g96, "sinc_g96_alias") < -109.0); // measured -115.3
    assert!(alias_case(&f96, "sinc_f96_alias") < -109.0); // measured -115.3
    assert!(alias_case(&f144, "sinc_f144_alias") < -155.0); // measured -162.2

    // Linear and cubic do not anti-alias; these lines only catch
    // unexpected changes (e.g. someone quietly adding a filter).
    let lin = alias_case(&linear, "linear_alias");
    assert!(
        lin > -25.0 && lin < 0.0,
        "linear alias residue moved: {lin:.2} dBFS"
    );
    let cub = alias_case(&cubic, "cubic_alias");
    assert!(cub < -6.0, "cubic alias residue moved: {cub:.2} dBFS"); // measured -11.5
}

// ---------------------------------------------------------------------------
// Passband flatness (44100 -> 48000, five tones @ -20 dBFS each)
// ---------------------------------------------------------------------------

const FLAT_TONES: [f64; 5] = [997.0, 5000.0, 10000.0, 15000.0, 19000.0];
const FLAT_AMP_DBFS: f64 = -20.0;

fn flatness_case(manager: &SrcManager, name: &str) -> Vec<f64> {
    let amp = 10.0f64.powf(FLAT_AMP_DBFS / 20.0);
    // Align every tone to an output-rate bin so no scalloping skews levels.
    let bin_hz = 48000.0 / FFT_N as f64;
    let freqs: Vec<f64> = FLAT_TONES
        .iter()
        .map(|&f| (f / bin_hz).round() * bin_hz)
        .collect();
    let input = multi_tone(44100.0, &freqs, amp, input_len_for(44100.0, 48000.0));
    let markers: Vec<(f64, String)> = freqs.iter().map(|&f| (f, format!("{:.0} Hz", f))).collect();
    let (db, bin_hz_out) = analyze(manager, &input, 48000.0, name, &markers);
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
fn passband_flatness_baseline() {
    let g96 = SrcManager::builder()
        .sample_rate(44100, 48000)
        .quality(Quality::Bit16Fast)
        .trans_width(TW)
        .generic()
        .build()
        .unwrap();
    let f96 = SrcManager::builder()
        .sample_rate(44100, 48000)
        .quality(Quality::Bit16Fast)
        .trans_width(TW)
        .fast()
        .build()
        .unwrap();
    let linear = SrcManager::with_sample_rate(44100, 48000).unwrap();
    let cubic = SrcManager::builder()
        .sample_rate(44100, 48000)
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();

    // Measured worst-case gain errors in comments; the cheap kernels roll
    // off near 19 kHz by design, so their limits track that reality.
    for (name, mgr, limit) in [
        ("sinc_f96_flat", &f96, 0.2),   // measured 0.00 dB
        ("sinc_g96_flat", &g96, 0.2),   // measured 0.00 dB
        ("cubic_flat", &cubic, 4.0),    // measured 3.73 dB @ 19 kHz
        ("linear_flat", &linear, 10.0), // measured 5.66 dB @ 19 kHz
    ] {
        let errs = flatness_case(mgr, name);
        let worst = errs.iter().fold(0.0f64, |a, &e| a.max(e.abs()));
        println!(
            "{name:<16} gain error per tone: {:>5.2} dB (worst {worst:.2} dB)",
            errs.iter()
                .map(|e| format!("{e:.2}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        assert!(
            worst < limit,
            "{name}: flatness {worst:.2} dB exceeds {limit} dB"
        );
    }
}

// ---------------------------------------------------------------------------
// Measured-trim filter design baseline (opt-in via `SrcBuilder::trim_filter`)
// ---------------------------------------------------------------------------

/// End-to-end worst-case stopband readout for 48 -> 44.1 with `TW = 0.05`
/// (stopband edge 21.5 kHz): convert single stopband tones on a dense grid
/// (the worst lobe sits just past the stop edge and is narrow, ~1/order) and
/// measure the attenuated (or folded) residue. Tone level is -6 dBFS, so
/// `residue ~= -6 dB + |H(f)|` directly reads the realized stopband.
fn trim_alias_case(atten: f64, trimmed: bool) -> (f64, u32, u128) {
    // Dense grid across the first stopband lobes (21550..22100) plus deeper
    // points; 25 Hz spacing resolves the ~85 Hz-wide first lobe at order ~300.
    let mut tones: Vec<f64> = (21_500i32..=22_100).step_by(25).map(|f| f as f64).collect();
    tones.extend_from_slice(&[23_000.0, 23_700.0]);
    let mut b = SrcManager::builder()
        .sample_rate(48000, 44100)
        .attenuation(atten)
        .trans_width(TW)
        .fast();
    b = if trimmed { b.trim_filter(true) } else { b };
    let t0 = std::time::Instant::now();
    let m = b.build().unwrap();
    let build_ms = t0.elapsed().as_millis();
    let order = m.order().unwrap();
    let input_len = input_len_for(48000.0, 44100.0);
    let mut worst = f64::NEG_INFINITY;
    for &f in &tones {
        let (ft, input) = binned_tone(48000.0, 48000.0, f, TONE_AMP, input_len);
        let out = m.convert(&input);
        let core = &out[EDGE_TRIM..EDGE_TRIM + FFT_N];
        let (db, bin_hz) = spectrum_db(core, 44100.0);
        // Above the 44.1 kHz output Nyquist the tone folds to 44100 - f.
        let out_f = if ft > 22050.0 { 44100.0 - ft } else { ft };
        // The time-varying polyphase branches spread the worst-branch
        // response into sidebands spaced fs_out/160 = 275.6 Hz; include a
        // few of them, otherwise the peak reads only the branch average.
        let sb = 44100.0 / 160.0;
        let (residue, at) = peak_in(&db, bin_hz, out_f - 2.2 * sb, out_f + 2.2 * sb);
        if std::env::var("TRIM_DEBUG").is_ok() {
            println!(
                "  tone {ft:>9.2} -> out {out_f:>9.2}: residue {residue:>8.2} dBFS at {at:>9.2} Hz"
            );
        }
        worst = worst.max(residue);
    }
    (worst, order, build_ms)
}

/// Measured-trim baseline. Pinned release-mode measurements (48 -> 44.1,
/// pass-band tones sweep, worst stopband residue; tone at -6 dBFS so
/// `residue ~= -6 dB + |H|`):
///
/// | atten | design  | order | worst alias dBFS |
/// |-------|---------|-------|------------------|
/// | 96    | formula | 286   | -103.64          |
/// | 96    | trimmed | 268   | -102.16          |
/// | 120   | formula | 358   | -127.85          |
/// | 120   | trimmed | 344   | -128.66          |
/// | 144   | formula | 432   | -150.74          |
/// | 144   | trimmed | 414   | -150.59          |
///
/// The trimmed design saves 14-18 taps at identical end-to-end rejection;
/// trimmed passband flatness measures 0.00 dB worst-case (same as formula).
#[test]
fn trimmed_design_baseline() {
    println!(
        "\n{:>8} {:>8} {:>14} {:>14} {:>10}",
        "atten", "design", "order", "worst alias", "resid-atten"
    );
    for atten in [96.0f64, 120.0, 144.0] {
        let (plain_res, plain_order, _plain_ms) = trim_alias_case(atten, false);
        let (trim_res, trim_order, _trim_ms) = trim_alias_case(atten, true);
        println!(
            "{:>8.0} {:>8} {:>14} {:>14.2} {:>10.2}",
            atten,
            "formula",
            plain_order,
            plain_res,
            plain_res - (-atten)
        );
        println!(
            "{:>8.0} {:>8} {:>14} {:>14.2} {:>10.2}",
            atten,
            "trimmed",
            trim_order,
            trim_res,
            trim_res - (-atten)
        );
        // The trimmed design must meet the requested stopband on every
        // polyphase branch: residue <= tone(-6 dBFS) - atten (+ slack).
        assert!(
            trim_res < -atten - 4.0,
            "trimmed design misses spec at {atten}: {trim_res:.2} dBFS"
        );
        // The trim must not blow up the order to buy compliance.
        assert!(
            trim_order <= plain_order + 16,
            "trimmed order {} >> formula order {}",
            trim_order,
            plain_order
        );
    }

    // Passband must be unaffected: same flatness limits as the baseline.
    let trimmed_flat = SrcManager::builder()
        .sample_rate(44100, 48000)
        .quality(Quality::Bit16Fast)
        .trans_width(TW)
        .fast()
        .trim_filter(true)
        .build()
        .unwrap();
    let errs = flatness_case(&trimmed_flat, "trim_trim_flat");
    let worst = errs.iter().fold(0.0f64, |a, &e| a.max(e.abs()));
    println!("trimmed flatness worst: {worst:.2} dB");
    assert!(worst < 0.2, "trimmed flatness {worst:.2} dB exceeds 0.2 dB");
}

// ---------------------------------------------------------------------------
// Sweep artifact for spectrogram visualization via plots.py
// ---------------------------------------------------------------------------

#[test]
fn sweep_artifact() {
    let input = log_sweep(44100.0, 20.0, 20000.0, 3.0, 0.25);
    let f96 = SrcManager::builder()
        .sample_rate(44100, 48000)
        .quality(Quality::Bit16Fast)
        .trans_width(TW)
        .fast()
        .build()
        .unwrap();
    let out = f96.convert(&input);
    let expected = (input.len() as f64 * 48000.0 / 44100.0) as usize;
    assert!(
        (out.len() as i64 - expected as i64).abs() <= 2,
        "sweep length {} vs {expected}",
        out.len()
    );
    let dir = artifact_dir(".");
    write_raw_f64(&dir.join("sweep_441_480_out.f64"), &out);
    write_raw_f64(&dir.join("sweep_441_480_in.f64"), &input);
    println!(
        "sweep artifacts: {} (plot with plots.py: raw_spectrogram('sweep_441_480_out.f64', 48000))",
        dir.join("sweep_441_480_out.f64").display()
    );
}

// ---------------------------------------------------------------------------
// Red lines (measured on the reference machine; keep ~6 dB headroom)
// ---------------------------------------------------------------------------

/// THD+N red lines; measured values in comments (reference machine,
/// 997 Hz @ -6 dBFS, 44100 -> 48000). Headroom >= 6 dB.
fn thdn_red_line(name: &str) -> f64 {
    match name {
        // Fast LUT coefficients are full-precision f64: noise+distortion is
        // bounded by f64 arithmetic, not table quantization.
        "sinc_f96_thdn" => -125.0,  // measured -134.1
        "sinc_f144_thdn" => -125.0, // measured: below the f64 floor (clamped)
        // Generic quantizes the half table (q=128, linear interpolation).
        "sinc_g96_thdn" => -125.0, // measured -133.2
        _ => unreachable!("unknown case {name}"),
    }
}

fn spur_red_line(name: &str) -> f64 {
    match name {
        "sinc_f96_thdn" => -135.0,  // measured -144.3
        "sinc_f144_thdn" => -135.0, // measured -180.7
        "sinc_g96_thdn" => -135.0,  // measured -144.2
        _ => unreachable!("unknown case {name}"),
    }
}

fn cheap_thdn_red_line(name: &str) -> f64 {
    match name {
        // Cheap-kernel reality checks; a regression beyond these means the
        // interpolation math changed.
        "linear_thdn" => -55.0, // measured -62.5 (interp f64 noise floor)
        "cubic_thdn" => -83.0,  // measured -89.6
        _ => unreachable!("unknown case {name}"),
    }
}
