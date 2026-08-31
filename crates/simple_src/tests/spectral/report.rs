//! Self-contained quality report generator.
//!
//! Recomputes every quality baseline in memory (full-density spectra, so the
//! overlay charts do not suffer CSV downsampling), collects the metrics into
//! summary tables, renders comparison charts, and assembles one
//! self-contained `report.html`:
//!
//! ```text
//! $CARGO_TARGET_TMPDIR/quality/report.html          (with all artifacts)
//! <repo>/output/report/index.html                   (local copy)
//! ```
//!
//! Run with the other spectral tests:
//! ```text
//! cargo test -p simple_src --test spectral -- --nocapture
//! ```
//!
//! The report is meant to be browsed as a single tab: tables up front, every
//! chart folded into a `<details>` block (click to expand), with anchors for
//! the sections. It is intentionally independent of the baseline assertion
//! tests (no ordering dependency), recomputing the `trim_design_baseline`
//! grid with a few critical tones to stay fast.

use std::time::Instant;

use crate::analysis::*;
use simple_src::{Kernel, Quality, SrcManager};

/// Transition width used for every sinc case (fraction of Nyquist).
const TW: f64 = 0.05;
/// Reference tone level: -6 dBFS.
const TONE_AMP: f64 = 0.5;
const FLAT_TONES: [f64; 5] = [997.0, 5000.0, 10000.0, 15000.0, 19000.0];
const FLAT_AMP_DBFS: f64 = -20.0;

fn out_samples_needed() -> usize {
    FFT_N + 2 * EDGE_TRIM
}

fn input_len_for(old_fs: f64, new_fs: f64) -> usize {
    (out_samples_needed() as f64 * old_fs / new_fs).ceil() as usize + 64
}

// ---------------------------------------------------------------------------
// Baseline recomputation (mirrors the assertion tests, but returns the
// full-density spectrum so charts can be rendered from memory)
// ---------------------------------------------------------------------------

struct ThdnCase {
    name: String,
    fundamental_dbfs: f64,
    thd_db: f64,
    thd_plus_n_db: f64,
    max_spur_dbfs: f64,
    spur_hz: f64,
    db: Vec<f64>,
}

fn thdn_case(quality: Quality, fast: bool, name: &str) -> ThdnCase {
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
    let out = m.convert(&input);
    let core = &out[EDGE_TRIM..EDGE_TRIM + FFT_N];
    let (db, bin_hz) = spectrum_db(core, 48000.0);
    let mt = thdn(&db, bin_hz, f);
    ThdnCase {
        name: name.into(),
        fundamental_dbfs: mt.fundamental_dbfs,
        thd_db: mt.thd_db,
        thd_plus_n_db: mt.thd_plus_n_db,
        max_spur_dbfs: mt.max_spur_dbfs,
        spur_hz: mt.spur_hz,
        db,
    }
}

struct AliasCase {
    name: String,
    residue_dbfs: f64,
    at_hz: f64,
    db: Vec<f64>,
}

fn alias_case(manager: &SrcManager, name: &str) -> AliasCase {
    let (_, input) = binned_tone(
        48000.0,
        48000.0,
        23000.0,
        TONE_AMP,
        input_len_for(48000.0, 44100.0),
    );
    let out = manager.convert(&input);
    let core = &out[EDGE_TRIM..EDGE_TRIM + FFT_N];
    let (db, bin_hz) = spectrum_db(core, 44100.0);
    let (residue, at) = peak_in(&db, bin_hz, 10_000.0, 22_000.0);
    AliasCase {
        name: name.into(),
        residue_dbfs: residue,
        at_hz: at,
        db,
    }
}

struct FlatCase {
    name: String,
    errs: Vec<f64>,
    worst: f64,
    db: Vec<f64>,
}

fn flatness_case(manager: &SrcManager, name: &str) -> FlatCase {
    let amp = 10.0f64.powf(FLAT_AMP_DBFS / 20.0);
    let bin_hz = 48000.0 / FFT_N as f64;
    let freqs: Vec<f64> = FLAT_TONES
        .iter()
        .map(|&f| (f / bin_hz).round() * bin_hz)
        .collect();
    let input = multi_tone(44100.0, &freqs, amp, input_len_for(44100.0, 48000.0));
    let out = manager.convert(&input);
    let core = &out[EDGE_TRIM..EDGE_TRIM + FFT_N];
    let (db, bin_hz) = spectrum_db(core, 48000.0);
    let errs: Vec<f64> = freqs
        .iter()
        .map(|&f| {
            let (level, _) = peak_in(&db, bin_hz, f * 0.99, f * 1.01);
            level - FLAT_AMP_DBFS
        })
        .collect();
    let worst = errs.iter().fold(0.0f64, |a, e| a.max(e.abs()));
    FlatCase {
        name: name.into(),
        errs,
        worst,
        db,
    }
}

struct TrimCase {
    atten: f64,
    trimmed: bool,
    order: u32,
    worst_dbfs: f64,
    build_ms: u128,
}

fn trim_case(atten: f64, trimmed: bool) -> TrimCase {
    // A few tones across the critical first stopband lobes (the full grid in
    // trim_design_baseline uses ~25 Hz spacing; this keeps the report fast
    // while still catching the worst lobe, which sits just past 21550 Hz).
    let tones = [21_575.0, 21_650.0, 21_800.0, 22_000.0, 23_000.0, 23_700.0];
    let mut b = SrcManager::builder()
        .sample_rate(48000, 44100)
        .attenuation(atten)
        .trans_width(TW)
        .fast();
    b = if trimmed { b.trim_filter(true) } else { b };
    let t0 = Instant::now();
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
        let out_f = if ft > 22050.0 { 44100.0 - ft } else { ft };
        let sb = 44100.0 / 160.0;
        let (residue, _) = peak_in(&db, bin_hz, out_f - 2.2 * sb, out_f + 2.2 * sb);
        worst = worst.max(residue);
    }
    TrimCase {
        atten,
        trimmed,
        order,
        worst_dbfs: worst,
        build_ms,
    }
}

struct QualityCost {
    name: String,
    atten: f64,
    quantify: u32,
    order: u32,
    generic_lut: usize,
    fast_lut: usize,
}

fn quality_costs() -> Vec<QualityCost> {
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
    presets
        .into_iter()
        .map(|q| {
            let generic = SrcManager::builder()
                .ratio(2.0)
                .generic()
                .quality(q)
                .trans_width(0.2)
                .build()
                .unwrap();
            let fast = SrcManager::builder()
                .ratio(2.0)
                .fast()
                .quality(q)
                .trans_width(0.2)
                .build()
                .unwrap();
            QualityCost {
                name: format!("{q:?}"),
                atten: q.attenuation(),
                quantify: q.quantify(),
                order: generic.order().unwrap(),
                generic_lut: generic.lut_len().unwrap(),
                fast_lut: fast.lut_len().unwrap(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// HTML assembly
// ---------------------------------------------------------------------------

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn f2(x: f64) -> String {
    format!("{x:.2}")
}

fn f3(x: f64) -> String {
    format!("{x:.3}")
}

fn html_table(headers: &[&str], rows: &[Vec<String>], right_align: &[bool]) -> String {
    let mut s = String::from("<table><thead><tr>");
    for h in headers {
        s.push_str(&format!("<th>{}</th>", esc(h)));
    }
    s.push_str("</tr></thead><tbody>");
    for r in rows {
        s.push_str("<tr>");
        for (i, c) in r.iter().enumerate() {
            let cls = if right_align.get(i).copied().unwrap_or(true) {
                " class=\"num\""
            } else {
                ""
            };
            s.push_str(&format!("<td{cls}>{}</td>", esc(c)));
        }
        s.push_str("</tr>");
    }
    s.push_str("</tbody></table>");
    s
}

fn metric_bad(v: f64) -> String {
    // color the THD+N / spur cells: the deeper the better.
    format!(
        "<span style=\"color:{}\">{:.1}</span>",
        if v < -120.0 {
            "#2e7d32"
        } else if v < -90.0 {
            "#388e3c"
        } else if v < -60.0 {
            "#ef6c00"
        } else {
            "#c62828"
        },
        v
    )
}

fn table_row_flex(cells: &[String], right: &[bool]) -> String {
    let mut s = String::from("<tr>");
    for (i, c) in cells.iter().enumerate() {
        let cls = if right.get(i).copied().unwrap_or(true) {
            " class=\"num\""
        } else {
            ""
        };
        s.push_str(&format!("<td{cls}>{}</td>", esc(c)));
    }
    s.push_str("</tr>");
    s
}

/// Rows of the flatness table with per-tone numeric columns.
fn flatness_table(cases: &[FlatCase]) -> String {
    let mut s = String::from(
        "<table><thead><tr><th>case</th><th>997 Hz</th><th>5 kHz</th><th>10 kHz</th><th>15 kHz</th><th>19 kHz</th><th>worst dB</th></tr></thead><tbody>",
    );
    for c in cases {
        let mut cells = vec![c.name.clone()];
        cells.extend(c.errs.iter().map(|&e| f2(e)));
        cells.push(f2(c.worst));
        let right = [false, true, true, true, true, true, true];
        s.push_str(&table_row_flex(&cells, &right));
    }
    s.push_str("</tbody></table>");
    s
}

fn section(title: &str, anchor: &str, body: &str) -> String {
    format!(
        "<h2 id=\"{anchor}\"><a class=\"sec\" href=\"#{anchor}\">§</a> {}</h2>\n{body}",
        esc(title)
    )
}

fn figure(svg_path: &std::path::Path, caption: &str) -> String {
    let svg = std::fs::read_to_string(svg_path)
        .unwrap_or_else(|_| format!("<!-- missing {svg_path:?} -->"));
    format!(
        "<details class=\"fig\"><summary>{}</summary><div class=\"frame\">{svg}</div></details>",
        esc(caption)
    )
}

/// Embed every SVG already written for `names` (single-trace spectra).
fn figure_set(dir: &std::path::Path, names: &[&str], caption: &str) -> String {
    let mut s = format!("<details class=\"fig\"><summary>{}</summary>", esc(caption));
    for n in names {
        if let Ok(svg) = std::fs::read_to_string(dir.join(format!("{n}.svg"))) {
            s.push_str(&svg);
        }
    }
    s.push_str("</details>");
    s
}

// ---------------------------------------------------------------------------
// Report entry point
// ---------------------------------------------------------------------------

fn build_report() {
    let dir = artifact_dir(".");
    println!("report artifacts in {dir:?}");

    // ---- THD+N / spur -----------------------------------------------------
    let mut thdn_cases = vec![
        thdn_case(Quality::Bit16Fast, false, "sinc_g96_thdn"),
        thdn_case(Quality::Bit16Fast, true, "sinc_f96_thdn"),
        thdn_case(Quality::Bit24Fast, true, "sinc_f144_thdn"),
    ];
    let linear = SrcManager::with_sample_rate(44100, 48000).unwrap();
    let cubic = SrcManager::builder()
        .sample_rate(44100, 48000)
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();
    thdn_cases.push(thdn_kernel(&linear, "linear_thdn"));
    thdn_cases.push(thdn_kernel(&cubic, "cubic_thdn"));

    // write the single-trace SVGs the report embeds
    for c in &thdn_cases {
        write_svg(
            &dir.join(format!("{}.svg", c.name)),
            &c.db,
            &c.name,
            48000.0 / 2.0,
            &[(997.0, "997 Hz".into())],
        );
    }

    // ---- alias ------------------------------------------------------------
    let mut alias_cases = Vec::new();
    for (q, fast, name) in [
        (Quality::Bit16Fast, false, "sinc_g96_alias"),
        (Quality::Bit16Fast, true, "sinc_f96_alias"),
        (Quality::Bit24Fast, true, "sinc_f144_alias"),
    ] {
        let mut b = SrcManager::builder()
            .sample_rate(48000, 44100)
            .quality(q)
            .trans_width(TW);
        b = if fast { b.fast() } else { b.generic() };
        alias_cases.push(alias_case(&b.build().unwrap(), name));
    }
    let lin_alias = SrcManager::with_sample_rate(48000, 44100).unwrap();
    let cub_alias = SrcManager::builder()
        .sample_rate(48000, 44100)
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();
    alias_cases.push(alias_case(&lin_alias, "linear_alias"));
    alias_cases.push(alias_case(&cub_alias, "cubic_alias"));
    for c in &alias_cases {
        write_svg(
            &dir.join(format!("{}.svg", c.name)),
            &c.db,
            &c.name,
            44100.0 / 2.0,
            &[(23_000.0, "23 kHz → folds".into())],
        );
    }

    // ---- passband flatness ------------------------------------------------
    let mut flat_cases = Vec::new();
    {
        let b_g = SrcManager::builder()
            .sample_rate(44100, 48000)
            .quality(Quality::Bit16Fast)
            .trans_width(TW)
            .generic();
        flat_cases.push(flatness_case(&b_g.build().unwrap(), "sinc_g96_flat"));
        let b_f = SrcManager::builder()
            .sample_rate(44100, 48000)
            .quality(Quality::Bit16Fast)
            .trans_width(TW)
            .fast();
        flat_cases.push(flatness_case(&b_f.build().unwrap(), "sinc_f96_flat"));
    }
    flat_cases.push(flatness_case(&cubic, "cubic_flat"));
    flat_cases.push(flatness_case(&linear, "linear_flat"));
    for c in &flat_cases {
        write_svg(
            &dir.join(format!("{}.svg", c.name)),
            &c.db,
            &c.name,
            48000.0 / 2.0,
            &[],
        );
    }

    // ---- trim vs formula --------------------------------------------------
    let trim_cases: Vec<TrimCase> = [96.0f64, 120.0, 144.0]
        .into_iter()
        .flat_map(|a| [trim_case(a, false), trim_case(a, true)])
        .collect();

    // ---- quality tier cost ------------------------------------------------
    let costs = quality_costs();

    // ---- comparison charts ------------------------------------------------
    let g96 = &thdn_cases[0];
    let f96 = &thdn_cases[1];
    write_overlay_svg(
        &dir.join("thdn_generic_vs_fast.svg"),
        &[("sinc_g96", &g96.db), ("sinc_f96", &f96.db)],
        "generic vs fast · THD+N spectrum (44100 → 48000, 997 Hz @ −6 dBFS)",
        24_000.0,
    );
    let ag = &alias_cases[0];
    let af = &alias_cases[1];
    write_overlay_svg(
        &dir.join("alias_generic_vs_fast.svg"),
        &[("sinc_g96", &ag.db), ("sinc_f96", &af.db)],
        "generic vs fast · alias spectrum (48000 → 44100, 23 kHz folds to 21.1 kHz)",
        22_050.0,
    );

    // THD+N bars
    let mut bars: Vec<(&str, f64)> = thdn_cases
        .iter()
        .map(|c| (c.name.as_str(), c.thd_plus_n_db))
        .collect();
    write_hbar_svg(
        &dir.join("thdn_compare.svg"),
        &bars,
        "THD+N by kernel (997 Hz @ −6 dBFS)",
        -180.0,
        0.0,
    );
    bars = thdn_cases
        .iter()
        .map(|c| (c.name.as_str(), c.max_spur_dbfs))
        .collect();
    write_hbar_svg(
        &dir.join("spur_compare.svg"),
        &bars,
        "max spur by kernel",
        -200.0,
        0.0,
    );
    let alias_bars: Vec<(&str, f64)> = alias_cases
        .iter()
        .map(|c| (c.name.as_str(), c.residue_dbfs))
        .collect();
    write_hbar_svg(
        &dir.join("alias_compare.svg"),
        &alias_bars,
        "stopband alias residue (23 kHz @ −6 dBFS → folds to 21.1 kHz)",
        -170.0,
        0.0,
    );
    let flat_bars: Vec<(&str, f64)> = flat_cases
        .iter()
        .map(|c| (c.name.as_str(), -c.worst))
        .collect();
    write_hbar_svg(
        &dir.join("flatness_compare.svg"),
        &flat_bars,
        "passband flatness (worst tone error, −dB for visual consistency)",
        -10.0,
        0.0,
    );
    let trim_bars: Vec<(String, f64)> = trim_cases
        .iter()
        .map(|c| {
            let label = format!(
                "a{:.0} {}",
                c.atten,
                if c.trimmed { "trim" } else { "formula" }
            );
            (label, -(c.worst_dbfs))
        })
        .collect();
    let trim_refs: Vec<(&str, f64)> = trim_bars.iter().map(|(l, v)| (l.as_str(), *v)).collect();
    write_hbar_svg(
        &dir.join("trim_compare.svg"),
        &trim_refs,
        "trim vs formula · stopband rejection (−dBFS; higher is deeper)",
        -170.0,
        -60.0,
    );

    // quality cost curves (log10 LUT)
    let generic_pts: Vec<(f64, f64)> = costs
        .iter()
        .map(|c| (c.atten, (c.generic_lut as f64).log10()))
        .collect();
    let fast_pts: Vec<(f64, f64)> = costs
        .iter()
        .map(|c| (c.atten, (c.fast_lut as f64).log10()))
        .collect();
    write_scatter_svg(
        &dir.join("quality_cost.svg"),
        &[("generic", generic_pts), ("fast", fast_pts)],
        "LUT size by quality tier (ratio 2.0, tw 0.2; log10 LUT)",
        "attenuation (dB)",
        "log10 LUT entries",
    );

    // ---- HTML -------------------------------------------------------------
    let mut html = String::new();
    html.push_str(
        r##"<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>simple_src · quality report</title><style>
body{font-family:system-ui,-apple-system,sans-serif;margin:0 auto;max-width:1200px;padding:1rem 2rem;background:#fafafa;color:#222}
h1{font-size:1.6rem}h2{margin-top:2.4rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}
a.sec{color:#888;text-decoration:none}
table{border-collapse:collapse;margin:.6rem 0 1.2rem;background:#fff;box-shadow:0 1px 2px #0001}
th,td{border:1px solid #ddd;padding:.28rem .7rem;text-align:left;font-size:.9rem}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
th{background:#f0f0f0;font-weight:600}
details.fig{margin:.5rem 0;border:1px solid #ddd;background:#fff;border-radius:4px}
details.fig summary{cursor:pointer;padding:.55rem .9rem;font-weight:600;font-size:.92rem}
details.fig .frame{overflow-x:auto;padding:0}
svg{display:block;width:100%;height:auto;max-width:1200px}
code{background:#eee;padding:.1rem .35rem;border-radius:3px;font-size:.85rem}
p.note{color:#555;font-size:.9rem}
</style></head><body>
<h1>simple_src · quality report</h1>
<p class="note">Generated by <code>cargo test -p simple_src --test spectral</code>.
Every chart is folded; click a summary to expand. The report is
self-contained (all SVGs inline) and can be opened in any browser.</p>
<nav><p>
<a href="#thdn">THD+N &amp; spur</a> ·
<a href="#alias">alias</a> ·
<a href="#flat">flatness</a> ·
<a href="#compare">generic vs fast</a> ·
<a href="#trim">trim vs formula</a> ·
<a href="#cost">quality cost</a> ·
<a href="#repro">reproduce</a>
</p></nav>
"##,
    );

    // 1. THD+N
    {
        let headers = [
            "case",
            "fund dBFS",
            "THD dB",
            "THD+N dB",
            "spur dB",
            "spur kHz",
        ];
        let rows: Vec<Vec<String>> = thdn_cases
            .iter()
            .map(|c| {
                vec![
                    c.name.clone(),
                    f2(c.fundamental_dbfs),
                    f2(c.thd_db),
                    f2(c.thd_plus_n_db),
                    metric_bad(c.max_spur_dbfs),
                    f2(c.spur_hz / 1000.0),
                ]
            })
            .collect();
        let right = [false, true, true, true, true, true];
        html.push_str(&section(
            "1 · THD+N &amp; max spur — 44100 → 48000, 997 Hz @ −6 dBFS",
            "thdn",
            &html_table(&headers, &rows, &right),
        ));
        html.push_str(&figure(&dir.join("thdn_compare.svg"), "THD+N bar chart"));
        html.push_str(&figure(&dir.join("spur_compare.svg"), "Max spur bar chart"));
        let names: Vec<&str> = thdn_cases.iter().map(|c| c.name.as_str()).collect();
        html.push_str(&figure_set(&dir, &names, "THD+N spectra"));
    }

    // 2. alias
    {
        let headers = ["case", "residue dBFS", "at kHz"];
        let rows: Vec<Vec<String>> = alias_cases
            .iter()
            .map(|c| {
                vec![
                    c.name.clone(),
                    metric_bad(c.residue_dbfs),
                    f3(c.at_hz / 1000.0),
                ]
            })
            .collect();
        html.push_str(&section(
            "2 · stopband alias — 48000 → 44100, 23 kHz @ −6 dBFS folds to 21.1 kHz",
            "alias",
            &html_table(&headers, &rows, &[false, true, true]),
        ));
        html.push_str(&figure(
            &dir.join("alias_compare.svg"),
            "Alias residue bar chart",
        ));
        let names: Vec<&str> = alias_cases.iter().map(|c| c.name.as_str()).collect();
        html.push_str(&figure_set(&dir, &names, "Alias spectra"));
    }

    // 3. flatness
    {
        html.push_str(&section(
            "3 · passband flatness — five tones @ −20 dBFS (gain error per tone)",
            "flat",
            &flatness_table(&flat_cases),
        ));
        html.push_str(&figure(
            &dir.join("flatness_compare.svg"),
            "Worst flatness error bar chart (−dB)",
        ));
        let names: Vec<&str> = flat_cases.iter().map(|c| c.name.as_str()).collect();
        html.push_str(&figure_set(&dir, &names, "Flatness spectra"));
    }

    // 4. generic vs fast
    {
        html.push_str(&section(
            "4 · generic vs fast — same filter order, different table layouts",
            "compare",
            "<p class=\"note\">Generic interpolates between quantify rows (passband ripple ~0.15 dB at q=8); \
             Fast stores one tap set per rational phase. Stopband rejection is identical — see the overlay.</p>",
        ));
        html.push_str(&figure(
            &dir.join("thdn_generic_vs_fast.svg"),
            "THD+N spectrum overlay",
        ));
        html.push_str(&figure(
            &dir.join("alias_generic_vs_fast.svg"),
            "Alias spectrum overlay",
        ));
    }

    // 5. trim vs formula
    {
        let headers = ["atten", "design", "order", "worst alias dBFS", "init ms"];
        let rows: Vec<Vec<String>> = trim_cases
            .iter()
            .map(|c| {
                vec![
                    format!("{:.0}", c.atten),
                    (if c.trimmed { "trimmed" } else { "formula" }).to_string(),
                    format!("{}", c.order),
                    f2(c.worst_dbfs),
                    format!("{}", c.build_ms),
                ]
            })
            .collect();
        html.push_str(&section(
            "5 · measured-trim vs formula design — 48000 → 44100, fast path (report grid: 6 tones)",
            "trim",
            &html_table(&headers, &rows, &[false, false, true, true, true]),
        ));
        html.push_str(&figure(
            &dir.join("trim_compare.svg"),
            "Stopband rejection: formula vs trimmed (deeper is better)",
        ));
    }

    // 6. quality cost
    {
        let headers = [
            "preset",
            "atten",
            "quantify",
            "order",
            "generic LUT",
            "fast LUT",
        ];
        let rows: Vec<Vec<String>> = costs
            .iter()
            .map(|c| {
                vec![
                    c.name.clone(),
                    format!("{:.0}", c.atten),
                    format!("{}", c.quantify),
                    format!("{}", c.order),
                    format!("{}", c.generic_lut),
                    format!("{}", c.fast_lut),
                ]
            })
            .collect();
        html.push_str(&section(
            "6 · quality tier cost — ratio 2.0, tw 0.2 (LUT entries; fast = numerator per phase)",
            "cost",
            &html_table(&headers, &rows, &[false, true, true, true, true, true]),
        ));
        html.push_str(&figure(
            &dir.join("quality_cost.svg"),
            "LUT size vs attenuation (log scale)",
        ));
    }

    // 7. reproduce
    {
        html.push_str(&section(
            "7 · reproduce",
            "repro",
            "<ul>
<li>Generate the report: <code>cargo test -p simple_src --test spectral -- --nocapture</code></li>
<li>Artifacts live in <code>$CARGO_TARGET_TMPDIR/quality/</code>; a local copy is written to <code>output/report/index.html</code></li>
<li>Feed raw sweeps to plots.py: <code>plots.raw_spectrogram('sweep_441_480_out.f64', 48000)</code></li>
<li>Open the quality CSV in any plotting tool: <code>freq_hz,dbfs</code> per case.</li>
</ul>",
        ));
    }

    html.push_str("</body></html>");

    let report_path = dir.join("report.html");
    std::fs::write(&report_path, &html).unwrap();
    println!("report: {}", report_path.display());

    // local copy next to the repo's output/ folder for convenient browsing
    let local = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("output")
        .join("report");
    if std::fs::create_dir_all(&local).is_ok() {
        let dest = local.join("index.html");
        if std::fs::write(&dest, &html).is_ok() {
            println!("local copy: {}", dest.display());
        }
    }
}

/// Linear/cubic THD+N cases (ratio-only kernels, zero latency).
fn thdn_kernel(manager: &SrcManager, name: &str) -> ThdnCase {
    let (f, input) = binned_tone(
        44100.0,
        48000.0,
        997.0,
        TONE_AMP,
        input_len_for(44100.0, 48000.0),
    );
    let out = manager.convert(&input);
    let core = &out[EDGE_TRIM..EDGE_TRIM + FFT_N];
    let (db, bin_hz) = spectrum_db(core, 48000.0);
    let mt = thdn(&db, bin_hz, f);
    ThdnCase {
        name: name.into(),
        fundamental_dbfs: mt.fundamental_dbfs,
        thd_db: mt.thd_db,
        thd_plus_n_db: mt.thd_plus_n_db,
        max_spur_dbfs: mt.max_spur_dbfs,
        spur_hz: mt.spur_hz,
        db,
    }
}

#[test]
fn generate_report() {
    build_report();
}
