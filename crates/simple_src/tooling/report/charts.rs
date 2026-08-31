// ---------------------------------------------------------------------------
// Report chart primitives (self-contained SVG, embedded into report.html)
// ---------------------------------------------------------------------------
//! Self-contained SVG chart primitives for the quality report. Standalone on
//! purpose: it only uses `std`, so the report example and any other consumer
//! can share it without pulling different analysis dependencies.

use std::fmt::Write as _;
use std::path::Path;

const PALETTE: [&str; 6] = [
    "#0a58ca", "#c62828", "#2e7d32", "#6a1b9a", "#ef6c00", "#00838f",
];

const CH_W: f64 = 960.0;
const CH_H: f64 = 380.0;
const CH_ML: f64 = 62.0;
const CH_MR: f64 = 16.0;
const CH_MT: f64 = 34.0;
const CH_MB: f64 = 40.0;

fn chart_x_of(nyquist: f64) -> impl Fn(f64) -> f64 {
    move |f: f64| CH_ML + (f / nyquist) * (CH_W - CH_ML - CH_MR)
}

fn chart_y_of(y_max: f64, y_min: f64) -> impl Fn(f64) -> f64 {
    move |v: f64| CH_MT + (y_max - v) / (y_max - y_min) * (CH_H - CH_MT - CH_MB)
}

fn chart_open(title: &str, s: &mut String) {
    let _ = writeln!(
        s,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{CH_W}" height="{CH_H}" viewBox="0 0 {CH_W} {CH_H}">"#
    );
    let _ = writeln!(s, r#"<rect width="{CH_W}" height="{CH_H}" fill="white"/>"#);
    let _ = writeln!(
        s,
        r#"<text x="{CH_ML}" y="20" font-size="14" font-family="sans-serif" font-weight="bold">{}</text>"#,
        title.replace('&', "&amp;")
    );
}

/// Horizontal grid every 20 dB plus vertical grid every 2 kHz.
fn chart_axes(nyquist: f64, y_max: f64, y_min: f64, s: &mut String) {
    let x_of = chart_x_of(nyquist);
    let y_of = chart_y_of(y_max, y_min);
    let mut v = y_max;
    while v >= y_min {
        let y = y_of(v);
        let _ = writeln!(
            s,
            r##"<line x1="{CH_ML}" y1="{y}" x2="{}" y2="{y}" stroke="#ddd" stroke-width="1"/>"##,
            CH_W - CH_MR
        );
        let _ = writeln!(
            s,
            r#"<text x="{}" y="{y}" font-size="10" font-family="monospace" text-anchor="end" dominant-baseline="middle">{v}</text>"#,
            CH_ML - 6.0
        );
        v -= 20.0;
    }
    let mut f = 0.0;
    while f <= nyquist {
        let x = x_of(f);
        let _ = writeln!(
            s,
            r##"<line x1="{x}" y1="{CH_MT}" x2="{x}" y2="{}" stroke="#eee" stroke-width="1"/>"##,
            CH_H - CH_MB
        );
        let _ = writeln!(
            s,
            r#"<text x="{x}" y="{}" font-size="10" font-family="sans-serif" text-anchor="middle">{:.0}k</text>"#,
            CH_H - CH_MB + 14.0,
            f / 1000.0
        );
        f += 2000.0;
    }
}

fn chart_close(s: &mut String) {
    let _ = writeln!(s, "</svg>");
}

/// Decimate a spectrum to one pixel column per point, taking the max-abs
/// peak per column (keeps narrow spurs visible).
fn chart_polyline(db: &[f64], color: &str, width: f64, nyquist: f64, s: &mut String) {
    let cols = (CH_W - CH_ML - CH_MR) as usize;
    let _x_of = chart_x_of(nyquist);
    let y_of = chart_y_of(10.0, -200.0);
    let mut pts = String::new();
    for col in 0..cols {
        let b0 = col * db.len() / cols;
        let b1 = ((col + 1) * db.len() / cols).max(b0 + 1);
        let peak = db[b0..b1].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let _ = write!(
            pts,
            "{:.1},{:.1} ",
            CH_ML + col as f64,
            y_of(peak.clamp(-200.0, 10.0))
        );
    }
    let _ = writeln!(
        s,
        r##"<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"/>"##
    );
}

/// Multi-trace spectrum overlay with a legend, drawn at full density.
pub fn write_overlay_svg(path: &Path, series: &[(&str, &[f64])], title: &str, nyquist: f64) {
    let mut s = String::new();
    chart_open(title, &mut s);
    chart_axes(nyquist, 10.0, -200.0, &mut s);
    for (i, (label, db)) in series.iter().enumerate() {
        chart_polyline(db, PALETTE[i % PALETTE.len()], 1.2, nyquist, &mut s);
        // legend swatch, two per row at the top right
        let col = i / 8;
        let row = i % 8;
        let lx = CH_W - CH_MR - 190.0 - col as f64 * 210.0;
        let ly = CH_MT + 6.0 + row as f64 * 14.0;
        let _ = writeln!(
            s,
            r#"<rect x="{lx}" y="{}" width="12" height="4" fill="{}"/>"#,
            ly - 2.0,
            PALETTE[i % PALETTE.len()]
        );
        let _ = writeln!(
            s,
            r#"<text x="{}" y="{ly}" font-size="10" font-family="monospace">{}</text>"#,
            lx + 16.0,
            label.replace('&', "&amp;")
        );
    }
    chart_close(&mut s);
    std::fs::write(path, s).unwrap();
}

/// Horizontal bar chart for comparing scalar metrics (THD+N, spur, …).
/// `ymin`/`ymax` give the value axis range; bars run from `ymin` to their
/// value so negative dB numbers grow visually to the right.
pub fn write_hbar_svg(path: &Path, bars: &[(&str, f64)], title: &str, ymin: f64, ymax: f64) {
    const ROW_H: f64 = 30.0;
    const BAR_PAD: f64 = 8.0;
    let h = (bars.len() as f64) * ROW_H + CH_MT + CH_MB;
    let mut s = String::new();
    let _ = writeln!(
        s,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{CH_W}" height="{h}" viewBox="0 0 {CH_W} {h}">"#
    );
    let _ = writeln!(s, r#"<rect width="{CH_W}" height="{h}" fill="white"/>"#);
    let _ = writeln!(
        s,
        r#"<text x="{CH_ML}" y="20" font-size="14" font-family="sans-serif" font-weight="bold">{}</text>"#,
        title.replace('&', "&amp;")
    );
    let plot_w = CH_W - CH_ML - CH_MR;
    let plot_h = h - CH_MT - CH_MB;
    let x_of = |v: f64| CH_ML + (v - ymin) / (ymax - ymin) * plot_w;
    let y_of = |row: usize| CH_MT + row as f64 * ROW_H;
    // zero line when it lies inside the range
    if ymin < 0.0 && 0.0 < ymax {
        let x0 = x_of(0.0);
        let _ = writeln!(
            s,
            r##"<line x1="{x0}" y1="{CH_MT}" x2="{x0}" y2="{}" stroke="#bbb" stroke-width="1"/>"##,
            CH_MT + plot_h
        );
    }
    for (i, (label, value)) in bars.iter().enumerate() {
        let y = y_of(i) + BAR_PAD;
        let x1 = x_of(ymin);
        let x2 = x_of(*value);
        let _ = writeln!(
            s,
            r#"<text x="{}" y="{}" font-size="11" font-family="sans-serif" text-anchor="end" dominant-baseline="middle">{}</text>"#,
            CH_ML - 6.0,
            y + ROW_H / 2.0 - BAR_PAD,
            label.replace('&', "&amp;")
        );
        let _ = writeln!(
            s,
            r#"<rect x="{x1}" y="{y}" width="{}" height="{}" fill="{}"/>"#,
            (x2 - x1).max(1.0),
            ROW_H - 2.0 * BAR_PAD,
            PALETTE[i % PALETTE.len()]
        );
        let _ = writeln!(
            s,
            r#"<text x="{}" y="{}" font-size="10" font-family="monospace" dominant-baseline="middle">{:.2}</text>"#,
            x2 + 4.0,
            y + ROW_H / 2.0 - BAR_PAD,
            value
        );
    }
    chart_close(&mut s);
    std::fs::write(path, s).unwrap();
}

/// Line/scatter chart (e.g. quality tier cost curves).
pub fn write_scatter_svg(
    path: &Path,
    series: &[(&str, Vec<(f64, f64)>)],
    title: &str,
    xlabel: &str,
    ylabel: &str,
) {
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for (_, pts) in series {
        for &(x, y) in pts {
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
    }
    if !(xmin.is_finite() && xmax > xmin && ymin.is_finite() && ymax > ymin) {
        return; // nothing to draw
    }
    let xpad = (xmax - xmin) * 0.05; // pad both sides of the x axis
    let ypad = (ymax - ymin) * 0.08;
    let x_of =
        |x: f64| CH_ML + (x - (xmin - xpad)) / (xmax - xmin + 2.0 * xpad) * (CH_W - CH_ML - CH_MR);
    let y_of =
        |y: f64| CH_MT + (ymax + ypad - y) / (ymax - ymin + 2.0 * ypad) * (CH_H - CH_MT - CH_MB);
    let mut s = String::new();
    chart_open(title, &mut s);
    // light grid
    let _ = writeln!(
        s,
        r##"<line x1="{CH_ML}" y1="{CH_MT}" x2="{CH_ML}" y2="{}" stroke="#ddd" stroke-width="1"/>"##,
        CH_H - CH_MB
    );
    let _ = writeln!(
        s,
        r##"<line x1="{CH_ML}" y1="{}" x2="{}" y2="{}" stroke="#ddd" stroke-width="1"/>"##,
        CH_H - CH_MB,
        CH_W - CH_MR,
        CH_H - CH_MB
    );
    for (i, (label, pts)) in series.iter().enumerate() {
        let color = PALETTE[i % PALETTE.len()];
        // connecting polyline + point markers
        let line: String = pts
            .iter()
            .map(|&(x, y)| format!("{:.1},{:.1}", x_of(x), y_of(y)))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(
            s,
            r##"<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.4"/>"##
        );
        for &(x, y) in pts {
            let _ = writeln!(
                s,
                r#"<circle cx="{}" cy="{}" r="3.2" fill="{color}"/>"#,
                x_of(x),
                y_of(y)
            );
        }
        let lx = CH_ML + 8.0;
        let ly = CH_MT + 12.0 + i as f64 * 15.0;
        let _ = writeln!(
            s,
            r#"<rect x="{lx}" y="{}" width="12" height="4" fill="{color}"/>"#,
            ly - 2.0
        );
        let _ = writeln!(
            s,
            r#"<text x="{}" y="{ly}" font-size="10" font-family="sans-serif">{}</text>"#,
            lx + 16.0,
            label.replace('&', "&amp;")
        );
    }
    let _ = writeln!(
        s,
        r##"<text x="{CH_ML}" y="{}" font-size="11" font-family="sans-serif" fill="#555555">{}</text>"##,
        CH_H - CH_MB + 18.0,
        xlabel.replace('&', "&amp;")
    );
    let _ = writeln!(
        s,
        r##"<text x="14" y="{}" font-size="11" font-family="sans-serif" fill="#555555" transform="rotate(-90 14 {})">{}</text>"##,
        CH_MT + 40.0,
        CH_MT + 40.0,
        ylabel.replace('&', "&amp;")
    );
    chart_close(&mut s);
    std::fs::write(path, s).unwrap();
}
