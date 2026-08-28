//! Signal-analysis helpers for the spectral baseline tests.
//!
//! Self-contained on purpose: a radix-2 FFT, Hann windowing, and the
//! metric extractors (THD+N, max spur, alias residue, tone levels) needed
//! to pin quality red lines without external dependencies. Artifacts
//! (CSV + SVG spectra, raw sweeps) are written to `CARGO_TARGET_TMPDIR`
//! so failures can be inspected visually.

use std::f64::consts::PI;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

pub const FFT_N: usize = 65536;
/// Samples trimmed from each end of a converted buffer before analysis
/// (filter edge effects; one-shot `convert` already drops sinc latency).
pub const EDGE_TRIM: usize = 2048;

// ---------------------------------------------------------------------------
// FFT + spectrum
// ---------------------------------------------------------------------------

/// In-place iterative radix-2 Cooley–Tukey FFT with a precomputed twiddle
/// table. `re.len()` must be a power of two.
pub fn fft_inplace(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    assert!(n.is_power_of_two());
    let bits = n.trailing_zeros();
    let twiddle: Vec<(f64, f64)> = (0..n / 2)
        .map(|k| {
            let ang = -2.0 * PI * k as f64 / n as f64;
            (ang.cos(), ang.sin())
        })
        .collect();
    for i in 0..n {
        let j = (i.reverse_bits()) >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let (c, s) = twiddle[k * step];
                let i0 = start + k;
                let i1 = i0 + half;
                let tre = re[i1] * c - im[i1] * s;
                let tim = re[i1] * s + im[i1] * c;
                re[i1] = re[i0] - tre;
                im[i1] = im[i0] - tim;
                re[i0] += tre;
                im[i0] += tim;
            }
        }
        len *= 2;
    }
}

/// Hann-windowed amplitude spectrum in dBFS (0 dBFS = full-scale sine).
/// Returns `(db_per_bin, bin_hz)` for bins `0..n/2`.
pub fn spectrum_db(x: &[f64], fs: f64) -> (Vec<f64>, f64) {
    let n = FFT_N;
    assert!(x.len() >= n, "need at least {n} samples, got {}", x.len());
    let mut re: Vec<f64> = (0..n)
        .map(|i| {
            let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos();
            x[i] * w
        })
        .collect();
    let mut im = vec![0.0; n];
    fft_inplace(&mut re, &mut im);
    // A full-scale sine through a Hann window peaks at |X| = A*N/4
    // (window coherent gain 0.5); scale by 4/n so it reads 0 dBFS.
    let scale = 4.0 / n as f64;
    let db: Vec<f64> = re[..n / 2]
        .iter()
        .zip(&im[..n / 2])
        .map(|(r, i)| 20.0 * (scale * (r * r + i * i).sqrt()).max(1e-12).log10())
        .collect();
    (db, fs / n as f64)
}

// ---------------------------------------------------------------------------
// Metric extraction
// ---------------------------------------------------------------------------

fn bin_of_floor(freq: f64, bin_hz: f64) -> usize {
    (freq / bin_hz) as usize
}

/// Peak level (dBFS) inside a frequency range, and its center frequency.
pub fn peak_in(db: &[f64], bin_hz: f64, lo: f64, hi: f64) -> (f64, f64) {
    let end = bin_of_floor(hi, bin_hz).min(db.len() - 1);
    let (best, best_b) = (bin_of_floor(lo, bin_hz)..=end).map(|b| (db[b], b)).fold(
        (f64::NEG_INFINITY, 0),
        |acc, x| if x.0 > acc.0 { x } else { acc },
    );
    (best, best_b as f64 * bin_hz)
}

/// Total power (linear) inside a frequency range.
fn band_power(db: &[f64], bin_hz: f64, lo: f64, hi: f64) -> f64 {
    let end = bin_of_floor(hi, bin_hz).min(db.len() - 1);
    (bin_of_floor(lo, bin_hz)..=end)
        .map(|b| 10.0f64.powf(db[b] / 10.0))
        .sum()
}

/// Bands (as `(lo_hz, hi_hz)` pairs) excluded from noise/spur searches.
pub type Exclusions = Vec<(f64, f64)>;

/// Tone + its harmonics 2..=8 (each ±0.5%), plus DC up to 50 Hz.
pub fn tone_exclusions(f0: f64, nyquist: f64) -> Exclusions {
    let mut ex: Exclusions = vec![(0.0, 50.0)];
    for k in 1..=8 {
        let f = f0 * k as f64;
        if f >= nyquist {
            break;
        }
        ex.push((f * 0.995, (f * 1.005).min(nyquist)));
    }
    ex
}

pub struct ThdnMetrics {
    /// Fundamental level in dBFS (amplitude fidelity check).
    pub fundamental_dbfs: f64,
    /// Total harmonic distortion relative to the fundamental, in dB.
    pub thd_db: f64,
    /// Noise + distortion (everything except the fundamental) relative to
    /// the fundamental, in dB. The headline quality figure.
    pub thd_plus_n_db: f64,
    /// Strongest spectral line outside all excluded bands, in dBFS.
    pub max_spur_dbfs: f64,
    /// Frequency of the max spur.
    pub spur_hz: f64,
}

/// THD+N and spur analysis around a tone at `f0`.
pub fn thdn(db: &[f64], bin_hz: f64, f0: f64) -> ThdnMetrics {
    let nyquist = (db.len() - 1) as f64 * bin_hz;
    let ex = tone_exclusions(f0, nyquist);
    // NOTE on conventions: the Hann window samples its main lobe at ±1 bins
    // only 6.02 dB down (discrete-bin structure), so integrated tone "power"
    // is 1.5x A^2 for a bin-aligned tone. That factor cancels in the
    // power *ratios* below (same convention for fundamental and harmonics);
    // only the absolute `fundamental_dbfs` reading uses the peak amplitude.
    let fundamental = band_power(db, bin_hz, f0 * 0.995, f0 * 1.005);
    let (fundamental_peak, _) = peak_in(db, bin_hz, f0 * 0.995, f0 * 1.005);
    let mut harmonic = 0.0;
    for k in 2..=8 {
        let f = f0 * k as f64;
        if f >= nyquist {
            break;
        }
        harmonic += band_power(db, bin_hz, f * 0.995, f * 1.005);
    }
    let total = band_power(db, bin_hz, 0.0, nyquist);
    // Near the f64 noise floor the subtraction can cancel to zero or below;
    // clamp so the reported ratio bottoms out at -180 dB instead of NaN.
    let noise = (total - fundamental - harmonic).max(fundamental * 1e-18);
    let (max_spur_dbfs, spur_hz) = max_spur(db, bin_hz, &ex);
    ThdnMetrics {
        fundamental_dbfs: fundamental_peak,
        thd_db: 10.0 * (harmonic / fundamental).log10(),
        thd_plus_n_db: 10.0 * (noise / fundamental).log10(),
        max_spur_dbfs,
        spur_hz,
    }
}

/// Strongest spectral line outside `exclusions`, in dBFS.
pub fn max_spur(db: &[f64], bin_hz: f64, exclusions: &Exclusions) -> (f64, f64) {
    let nyquist = (db.len() - 1) as f64 * bin_hz;
    let in_excl = |f: f64| exclusions.iter().any(|&(lo, hi)| f >= lo && f <= hi);
    (bin_of_floor(50.0, bin_hz)..=bin_of_floor(nyquist - 50.0, bin_hz))
        .map(|b| (db[b], b as f64 * bin_hz))
        .filter(|&(_v, f)| !in_excl(f))
        .fold(
            (f64::NEG_INFINITY, 0.0),
            |acc, x| {
                if x.0 > acc.0 { x } else { acc }
            },
        )
}

// ---------------------------------------------------------------------------
// Signal generation
// ---------------------------------------------------------------------------

/// A sine whose frequency is an exact bin of the *analysis* rate
/// (`fs_analysis / FFT_N`) so Hann leakage stays within ±1 bin, sampled at
/// `fs_gen` (the converter input rate; the physical frequency is preserved).
/// Returns `(freq_hz, samples)`.
pub fn binned_tone(
    fs_gen: f64,
    fs_analysis: f64,
    target_hz: f64,
    amp: f64,
    len: usize,
) -> (f64, Vec<f64>) {
    let bin_hz = fs_analysis / FFT_N as f64;
    let bin = (target_hz / bin_hz).round().max(1.0) as usize;
    let f = bin as f64 * bin_hz;
    let x: Vec<f64> = (0..len)
        .map(|i| amp * (2.0 * PI * f * i as f64 / fs_gen).sin())
        .collect();
    (f, x)
}

/// Equal-amplitude multi-tone for passband flatness.
pub fn multi_tone(fs: f64, freqs: &[f64], amp: f64, len: usize) -> Vec<f64> {
    let mut x = vec![0.0; len];
    for &f in freqs {
        // exact bin alignment per output rate is impossible for all tones at
        // once; align each at its own rate and accept <=0.2 dB scalloping.
        for (i, s) in x.iter_mut().enumerate() {
            *s += amp * (2.0 * PI * f * i as f64 / fs).sin();
        }
    }
    x
}

/// Logarithmic sweep `f0 -> f1` over `secs` seconds.
pub fn log_sweep(fs: f64, f0: f64, f1: f64, secs: f64, amp: f64) -> Vec<f64> {
    let n = (fs * secs) as usize;
    let k = (f1 / f0).ln();
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let phase = 2.0 * PI * f0 * secs / k * ((f1 / f0).powf(t / secs) - 1.0);
            amp * phase.sin()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Artifacts: CSV + SVG spectrum, raw sweeps
// ---------------------------------------------------------------------------

pub fn artifact_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("quality");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

pub fn write_csv(path: &Path, db: &[f64], bin_hz: f64, stride: usize) {
    let mut s = String::from("freq_hz,dbfs\n");
    for (b, &v) in db.iter().enumerate().step_by(stride) {
        let _ = writeln!(s, "{:.2},{:.3}", b as f64 * bin_hz, v);
    }
    std::fs::write(path, s).unwrap();
}

/// Minimal self-contained SVG spectrum plot (viewable in any browser).
pub fn write_svg(path: &Path, db: &[f64], title: &str, nyquist: f64, markers: &[(f64, String)]) {
    const W: f64 = 960.0;
    const H: f64 = 380.0;
    const M_L: f64 = 62.0;
    const M_R: f64 = 16.0;
    const M_T: f64 = 34.0;
    const M_B: f64 = 40.0;
    let y_min = -200.0;
    let y_max = 10.0;
    let x_of = |f: f64| M_L + (f / nyquist) * (W - M_L - M_R);
    let y_of = |v: f64| M_T + (y_max - v) / (y_max - y_min) * (H - M_T - M_B);

    let mut s = String::new();
    let _ = writeln!(
        s,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">"#
    );
    let _ = writeln!(s, r#"<rect width="{W}" height="{H}" fill="white"/>"#);
    let _ = writeln!(
        s,
        r#"<text x="{}" y="20" font-size="14" font-family="sans-serif" font-weight="bold">{}</text>"#,
        M_L, title
    );
    // horizontal grid every 20 dB
    let mut v = y_max;
    while v >= y_min {
        let y = y_of(v);
        let _ = writeln!(
            s,
            r##"<line x1="{M_L}" y1="{y}" x2="{}" y2="{y}" stroke="#ddd" stroke-width="1"/>"##,
            W - M_R
        );
        let _ = writeln!(
            s,
            r#"<text x="{}" y="{y}" font-size="10" font-family="monospace" text-anchor="end" dominant-baseline="middle">{v}</text>"#,
            M_L - 6.0
        );
        v -= 20.0;
    }
    // vertical grid every 2 kHz
    let mut f = 0.0;
    while f <= nyquist {
        let x = x_of(f);
        let _ = writeln!(
            s,
            r##"<line x1="{x}" y1="{M_T}" x2="{x}" y2="{}" stroke="#eee" stroke-width="1"/>"##,
            H - M_B
        );
        let _ = writeln!(
            s,
            r#"<text x="{x}" y="{}" font-size="10" font-family="sans-serif" text-anchor="middle">{:.0}k</text>"#,
            H - M_B + 14.0,
            f / 1000.0
        );
        f += 2000.0;
    }
    // spectrum polyline, max-abs downsampled to pixel columns
    let cols = (W - M_L - M_R) as usize;
    let mut pts = String::new();
    for col in 0..cols {
        let b0 = col * db.len() / cols;
        let b1 = ((col + 1) * db.len() / cols).max(b0 + 1);
        let peak = db[b0..b1].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let _ = write!(
            pts,
            "{:.1},{:.1} ",
            M_L + col as f64,
            y_of(peak.clamp(y_min, y_max))
        );
    }
    let _ = writeln!(
        s,
        r##"<polyline points="{pts}" fill="none" stroke="#0a58ca" stroke-width="1"/>"##
    );
    // markers
    for (f, label) in markers {
        let x = x_of(*f);
        let _ = writeln!(
            s,
            r##"<line x1="{x}" y1="{M_T}" x2="{x}" y2="{}" stroke="#c62828" stroke-dasharray="4 3"/>"##,
            H - M_B
        );
        let _ = writeln!(
            s,
            r##"<text x="{x}" y="{}" font-size="10" font-family="sans-serif" fill="#c62828" text-anchor="middle">{}</text>"##,
            M_T - 4.0,
            label.replace('&', "&amp;")
        );
    }
    let _ = writeln!(s, "</svg>");
    std::fs::write(path, s).unwrap();
}

pub fn write_raw_f64(path: &Path, x: &[f64]) {
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}
