//! Shared helpers for the `simple_src` benchmark suite.
//!
//! Each bench target includes this module with `#[path = "common/mod.rs"]`,
//! so the same `Conv`/`Shape` definitions and throughput helpers stay in one
//! place instead of being duplicated across bench files.
#![allow(dead_code)]

use simple_src::{Convert, Quality, SrcManager, flush_planar, process_planar};

pub enum Conv {
    C44k48k,
    C44k96k,
    C48k44k,
    C48k96k,
    C96k44k,
    C96k48k,
}

impl std::fmt::Display for Conv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Conv::C44k48k => "44k to 48k",
            Conv::C44k96k => "44k to 96k",
            Conv::C48k44k => "48k to 44k",
            Conv::C48k96k => "48k to 96k",
            Conv::C96k44k => "96k to 44k",
            Conv::C96k48k => "96k to 48k",
        };
        f.write_str(label)
    }
}

pub const R44K48K: f64 = 48000.0 / 44100.0;
pub const R44K96K: f64 = 96000.0 / 44100.0;
pub const R48K44K: f64 = 44100.0 / 48000.0;
pub const R48K96K: f64 = 2.0;
pub const R96K44K: f64 = 44100.0 / 96000.0;
pub const R96K48K: f64 = 0.5;
pub const TRANS44K: f64 = 2050.0 / 22050.0;
pub const TRANS48K: f64 = 4000.0 / 24000.0;

impl Conv {
    pub fn sample_num_10ms(&self) -> usize {
        match self {
            Conv::C48k44k | Conv::C96k44k => 441,
            Conv::C44k48k | Conv::C96k48k => 480,
            _ => 960,
        }
    }

    pub fn ratio(&self) -> f64 {
        match self {
            Conv::C44k48k => R44K48K,
            Conv::C44k96k => R44K96K,
            Conv::C48k44k => R48K44K,
            Conv::C48k96k => R48K96K,
            Conv::C96k44k => R96K44K,
            Conv::C96k48k => R96K48K,
        }
    }

    pub fn trans_width(&self) -> f64 {
        match self {
            Conv::C48k96k | Conv::C96k48k => TRANS48K,
            _ => TRANS44K,
        }
    }

    pub fn sample_rates(&self) -> (u32, u32) {
        match self {
            Conv::C44k48k => (44_100, 48_000),
            Conv::C44k96k => (44_100, 96_000),
            Conv::C48k44k => (48_000, 44_100),
            Conv::C48k96k => (48_000, 96_000),
            Conv::C96k44k => (96_000, 44_100),
            Conv::C96k48k => (96_000, 48_000),
        }
    }
}

// ---------------------------------------------------------------------------
// Batch / alternative-API helpers
// ---------------------------------------------------------------------------

/// Stage buffer capacity for batch benches (samples per `process_block` call).
pub const STAGE: usize = 4096;

/// Total output samples for a "1s" bench (matches the iterator benches).
pub fn conv_total_out(conv: &Conv) -> usize {
    conv.sample_num_10ms() * 100
}

/// Input long enough to produce `total_out` samples at `ratio`.
pub fn input_for(ratio: f64, total_out: usize) -> Vec<f64> {
    let n = ((total_out as f64) / ratio).ceil() as usize + 64;
    (0..n).map(|x| x as f64).collect()
}

/// Convert through `process_block` in slices of at most `quantum` output
/// samples, optionally draining the delay line with `flush` at the end.
/// Returns a checksum so the optimizer cannot elide the work.
pub fn batch_throughput(
    m: &SrcManager,
    input: &[f64],
    total_out: usize,
    quantum: usize,
    drain: bool,
) -> f64 {
    let mut cv = m.converter();
    let (mut cin, mut produced) = (0usize, 0usize);
    let mut sink = [0.0f64; STAGE];
    let mut acc = 0.0f64;
    while produced < total_out {
        let fill = STAGE.min(quantum).min(total_out - produced);
        let (c, p) = cv.process_block(&input[cin..], &mut sink[..fill]);
        if p == 0 {
            break;
        }
        cin += c;
        acc += divan::black_box(sink[p - 1]);
        produced += p;
    }
    if drain {
        loop {
            let n = cv.flush(&mut sink);
            if n == 0 {
                break;
            }
            acc += divan::black_box(sink[n - 1]);
        }
    }
    acc
}

/// Stereo planar conversion through `process_planar` + `flush_planar`.
pub fn planar_throughput(m: &SrcManager, left: &[f64], right: &[f64], total_out: usize) -> f64 {
    let mut convs = [m.converter(), m.converter()];
    let (mut cin, mut produced) = (0usize, 0usize);
    let mut sink_l = [0.0f64; STAGE];
    let mut sink_r = [0.0f64; STAGE];
    let mut acc = 0.0f64;
    while produced < total_out {
        let n_in = (left.len() - cin).min(STAGE);
        let n_out = (total_out - produced).min(STAGE);
        if n_in == 0 {
            break;
        }
        let (c, p) = process_planar(
            &mut convs,
            &[&left[cin..cin + n_in], &right[cin..cin + n_in]],
            &mut [&mut sink_l[..n_out], &mut sink_r[..n_out]],
        )
        .unwrap();
        if p == 0 {
            break;
        }
        cin += c;
        acc += divan::black_box(sink_l[p - 1]) + divan::black_box(sink_r[p - 1]);
        produced += p;
    }
    loop {
        let n = flush_planar(
            &mut convs,
            &mut [&mut sink_l[..STAGE], &mut sink_r[..STAGE]],
        )
        .unwrap();
        if n == 0 {
            break;
        }
        acc += divan::black_box(sink_l[n - 1]) + divan::black_box(sink_r[n - 1]);
    }
    acc
}

/// Build a sinc manager for `conv` at `atten` dB, selecting the Generic
/// (half-table) or Fast (polyphase LUT) path. `quantify` follows the
/// `Quality` preset pairing and is ignored by the Fast path.
pub fn sinc_manager(conv: &Conv, atten: f64, fast: bool) -> SrcManager {
    let quantify = match atten {
        a if a <= 48.0 => 8,
        a if a <= 96.0 => 128,
        a if a <= 120.0 => 512,
        _ => 2048,
    };
    let b = SrcManager::builder().ratio(conv.ratio());
    let b = if fast { b.fast() } else { b.generic() };
    b.attenuation(atten)
        .quantify(quantify)
        .trans_width(conv.trans_width())
        .build()
        .unwrap()
}

/// Build a sinc manager using the typical README path
/// (`sample_rate + quality + pass_freq`) instead of the raw attenuation path.
pub fn quality_sinc_manager(conv: &Conv, fast: bool) -> SrcManager {
    let (old_sr, new_sr) = conv.sample_rates();
    let mut b = SrcManager::builder()
        .sample_rate(old_sr, new_sr)
        .quality(Quality::Bit16Fast)
        .pass_freq(20_000);
    b = if fast { b.fast() } else { b.generic() };
    b.build().unwrap()
}

/// Build a sinc manager for the extra ratio-shape cases.
pub fn shape_sinc_manager(shape: &Shape, fast: bool) -> SrcManager {
    let mut b = SrcManager::builder()
        .ratio(shape.ratio())
        .attenuation(96.0)
        .quantify(128)
        .trans_width(0.05);
    b = if fast { b.fast() } else { b.generic() };
    b.build().unwrap()
}

/// Streaming 10ms sinc conversion through `process_block` with a caller-chosen
/// chunk size (no flush).
pub fn sinc_batch_throughput_q(m: &SrcManager, conv: &Conv, quantum: usize) -> f64 {
    let total_out = conv.sample_num_10ms();
    let input = input_for(conv.ratio(), total_out);
    batch_throughput(m, &input, total_out, quantum, false)
}

/// Streaming 10ms sinc conversion through `process_block` (no flush).
pub fn sinc_batch_throughput(m: &SrcManager, conv: &Conv) -> f64 {
    sinc_batch_throughput_q(m, conv, conv.sample_num_10ms())
}

/// Iterator throughput for a sinc manager (10ms of output samples).
pub fn sinc_iter_throughput(m: &SrcManager, conv: &Conv) -> f64 {
    let mut acc = 0.0f64;
    let iter = (0..).map(|x| x as f64);
    for s in m.converter().process(iter).take(conv.sample_num_10ms()) {
        acc += divan::black_box(s);
    }
    acc
}

/// Extra ratio-shape cases not covered by `Conv` (all are 1s @ 48k output).
pub enum Shape {
    /// Irrational ratio: exercises the float phase accumulator.
    FloatPi,
    /// Rational with numerator > 16384: exercises the non-table rational path.
    Generic20000Of19999,
    /// Extreme up-sampling bound of the supported ratio range.
    Up16,
    /// Extreme down-sampling bound of the supported ratio range.
    Down16,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Shape::FloatPi => "float pi",
            Shape::Generic20000Of19999 => "generic 20000/19999",
            Shape::Up16 => "16x up",
            Shape::Down16 => "16x down",
        })
    }
}

impl Shape {
    pub fn manager(&self) -> SrcManager {
        match self {
            Shape::FloatPi => SrcManager::with_ratio(std::f64::consts::PI).unwrap(),
            Shape::Generic20000Of19999 => SrcManager::with_sample_rate(19_999, 20_000).unwrap(),
            Shape::Up16 => SrcManager::with_ratio(16.0).unwrap(),
            Shape::Down16 => SrcManager::with_ratio(1.0 / 16.0).unwrap(),
        }
    }

    pub fn ratio(&self) -> f64 {
        match self {
            Shape::FloatPi => std::f64::consts::PI,
            Shape::Generic20000Of19999 => 20_000.0 / 19_999.0,
            Shape::Up16 => 16.0,
            Shape::Down16 => 1.0 / 16.0,
        }
    }
}

pub const SHAPE_TOTAL_OUT: usize = 48_000;
