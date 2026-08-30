use simple_src::{Convert, Kernel, SrcManager, flush_planar, process_planar};

fn main() {
    divan::main();
}

enum Conv {
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

const R44K48K: f64 = 48000.0 / 44100.0;
const R44K96K: f64 = 96000.0 / 44100.0;
const R48K44K: f64 = 44100.0 / 48000.0;
const R48K96K: f64 = 2.0;
const R96K44K: f64 = 44100.0 / 96000.0;
const R96K48K: f64 = 0.5;
const TRANS44K: f64 = 2050.0 / 22050.0;
const TRANS48K: f64 = 4000.0 / 24000.0;

impl Conv {
    fn sample_num_10ms(&self) -> usize {
        match self {
            Conv::C48k44k | Conv::C96k44k => 441,
            Conv::C44k48k | Conv::C96k48k => 480,
            _ => 960,
        }
    }

    fn ratio(&self) -> f64 {
        match self {
            Conv::C44k48k => R44K48K,
            Conv::C44k96k => R44K96K,
            Conv::C48k44k => R48K44K,
            Conv::C48k96k => R48K96K,
            Conv::C96k44k => R96K44K,
            Conv::C96k48k => R96K48K,
        }
    }

    fn trans_width(&self) -> f64 {
        match self {
            Conv::C48k96k | Conv::C96k48k => TRANS48K,
            _ => TRANS44K,
        }
    }
}

// ---------------------------------------------------------------------------
// Batch / alternative-API helpers
// ---------------------------------------------------------------------------

/// Stage buffer capacity for batch benches (samples per `process_block` call).
const STAGE: usize = 4096;

/// Total output samples for a "1s" bench (matches the iterator benches).
fn conv_total_out(conv: &Conv) -> usize {
    conv.sample_num_10ms() * 100
}

/// Input long enough to produce `total_out` samples at `ratio`.
fn input_for(ratio: f64, total_out: usize) -> Vec<f64> {
    let n = ((total_out as f64) / ratio).ceil() as usize + 64;
    (0..n).map(|x| x as f64).collect()
}

/// Convert through `process_block` in slices of at most `quantum` output
/// samples, optionally draining the delay line with `flush` at the end.
/// Returns a checksum so the optimizer cannot elide the work.
fn batch_throughput(
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
fn planar_throughput(m: &SrcManager, left: &[f64], right: &[f64], total_out: usize) -> f64 {
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
fn sinc_manager(conv: &Conv, atten: f64, fast: bool) -> SrcManager {
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

/// Streaming 10ms sinc conversion through `process_block` (no flush).
/// Streaming 10ms sinc conversion through `process_block` (no flush).
fn sinc_batch_throughput(m: &SrcManager, conv: &Conv) -> f64 {
    let total_out = conv.sample_num_10ms();
    let input = input_for(conv.ratio(), total_out);
    batch_throughput(m, &input, total_out, conv.sample_num_10ms(), false)
}

/// Iterator throughput for a sinc manager (10ms of output samples).
fn sinc_iter_throughput(m: &SrcManager, conv: &Conv) -> f64 {
    let mut acc = 0.0f64;
    let iter = (0..).map(|x| x as f64);
    for s in m.converter().process(iter).take(conv.sample_num_10ms()) {
        acc += divan::black_box(s);
    }
    acc
}

/// Extra ratio-shape cases not covered by `Conv` (all are 1s @ 48k output).
enum Shape {
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
    fn manager(&self) -> SrcManager {
        match self {
            Shape::FloatPi => SrcManager::with_ratio(std::f64::consts::PI).unwrap(),
            Shape::Generic20000Of19999 => SrcManager::with_sample_rate(19_999, 20_000).unwrap(),
            Shape::Up16 => SrcManager::with_ratio(16.0).unwrap(),
            Shape::Down16 => SrcManager::with_ratio(1.0 / 16.0).unwrap(),
        }
    }

    fn ratio(&self) -> f64 {
        match self {
            Shape::FloatPi => std::f64::consts::PI,
            Shape::Generic20000Of19999 => 20_000.0 / 19_999.0,
            Shape::Up16 => 16.0,
            Shape::Down16 => 1.0 / 16.0,
        }
    }
}

const SHAPE_TOTAL_OUT: usize = 48_000;

#[divan::bench(
    name="0. linear 1s",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn linear_1s(bencher: divan::Bencher, conv: &Conv) {
    let manager = SrcManager::with_ratio(conv.ratio()).unwrap();
    let sample_num = conv.sample_num_10ms() * 100;
    bencher.bench_local(move || {
        let iter = (0..).map(|x| x as f64);
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

#[divan::bench(
    name="0. cubic 1s",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn cubic_1s(bencher: divan::Bencher, conv: &Conv) {
    let manager = SrcManager::builder()
        .ratio(conv.ratio())
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();
    let sample_num = conv.sample_num_10ms() * 100;
    bencher.bench_local(move || {
        let iter = (0..).map(|x| x as f64);
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

#[divan::bench(
    name="1. init a96",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a96(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 96.0, false)
}

#[divan::bench(
    name="1. proc a96 10ms",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a96_10ms(bencher: divan::Bencher, conv: &Conv) {
    let manager = init_a96(conv);
    let sample_num = conv.sample_num_10ms();
    bencher.bench_local(move || {
        let iter = (0..).map(|x| x as f64);
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

#[divan::bench(
    name="2. init a120",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a120(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 120.0, false)
}

#[divan::bench(
    name="2. proc a120 10ms",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a120_10ms(bencher: divan::Bencher, conv: &Conv) {
    let manager = init_a120(conv);
    let sample_num = conv.sample_num_10ms();
    bencher.bench_local(move || {
        let iter = (0..).map(|x| x as f64);
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

#[divan::bench(
    name="3. init a144",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a144(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 144.0, false)
}

#[divan::bench(
    name="3. proc a144 10ms",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a144_10ms(bencher: divan::Bencher, conv: &Conv) {
    let manager = init_a144(conv);
    let sample_num = conv.sample_num_10ms();
    bencher.bench_local(move || {
        let iter = (0..).map(|x| x as f64);
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

// ---------------------------------------------------------------------------
// Batch (`process_block`) benches: the same scenarios as the iterator benches,
// so the per-sample dispatch path and the batch path stay comparable over time.
// ---------------------------------------------------------------------------

#[divan::bench(
    name = "0. linear 1s batch",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count = 300,
)]
fn linear_1s_batch(bencher: divan::Bencher, conv: &Conv) {
    let m = SrcManager::with_ratio(conv.ratio()).unwrap();
    let total_out = conv_total_out(conv);
    let input = input_for(conv.ratio(), total_out);
    bencher.bench_local(move || batch_throughput(&m, &input, total_out, STAGE, true));
}

#[divan::bench(
    name = "0. linear 1s batch 10ms chunks",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count = 300,
)]
fn linear_1s_batch_10ms(bencher: divan::Bencher, conv: &Conv) {
    let m = SrcManager::with_ratio(conv.ratio()).unwrap();
    let total_out = conv_total_out(conv);
    let input = input_for(conv.ratio(), total_out);
    // Streaming shape: one process_block call per ~10ms of output.
    let quantum = conv.sample_num_10ms();
    bencher.bench_local(move || batch_throughput(&m, &input, total_out, quantum, true));
}

#[divan::bench(
    name = "0. linear 1s convert (incl. alloc)",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count = 300,
)]
fn linear_1s_convert(bencher: divan::Bencher, conv: &Conv) {
    let m = SrcManager::with_ratio(conv.ratio()).unwrap();
    let total_out = conv_total_out(conv);
    let input = input_for(conv.ratio(), total_out);
    bencher.bench_local(move || {
        let out = m.convert(&input);
        divan::black_box(&out);
    });
}

#[divan::bench(
    name = "0. linear 1s planar stereo",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count = 300,
)]
fn linear_1s_planar(bencher: divan::Bencher, conv: &Conv) {
    let m = SrcManager::with_ratio(conv.ratio()).unwrap();
    let total_out = conv_total_out(conv);
    let input = input_for(conv.ratio(), total_out);
    let right = input.iter().map(|x| -x).collect::<Vec<f64>>();
    bencher.bench_local(move || planar_throughput(&m, &input, &right, total_out));
}

#[divan::bench(
    name = "0. cubic 1s batch",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count = 300,
)]
fn cubic_1s_batch(bencher: divan::Bencher, conv: &Conv) {
    let m = SrcManager::builder()
        .ratio(conv.ratio())
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();
    let total_out = conv_total_out(conv);
    let input = input_for(conv.ratio(), total_out);
    bencher.bench_local(move || batch_throughput(&m, &input, total_out, STAGE, true));
}

// --- sinc batch: streaming 10ms chunks, no flush (stream continues) ---

#[divan::bench(
    name = "1. proc a96 10ms batch",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count = 1000,
)]
fn proc_a96_10ms_batch(bencher: divan::Bencher, conv: &Conv) {
    let m = init_a96(conv);
    bencher.bench_local(move || sinc_batch_throughput(&m, conv));
}

#[divan::bench(
    name = "2. proc a120 10ms batch",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count = 1000,
)]
fn proc_a120_10ms_batch(bencher: divan::Bencher, conv: &Conv) {
    let m = init_a120(conv);
    bencher.bench_local(move || sinc_batch_throughput(&m, conv));
}

#[divan::bench(
    name = "3. proc a144 10ms batch",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count = 1000,
)]
fn proc_a144_10ms_batch(bencher: divan::Bencher, conv: &Conv) {
    let m = init_a144(conv);
    bencher.bench_local(move || sinc_batch_throughput(&m, conv));
}

// ---------------------------------------------------------------------------
// Ratio-shape coverage: float phase, large rational, and the ratio bounds.
// Each case has an iterator and a batch variant at 1s @ 48k output.
// ---------------------------------------------------------------------------

#[divan::bench(
    name = "0. linear shape iterator",
    args = [Shape::FloatPi, Shape::Generic20000Of19999, Shape::Up16, Shape::Down16],
    sample_count = 300,
)]
fn linear_shape_1s(bencher: divan::Bencher, shape: &Shape) {
    let m = shape.manager();
    let sample_num = SHAPE_TOTAL_OUT;
    bencher.bench_local(move || {
        let iter = (0..).map(|x| x as f64);
        for s in m.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

#[divan::bench(
    name = "0. linear shape batch",
    args = [Shape::FloatPi, Shape::Generic20000Of19999, Shape::Up16, Shape::Down16],
    sample_count = 300,
)]
fn linear_shape_1s_batch(bencher: divan::Bencher, shape: &Shape) {
    let m = shape.manager();
    let total_out = SHAPE_TOTAL_OUT;
    let input = input_for(shape.ratio(), total_out);
    bencher.bench_local(move || batch_throughput(&m, &input, total_out, STAGE, true));
}

// ---------------------------------------------------------------------------
// Sinc Fast path (polyphase LUT): init / iterator / batch, mirroring the
// Generic (half-table) presets above for a like-for-like comparison.
// `quantify` is ignored by the Fast path; attenuation and trans width match.
// ---------------------------------------------------------------------------

#[divan::bench(
    name="1. init a96 fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a96_fast(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 96.0, true)
}

#[divan::bench(
    name="1. proc a96 10ms fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a96_10ms_fast(bencher: divan::Bencher, conv: &Conv) {
    let manager = init_a96_fast(conv);
    bencher.bench_local(move || sinc_iter_throughput(&manager, conv))
}

#[divan::bench(
    name="1. proc a96 10ms fast batch",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a96_10ms_fast_batch(bencher: divan::Bencher, conv: &Conv) {
    let manager = init_a96_fast(conv);
    bencher.bench_local(move || sinc_batch_throughput(&manager, conv))
}

#[divan::bench(
    name="2. init a120 fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a120_fast(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 120.0, true)
}

#[divan::bench(
    name="2. proc a120 10ms fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a120_10ms_fast(bencher: divan::Bencher, conv: &Conv) {
    let manager = init_a120_fast(conv);
    bencher.bench_local(move || sinc_iter_throughput(&manager, conv))
}

#[divan::bench(
    name="2. proc a120 10ms fast batch",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a120_10ms_fast_batch(bencher: divan::Bencher, conv: &Conv) {
    let manager = init_a120_fast(conv);
    bencher.bench_local(move || sinc_batch_throughput(&manager, conv))
}

#[divan::bench(
    name="3. init a144 fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a144_fast(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 144.0, true)
}

#[divan::bench(
    name="3. proc a144 10ms fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a144_10ms_fast(bencher: divan::Bencher, conv: &Conv) {
    let manager = init_a144_fast(conv);
    bencher.bench_local(move || sinc_iter_throughput(&manager, conv))
}

#[divan::bench(
    name="3. proc a144 10ms fast batch",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a144_10ms_fast_batch(bencher: divan::Bencher, conv: &Conv) {
    let manager = init_a144_fast(conv);
    bencher.bench_local(move || sinc_batch_throughput(&manager, conv))
}

/// Forced dot-kernel benches (feature `internal-bench`): measure the portable
/// scalar fallback against the runtime-selected (AVX2+FMA) kernel on the same
/// machine. The AVX2-forced entries early-return on CPUs without AVX2+FMA, so
/// they only measure where the kernel actually runs; the scalar entries run
/// everywhere.
#[cfg(feature = "internal-bench")]
mod forced {
    use super::*;

    fn avx2_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn batch_forced(m: &SrcManager, conv: &Conv, force_scalar: bool) -> f64 {
        let total_out = conv.sample_num_10ms();
        let input = input_for(conv.ratio(), total_out);
        let mut cv = m.converter_forced_kernel(force_scalar);
        let mut sink = [0.0f64; STAGE];
        let (mut cin, mut produced, mut acc) = (0usize, 0usize, 0.0f64);
        while produced < total_out {
            let fill = STAGE.min(total_out - produced);
            let (c, p) = cv.process_block(&input[cin..], &mut sink[..fill]);
            if p == 0 {
                break;
            }
            cin += c;
            acc += divan::black_box(sink[p - 1]);
            produced += p;
        }
        acc
    }

    fn iter_forced(m: &SrcManager, conv: &Conv, force_scalar: bool) -> f64 {
        let mut acc = 0.0f64;
        let iter = (0..).map(|x| x as f64);
        for s in m
            .converter_forced_kernel(force_scalar)
            .process(iter)
            .take(conv.sample_num_10ms())
        {
            acc += divan::black_box(s);
        }
        acc
    }

    #[divan::bench(
        name = "4. forced scalar fast batch",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 500,
    )]
    fn forced_scalar_fast_batch(bencher: divan::Bencher, conv: &Conv) {
        let manager = sinc_manager(conv, 96.0, true);
        bencher.bench_local(move || batch_forced(&manager, conv, true));
    }

    #[divan::bench(
        name = "4. forced avx2 fast batch",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 500,
    )]
    fn forced_avx2_fast_batch(bencher: divan::Bencher, conv: &Conv) {
        if !avx2_available() {
            return; // AVX2 not present on this CPU: nothing to measure
        }
        let manager = sinc_manager(conv, 96.0, true);
        bencher.bench_local(move || batch_forced(&manager, conv, false));
    }

    #[divan::bench(
        name = "4. forced scalar fast iter",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 500,
    )]
    fn forced_scalar_fast_iter(bencher: divan::Bencher, conv: &Conv) {
        let manager = sinc_manager(conv, 96.0, true);
        bencher.bench_local(move || iter_forced(&manager, conv, true));
    }

    #[divan::bench(
        name = "4. forced avx2 fast iter",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 500,
    )]
    fn forced_avx2_fast_iter(bencher: divan::Bencher, conv: &Conv) {
        if !avx2_available() {
            return;
        }
        let manager = sinc_manager(conv, 96.0, true);
        bencher.bench_local(move || iter_forced(&manager, conv, false));
    }

    #[divan::bench(
        name = "4. forced scalar generic batch",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 200,
    )]
    fn forced_scalar_generic_batch(bencher: divan::Bencher, conv: &Conv) {
        let manager = sinc_manager(conv, 96.0, false);
        bencher.bench_local(move || batch_forced(&manager, conv, true));
    }

    #[divan::bench(
        name = "4. forced avx2 generic batch",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 200,
    )]
    fn forced_avx2_generic_batch(bencher: divan::Bencher, conv: &Conv) {
        if !avx2_available() {
            return;
        }
        let manager = sinc_manager(conv, 96.0, false);
        bencher.bench_local(move || batch_forced(&manager, conv, false));
    }
}
