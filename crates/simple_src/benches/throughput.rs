#[path = "common/mod.rs"]
mod common;
use common::*;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use simple_src::{Convert, Kernel, SrcManager};
use std::hint::black_box;

fn linear_1s(c: &mut Criterion) {
    let mut g = c.benchmark_group("0. linear 1s");
    for conv in ALL_CONVS {
        let sample_num = conv.sample_num_10ms() * 100;
        g.throughput(Throughput::Elements(sample_num as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let manager = SrcManager::with_ratio(conv.ratio()).unwrap();
            b.iter(|| {
                let iter = (0..).map(|x| x as f64);
                for s in manager.converter().process(iter).take(sample_num) {
                    black_box(s);
                }
            });
        });
    }
    g.finish();
}

fn cubic_1s(c: &mut Criterion) {
    let mut g = c.benchmark_group("0. cubic 1s");
    for conv in ALL_CONVS {
        let sample_num = conv.sample_num_10ms() * 100;
        g.throughput(Throughput::Elements(sample_num as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let manager = SrcManager::builder()
                .ratio(conv.ratio())
                .kernel(Kernel::Cubic)
                .build()
                .unwrap();
            b.iter(|| {
                let iter = (0..).map(|x| x as f64);
                for s in manager.converter().process(iter).take(sample_num) {
                    black_box(s);
                }
            });
        });
    }
    g.finish();
}

/// Iterator sinc benches: one 10ms quantum of output per call.
fn proc_iter(c: &mut Criterion, name: &str, atten: f64) {
    let mut g = c.benchmark_group(name);
    for conv in ALL_CONVS {
        g.throughput(Throughput::Elements(conv.sample_num_10ms() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let manager = sinc_manager(conv, atten, false);
            b.iter(|| {
                let iter = (0..).map(|x| x as f64);
                for s in manager
                    .converter()
                    .process(iter)
                    .take(conv.sample_num_10ms())
                {
                    black_box(s);
                }
            });
        });
    }
    g.finish();
}

fn proc_a96_10ms(c: &mut Criterion) {
    proc_iter(c, "1. proc a96 10ms", 96.0);
}

fn proc_a120_10ms(c: &mut Criterion) {
    proc_iter(c, "2. proc a120 10ms", 120.0);
}

fn proc_a144_10ms(c: &mut Criterion) {
    proc_iter(c, "3. proc a144 10ms", 144.0);
}

fn linear_1s_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("0. linear 1s batch");
    for conv in ALL_CONVS {
        let total_out = conv_total_out(&conv);
        g.throughput(Throughput::Elements(total_out as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = SrcManager::with_ratio(conv.ratio()).unwrap();
            let input = input_for(conv.ratio(), total_out);
            b.iter(|| batch_throughput(&m, &input, total_out, STAGE, true));
        });
    }
    g.finish();
}

fn linear_1s_batch_10ms(c: &mut Criterion) {
    let mut g = c.benchmark_group("0. linear 1s batch 10ms chunks");
    for conv in ALL_CONVS {
        let total_out = conv_total_out(&conv);
        g.throughput(Throughput::Elements(total_out as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = SrcManager::with_ratio(conv.ratio()).unwrap();
            let input = input_for(conv.ratio(), total_out);
            // Streaming shape: one process_block call per ~10ms of output.
            let quantum = conv.sample_num_10ms();
            b.iter(|| batch_throughput(&m, &input, total_out, quantum, true));
        });
    }
    g.finish();
}

fn linear_1s_convert(c: &mut Criterion) {
    let mut g = c.benchmark_group("0. linear 1s convert (incl. alloc)");
    for conv in ALL_CONVS {
        let total_out = conv_total_out(&conv);
        g.throughput(Throughput::Elements(total_out as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = SrcManager::with_ratio(conv.ratio()).unwrap();
            let input = input_for(conv.ratio(), total_out);
            b.iter(|| {
                let out = m.convert(&input);
                black_box(&out);
            });
        });
    }
    g.finish();
}

fn linear_1s_planar(c: &mut Criterion) {
    let mut g = c.benchmark_group("0. linear 1s planar stereo");
    for conv in ALL_CONVS {
        let total_out = conv_total_out(&conv);
        g.throughput(Throughput::Elements(2 * total_out as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = SrcManager::with_ratio(conv.ratio()).unwrap();
            let input = input_for(conv.ratio(), total_out);
            let right = input.iter().map(|x| -x).collect::<Vec<f64>>();
            b.iter(|| planar_throughput(&m, &input, &right, total_out));
        });
    }
    g.finish();
}

fn cubic_1s_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("0. cubic 1s batch");
    for conv in ALL_CONVS {
        let total_out = conv_total_out(&conv);
        g.throughput(Throughput::Elements(total_out as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = SrcManager::builder()
                .ratio(conv.ratio())
                .kernel(Kernel::Cubic)
                .build()
                .unwrap();
            let input = input_for(conv.ratio(), total_out);
            b.iter(|| batch_throughput(&m, &input, total_out, STAGE, true));
        });
    }
    g.finish();
}

/// Streaming 10ms sinc conversion through `process_block`. Unlike the 1s
/// batch benches the helper allocates its input inside the timed region —
/// part of the measured workload by design.
fn proc_batch(c: &mut Criterion, name: &str, atten: f64) {
    let mut g = c.benchmark_group(name);
    for conv in ALL_CONVS {
        g.throughput(Throughput::Elements(conv.sample_num_10ms() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = sinc_manager(conv, atten, false);
            b.iter(|| sinc_batch_throughput(&m, conv));
        });
    }
    g.finish();
}

fn proc_a96_10ms_batch(c: &mut Criterion) {
    proc_batch(c, "1. proc a96 10ms batch", 96.0);
}

fn proc_a120_10ms_batch(c: &mut Criterion) {
    proc_batch(c, "2. proc a120 10ms batch", 120.0);
}

fn proc_a144_10ms_batch(c: &mut Criterion) {
    proc_batch(c, "3. proc a144 10ms batch", 144.0);
}

fn linear_shape_1s(c: &mut Criterion) {
    let mut g = c.benchmark_group("0. linear shape iterator");
    for shape in ALL_SHAPES {
        g.throughput(Throughput::Elements(SHAPE_TOTAL_OUT as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&shape), &shape, |b, shape| {
            let m = shape.manager();
            b.iter(|| {
                let iter = (0..).map(|x| x as f64);
                for s in m.converter().process(iter).take(SHAPE_TOTAL_OUT) {
                    black_box(s);
                }
            });
        });
    }
    g.finish();
}

fn linear_shape_1s_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("0. linear shape batch");
    for shape in ALL_SHAPES {
        g.throughput(Throughput::Elements(SHAPE_TOTAL_OUT as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&shape), &shape, |b, shape| {
            let m = shape.manager();
            let input = input_for(shape.ratio(), SHAPE_TOTAL_OUT);
            b.iter(|| batch_throughput(&m, &input, SHAPE_TOTAL_OUT, STAGE, true));
        });
    }
    g.finish();
}

/// Fast-path iterator benches: one 10ms quantum of output per call.
fn proc_fast(c: &mut Criterion, name: &str, atten: f64) {
    let mut g = c.benchmark_group(name);
    for conv in ALL_CONVS {
        g.throughput(Throughput::Elements(conv.sample_num_10ms() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let manager = sinc_manager(conv, atten, true);
            b.iter(|| sinc_iter_throughput(&manager, conv));
        });
    }
    g.finish();
}

fn proc_a96_10ms_fast(c: &mut Criterion) {
    proc_fast(c, "1. proc a96 10ms fast", 96.0);
}

fn proc_a120_10ms_fast(c: &mut Criterion) {
    proc_fast(c, "2. proc a120 10ms fast", 120.0);
}

fn proc_a144_10ms_fast(c: &mut Criterion) {
    proc_fast(c, "3. proc a144 10ms fast", 144.0);
}

fn proc_a96_10ms_fast_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("1. proc a96 10ms fast batch");
    for conv in ALL_CONVS {
        g.throughput(Throughput::Elements(conv.sample_num_10ms() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let manager = sinc_manager(conv, 96.0, true);
            b.iter(|| sinc_batch_throughput(&manager, conv));
        });
    }
    g.finish();
}

fn proc_a120_10ms_fast_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("2. proc a120 10ms fast batch");
    for conv in ALL_CONVS {
        g.throughput(Throughput::Elements(conv.sample_num_10ms() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let manager = sinc_manager(conv, 120.0, true);
            b.iter(|| sinc_batch_throughput(&manager, conv));
        });
    }
    g.finish();
}

fn proc_a144_10ms_fast_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("3. proc a144 10ms fast batch");
    for conv in ALL_CONVS {
        g.throughput(Throughput::Elements(conv.sample_num_10ms() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let manager = sinc_manager(conv, 144.0, true);
            b.iter(|| sinc_batch_throughput(&manager, conv));
        });
    }
    g.finish();
}

fn linear_1s_batch_64(c: &mut Criterion) {
    let mut g = c.benchmark_group("0. linear 1s batch 64");
    for conv in QUANTUM_CONVS {
        let total_out = conv_total_out(&conv);
        g.throughput(Throughput::Elements(total_out as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = SrcManager::with_ratio(conv.ratio()).unwrap();
            let input = input_for(conv.ratio(), total_out);
            b.iter(|| batch_throughput(&m, &input, total_out, 64, true));
        });
    }
    g.finish();
}

fn cubic_1s_batch_64(c: &mut Criterion) {
    let mut g = c.benchmark_group("0. cubic 1s batch 64");
    for conv in QUANTUM_CONVS {
        let total_out = conv_total_out(&conv);
        g.throughput(Throughput::Elements(total_out as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = SrcManager::builder()
                .ratio(conv.ratio())
                .kernel(Kernel::Cubic)
                .build()
                .unwrap();
            let input = input_for(conv.ratio(), total_out);
            b.iter(|| batch_throughput(&m, &input, total_out, 64, true));
        });
    }
    g.finish();
}

/// Fixed-quantum (64/256) streaming sinc benches.
fn proc_batch_q(c: &mut Criterion, name: &str, quantum: usize, fast: bool) {
    let mut g = c.benchmark_group(name);
    for conv in QUANTUM_CONVS {
        g.throughput(Throughput::Elements(conv.sample_num_10ms() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = sinc_manager(conv, 96.0, fast);
            b.iter(|| sinc_batch_throughput_q(&m, conv, quantum));
        });
    }
    g.finish();
}

fn proc_a96_10ms_batch_64(c: &mut Criterion) {
    proc_batch_q(c, "1. proc a96 10ms batch 64", 64, false);
}

fn proc_a96_10ms_batch_256(c: &mut Criterion) {
    proc_batch_q(c, "1. proc a96 10ms batch 256", 256, false);
}

fn proc_a96_10ms_fast_batch_64(c: &mut Criterion) {
    proc_batch_q(c, "1. proc a96 10ms fast batch 64", 64, true);
}

fn proc_a96_10ms_fast_batch_256(c: &mut Criterion) {
    proc_batch_q(c, "1. proc a96 10ms fast batch 256", 256, true);
}

fn sinc_generic_shape_1s_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("5. sinc generic shape 1s batch");
    for shape in ALL_SHAPES {
        g.throughput(Throughput::Elements(SHAPE_TOTAL_OUT as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&shape), &shape, |b, shape| {
            let m = shape_sinc_manager(shape, false);
            let input = input_for(shape.ratio(), SHAPE_TOTAL_OUT);
            b.iter(|| batch_throughput(&m, &input, SHAPE_TOTAL_OUT, STAGE, true));
        });
    }
    g.finish();
}

fn sinc_fast_shape_1s_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("5. sinc fast shape 1s batch");
    for shape in UP16_SHAPES {
        g.throughput(Throughput::Elements(SHAPE_TOTAL_OUT as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&shape), &shape, |b, shape| {
            let m = shape_sinc_manager(shape, true);
            let input = input_for(shape.ratio(), SHAPE_TOTAL_OUT);
            b.iter(|| batch_throughput(&m, &input, SHAPE_TOTAL_OUT, STAGE, true));
        });
    }
    g.finish();
}

fn sinc_a96_1s_convert_generic(c: &mut Criterion) {
    let mut g = c.benchmark_group("1. sinc a96 1s convert generic");
    for conv in CONVERT_CONVS {
        let total_out = conv_total_out(&conv);
        g.throughput(Throughput::Elements(total_out as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = sinc_manager(conv, 96.0, false);
            let input = input_for(conv.ratio(), total_out);
            b.iter(|| {
                let out = m.convert(&input);
                black_box(&out);
            });
        });
    }
    g.finish();
}

fn sinc_a96_1s_convert_fast(c: &mut Criterion) {
    let mut g = c.benchmark_group("1. sinc a96 1s convert fast");
    for conv in CONVERT_CONVS {
        let total_out = conv_total_out(&conv);
        g.throughput(Throughput::Elements(total_out as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = sinc_manager(conv, 96.0, true);
            let input = input_for(conv.ratio(), total_out);
            b.iter(|| {
                let out = m.convert(&input);
                black_box(&out);
            });
        });
    }
    g.finish();
}

fn sinc_a96_1s_planar_fast(c: &mut Criterion) {
    let mut g = c.benchmark_group("1. sinc a96 1s planar fast");
    for conv in CONVERT_CONVS {
        let total_out = conv_total_out(&conv);
        g.throughput(Throughput::Elements(2 * total_out as u64));
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            let m = sinc_manager(conv, 96.0, true);
            let input = input_for(conv.ratio(), total_out);
            let right = input.iter().map(|x| -x).collect::<Vec<f64>>();
            b.iter(|| planar_throughput(&m, &input, &right, total_out));
        });
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets =
        linear_1s,
        cubic_1s,
        proc_a96_10ms,
        proc_a120_10ms,
        proc_a144_10ms,
        linear_1s_batch,
        linear_1s_batch_10ms,
        linear_1s_convert,
        linear_1s_planar,
        cubic_1s_batch,
        proc_a96_10ms_batch,
        proc_a120_10ms_batch,
        proc_a144_10ms_batch,
        linear_shape_1s,
        linear_shape_1s_batch,
        proc_a96_10ms_fast,
        proc_a96_10ms_fast_batch,
        proc_a120_10ms_fast,
        proc_a120_10ms_fast_batch,
        proc_a144_10ms_fast,
        proc_a144_10ms_fast_batch,
        linear_1s_batch_64,
        cubic_1s_batch_64,
        proc_a96_10ms_batch_64,
        proc_a96_10ms_batch_256,
        proc_a96_10ms_fast_batch_64,
        proc_a96_10ms_fast_batch_256,
        sinc_generic_shape_1s_batch,
        sinc_fast_shape_1s_batch,
        sinc_a96_1s_convert_generic,
        sinc_a96_1s_convert_fast,
        sinc_a96_1s_planar_fast,
}
criterion_main!(benches);
