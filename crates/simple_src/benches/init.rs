#[path = "common/mod.rs"]
mod common;
use common::*;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// One-shot manager construction (LUT build + trim design). No Throughput:
/// the interesting number is the build time itself.
fn init_conv(c: &mut Criterion, name: &str, atten: f64, fast: bool) {
    let mut g = c.benchmark_group(name);
    for conv in ALL_CONVS {
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            b.iter(|| sinc_manager(conv, atten, fast));
        });
    }
    g.finish();
}

fn init_a96(c: &mut Criterion) {
    init_conv(c, "1. init a96", 96.0, false);
}

fn init_a120(c: &mut Criterion) {
    init_conv(c, "2. init a120", 120.0, false);
}

fn init_a144(c: &mut Criterion) {
    init_conv(c, "3. init a144", 144.0, false);
}

fn init_a96_fast(c: &mut Criterion) {
    init_conv(c, "1. init a96 fast", 96.0, true);
}

fn init_a120_fast(c: &mut Criterion) {
    init_conv(c, "2. init a120 fast", 120.0, true);
}

fn init_a144_fast(c: &mut Criterion) {
    init_conv(c, "3. init a144 fast", 144.0, true);
}

fn init_quality_bit16_generic(c: &mut Criterion) {
    let mut g = c.benchmark_group("6. init bit16 quality generic");
    for conv in ALL_CONVS {
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            b.iter(|| quality_sinc_manager(conv, false));
        });
    }
    g.finish();
}

fn init_quality_bit16_fast(c: &mut Criterion) {
    let mut g = c.benchmark_group("6. init bit16 quality fast");
    for conv in ALL_CONVS {
        g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
            b.iter(|| quality_sinc_manager(conv, true));
        });
    }
    g.finish();
}

/// Extra ratio-shape cases not covered by the conversion directions.
fn init_shape(c: &mut Criterion, name: &str, shapes: &[Shape], fast: bool) {
    let mut g = c.benchmark_group(name);
    for shape in shapes {
        g.bench_with_input(BenchmarkId::from_parameter(shape), shape, |b, shape| {
            b.iter(|| shape_sinc_manager(shape, fast));
        });
    }
    g.finish();
}

fn sinc_generic_shape_init(c: &mut Criterion) {
    init_shape(c, "5. sinc generic shape init", &ALL_SHAPES, false);
}

fn sinc_fast_shape_init(c: &mut Criterion) {
    init_shape(c, "5. sinc fast shape init", &UP16_SHAPES, true);
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets =
        init_a96,
        init_a120,
        init_a144,
        init_a96_fast,
        init_a120_fast,
        init_a144_fast,
        init_quality_bit16_generic,
        init_quality_bit16_fast,
        sinc_generic_shape_init,
        sinc_fast_shape_init,
}
criterion_main!(benches);
