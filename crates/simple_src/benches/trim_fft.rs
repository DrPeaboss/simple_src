use criterion::{Criterion, PlottingBackend, criterion_group, criterion_main};
use simple_src::SrcManager;

fn build_formula() -> SrcManager {
    SrcManager::builder()
        .sample_rate(48000, 44100)
        .attenuation(96.0)
        .trans_width(2050.0 / 22050.0)
        .fast()
        .build()
        .unwrap()
}

fn build_trim(ratio: f64, atten: f64, tw: f64) -> SrcManager {
    let mut b = SrcManager::builder()
        .ratio(ratio)
        .attenuation(atten)
        .trans_width(tw)
        .fast();
    b = b.trim_filter(true);
    b.build().unwrap()
}

/// Baseline formula design (no trim search) for comparison.
fn formula_48_44(c: &mut Criterion) {
    c.bench_function("trim. 48->44.1 formula", |b| b.iter(build_formula));
}

/// Default measured-trim build. With the `rustfft` feature (default) this
/// exercises RustFFT; with `--no-default-features` it exercises the
/// hand-written radix-2 FFT.
fn trim_48_44(c: &mut Criterion) {
    c.bench_function("trim. 48->44.1 A=96", |b| {
        b.iter(|| build_trim(44100.0 / 48000.0, 96.0, 2050.0 / 22050.0))
    });
}

/// Common upsampling case with a narrow stop-band scan.
fn trim_44_48(c: &mut Criterion) {
    c.bench_function("trim. 44.1->48 A=96", |b| {
        b.iter(|| build_trim(48000.0 / 44100.0, 96.0, 2050.0 / 22050.0))
    });
}

/// Heavy case: low ratio + wide stop-band scan + high attenuation.
fn trim_96_44_a144(c: &mut Criterion) {
    c.bench_function("trim. 96->44.1 A=144", |b| {
        b.iter(|| build_trim(44100.0 / 96000.0, 144.0, 2050.0 / 22050.0))
    });
}

use std::time::Duration;

// Mirrors common::criterion_config (this target does not include common):
// plotting off and a 10x-smaller bootstrap because the perf-baseline
// tooling reads the median point estimate; both otherwise dominate the
// per-bench wall time.
criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(1))
        .sample_size(50)
        .nresamples(10_000)
        .plotting_backend(PlottingBackend::None);
    targets = formula_48_44, trim_48_44, trim_44_48, trim_96_44_a144
}
criterion_main!(benches);
