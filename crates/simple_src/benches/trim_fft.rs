use divan::Bencher;
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

fn main() {
    divan::main();
}

/// Baseline formula design (no trim search) for comparison.
#[divan::bench(name = "trim. 48->44.1 formula", sample_count = 20)]
fn formula_48_44(bencher: Bencher) {
    bencher.bench_local(build_formula);
}

/// Default measured-trim build. With the `rustfft` feature (default) this
/// exercises RustFFT; with `--no-default-features` it exercises the
/// hand-written radix-2 FFT.
#[divan::bench(name = "trim. 48->44.1 A=96", sample_count = 20)]
fn trim_48_44(bencher: Bencher) {
    bencher.bench_local(|| build_trim(44100.0 / 48000.0, 96.0, 2050.0 / 22050.0));
}

/// Common upsampling case with a narrow stop-band scan.
#[divan::bench(name = "trim. 44.1->48 A=96", sample_count = 20)]
fn trim_44_48(bencher: Bencher) {
    bencher.bench_local(|| build_trim(48000.0 / 44100.0, 96.0, 2050.0 / 22050.0));
}

/// Heavy case: low ratio + wide stop-band scan + high attenuation.
#[divan::bench(name = "trim. 96->44.1 A=144", sample_count = 5)]
fn trim_96_44_a144(bencher: Bencher) {
    bencher.bench_local(|| build_trim(44100.0 / 96000.0, 144.0, 2050.0 / 22050.0));
}
