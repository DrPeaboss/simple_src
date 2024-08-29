use criterion::{black_box, criterion_group, criterion_main, Criterion};
use simple_src::{sinc, Convert};

fn src_bench_init(c: &mut Criterion) {
    c.bench_function("src 44k to 48k a150 init", |b| {
        b.iter(|| {
            let manager = sinc::Manager::new(48000.0 / 44100.0, 150.0, 2048, 2050.0 / 22050.0);
            black_box(manager);
        })
    });
    c.bench_function("src 48k to 44k a150 init", |b| {
        b.iter(|| {
            let manager = sinc::Manager::new(44100.0 / 48000.0, 150.0, 2048, 2050.0 / 22050.0);
            black_box(manager);
        })
    });
    c.bench_function("src 96k to 44k a150 init", |b| {
        b.iter(|| {
            let manager = sinc::Manager::new(44100.0 / 96000.0, 150.0, 2048, 2050.0 / 22050.0);
            black_box(manager);
        })
    });
    c.bench_function("src 96k to 48k a150 init", |b| {
        b.iter(|| {
            let manager = sinc::Manager::new(48000.0 / 96000.0, 150.0, 2048, 4000.0 / 24000.0);
            black_box(manager);
        })
    });
}

fn src_bench_process_1s(c: &mut Criterion) {
    let manager = sinc::Manager::new(48000.0 / 44100.0, 150.0, 2048, 2050.0 / 22050.0);
    c.bench_function("src 44k to 48k a150 1s", |b| {
        b.iter(|| {
            let iter = (0..).map(|x| x as f64).into_iter();
            for s in manager.converter().process(iter).take(48000) {
                black_box(s);
            }
        })
    });
    let manager = sinc::Manager::new(44100.0 / 48000.0, 150.0, 2048, 2050.0 / 22050.0);
    c.bench_function("src 48k to 44k a150 1s", |b| {
        b.iter(|| {
            let iter = (0..).map(|x| x as f64).into_iter();
            for s in manager.converter().process(iter).take(44100) {
                black_box(s);
            }
        })
    });
    let manager = sinc::Manager::new(44100.0 / 96000.0, 150.0, 2048, 2050.0 / 22050.0);
    c.bench_function("src 96k to 44k a150 1s", |b| {
        b.iter(|| {
            let iter = (0..).map(|x| x as f64).into_iter();
            for s in manager.converter().process(iter).take(44100) {
                black_box(s);
            }
        })
    });
    let manager = sinc::Manager::new(48000.0 / 96000.0, 150.0, 2048, 4000.0 / 24000.0);
    c.bench_function("src 96k to 48k a150 1s", |b| {
        b.iter(|| {
            let iter = (0..).map(|x| x as f64).into_iter();
            for s in manager.converter().process(iter).take(48000) {
                black_box(s);
            }
        })
    });
}

criterion_group!(benches, src_bench_init, src_bench_process_1s);
criterion_main!(benches);
