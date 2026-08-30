#[path = "common/mod.rs"]
mod common;
use common::*;
use simple_src::{Convert, Kernel, SrcManager};

fn main() {
    divan::main();
}

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
    name="1. proc a96 10ms",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a96_10ms(bencher: divan::Bencher, conv: &Conv) {
    let manager = sinc_manager(conv, 96.0, false);
    let sample_num = conv.sample_num_10ms();
    bencher.bench_local(move || {
        let iter = (0..).map(|x| x as f64);
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

#[divan::bench(
    name="2. proc a120 10ms",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a120_10ms(bencher: divan::Bencher, conv: &Conv) {
    let manager = sinc_manager(conv, 120.0, false);
    let sample_num = conv.sample_num_10ms();
    bencher.bench_local(move || {
        let iter = (0..).map(|x| x as f64);
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

#[divan::bench(
    name="3. proc a144 10ms",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a144_10ms(bencher: divan::Bencher, conv: &Conv) {
    let manager = sinc_manager(conv, 144.0, false);
    let sample_num = conv.sample_num_10ms();
    bencher.bench_local(move || {
        let iter = (0..).map(|x| x as f64);
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

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

#[divan::bench(
    name = "1. proc a96 10ms batch",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count = 1000,
)]
fn proc_a96_10ms_batch(bencher: divan::Bencher, conv: &Conv) {
    let m = sinc_manager(conv, 96.0, false);
    bencher.bench_local(move || sinc_batch_throughput(&m, conv));
}

#[divan::bench(
    name = "2. proc a120 10ms batch",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count = 1000,
)]
fn proc_a120_10ms_batch(bencher: divan::Bencher, conv: &Conv) {
    let m = sinc_manager(conv, 120.0, false);
    bencher.bench_local(move || sinc_batch_throughput(&m, conv));
}

#[divan::bench(
    name = "3. proc a144 10ms batch",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count = 1000,
)]
fn proc_a144_10ms_batch(bencher: divan::Bencher, conv: &Conv) {
    let m = sinc_manager(conv, 144.0, false);
    bencher.bench_local(move || sinc_batch_throughput(&m, conv));
}

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

#[divan::bench(
    name="1. proc a96 10ms fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a96_10ms_fast(bencher: divan::Bencher, conv: &Conv) {
    let manager = sinc_manager(conv, 96.0, true);
    bencher.bench_local(move || sinc_iter_throughput(&manager, conv))
}

#[divan::bench(
    name="1. proc a96 10ms fast batch",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a96_10ms_fast_batch(bencher: divan::Bencher, conv: &Conv) {
    let manager = sinc_manager(conv, 96.0, true);
    bencher.bench_local(move || sinc_batch_throughput(&manager, conv))
}

#[divan::bench(
    name="2. proc a120 10ms fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a120_10ms_fast(bencher: divan::Bencher, conv: &Conv) {
    let manager = sinc_manager(conv, 120.0, true);
    bencher.bench_local(move || sinc_iter_throughput(&manager, conv))
}

#[divan::bench(
    name="2. proc a120 10ms fast batch",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a120_10ms_fast_batch(bencher: divan::Bencher, conv: &Conv) {
    let manager = sinc_manager(conv, 120.0, true);
    bencher.bench_local(move || sinc_batch_throughput(&manager, conv))
}

#[divan::bench(
    name="3. proc a144 10ms fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a144_10ms_fast(bencher: divan::Bencher, conv: &Conv) {
    let manager = sinc_manager(conv, 144.0, true);
    bencher.bench_local(move || sinc_iter_throughput(&manager, conv))
}

#[divan::bench(
    name="3. proc a144 10ms fast batch",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a144_10ms_fast_batch(bencher: divan::Bencher, conv: &Conv) {
    let manager = sinc_manager(conv, 144.0, true);
    bencher.bench_local(move || sinc_batch_throughput(&manager, conv))
}

#[divan::bench(
    name = "0. linear 1s batch 64",
    args = [Conv::C44k48k, Conv::C48k44k, Conv::C96k44k],
    sample_count = 200,
)]
fn linear_1s_batch_64(bencher: divan::Bencher, conv: &Conv) {
    let m = SrcManager::with_ratio(conv.ratio()).unwrap();
    let total_out = conv_total_out(conv);
    let input = input_for(conv.ratio(), total_out);
    bencher.bench_local(move || batch_throughput(&m, &input, total_out, 64, true));
}

#[divan::bench(
    name = "0. cubic 1s batch 64",
    args = [Conv::C44k48k, Conv::C48k44k, Conv::C96k44k],
    sample_count = 200,
)]
fn cubic_1s_batch_64(bencher: divan::Bencher, conv: &Conv) {
    let m = SrcManager::builder()
        .ratio(conv.ratio())
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();
    let total_out = conv_total_out(conv);
    let input = input_for(conv.ratio(), total_out);
    bencher.bench_local(move || batch_throughput(&m, &input, total_out, 64, true));
}

#[divan::bench(
    name = "1. proc a96 10ms batch 64",
    args = [Conv::C44k48k, Conv::C48k44k, Conv::C96k44k],
    sample_count = 500,
)]
fn proc_a96_10ms_batch_64(bencher: divan::Bencher, conv: &Conv) {
    let m = sinc_manager(conv, 96.0, false);
    bencher.bench_local(move || sinc_batch_throughput_q(&m, conv, 64));
}

#[divan::bench(
    name = "1. proc a96 10ms batch 256",
    args = [Conv::C44k48k, Conv::C48k44k, Conv::C96k44k],
    sample_count = 500,
)]
fn proc_a96_10ms_batch_256(bencher: divan::Bencher, conv: &Conv) {
    let m = sinc_manager(conv, 96.0, false);
    bencher.bench_local(move || sinc_batch_throughput_q(&m, conv, 256));
}

#[divan::bench(
    name = "1. proc a96 10ms fast batch 64",
    args = [Conv::C44k48k, Conv::C48k44k, Conv::C96k44k],
    sample_count = 500,
)]
fn proc_a96_10ms_fast_batch_64(bencher: divan::Bencher, conv: &Conv) {
    let m = sinc_manager(conv, 96.0, true);
    bencher.bench_local(move || sinc_batch_throughput_q(&m, conv, 64));
}

#[divan::bench(
    name = "1. proc a96 10ms fast batch 256",
    args = [Conv::C44k48k, Conv::C48k44k, Conv::C96k44k],
    sample_count = 500,
)]
fn proc_a96_10ms_fast_batch_256(bencher: divan::Bencher, conv: &Conv) {
    let m = sinc_manager(conv, 96.0, true);
    bencher.bench_local(move || sinc_batch_throughput_q(&m, conv, 256));
}

#[divan::bench(
    name = "5. sinc generic shape 1s batch",
    args = [Shape::FloatPi, Shape::Generic20000Of19999, Shape::Up16, Shape::Down16],
    sample_count = 100,
)]
fn sinc_generic_shape_1s_batch(bencher: divan::Bencher, shape: &Shape) {
    let m = shape_sinc_manager(shape, false);
    let total_out = SHAPE_TOTAL_OUT;
    let input = input_for(shape.ratio(), total_out);
    bencher.bench_local(move || batch_throughput(&m, &input, total_out, STAGE, true));
}

#[divan::bench(
    name = "5. sinc fast shape 1s batch",
    args = [Shape::Up16, Shape::Down16],
    sample_count = 100,
)]
fn sinc_fast_shape_1s_batch(bencher: divan::Bencher, shape: &Shape) {
    let m = shape_sinc_manager(shape, true);
    let total_out = SHAPE_TOTAL_OUT;
    let input = input_for(shape.ratio(), total_out);
    bencher.bench_local(move || batch_throughput(&m, &input, total_out, STAGE, true));
}

#[divan::bench(
    name = "1. sinc a96 1s convert generic",
    args = [Conv::C44k48k, Conv::C48k44k],
    sample_count = 100,
)]
fn sinc_a96_1s_convert_generic(bencher: divan::Bencher, conv: &Conv) {
    let m = sinc_manager(conv, 96.0, false);
    let total_out = conv_total_out(conv);
    let input = input_for(conv.ratio(), total_out);
    bencher.bench_local(move || {
        let out = m.convert(&input);
        divan::black_box(&out);
    });
}

#[divan::bench(
    name = "1. sinc a96 1s convert fast",
    args = [Conv::C44k48k, Conv::C48k44k],
    sample_count = 100,
)]
fn sinc_a96_1s_convert_fast(bencher: divan::Bencher, conv: &Conv) {
    let m = sinc_manager(conv, 96.0, true);
    let total_out = conv_total_out(conv);
    let input = input_for(conv.ratio(), total_out);
    bencher.bench_local(move || {
        let out = m.convert(&input);
        divan::black_box(&out);
    });
}

#[divan::bench(
    name = "1. sinc a96 1s planar fast",
    args = [Conv::C44k48k, Conv::C48k44k],
    sample_count = 100,
)]
fn sinc_a96_1s_planar_fast(bencher: divan::Bencher, conv: &Conv) {
    let m = sinc_manager(conv, 96.0, true);
    let total_out = conv_total_out(conv);
    let input = input_for(conv.ratio(), total_out);
    let right = input.iter().map(|x| -x).collect::<Vec<f64>>();
    bencher.bench_local(move || planar_throughput(&m, &input, &right, total_out));
}
