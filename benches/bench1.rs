use simple_src::{linear, sinc, Convert};

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

impl ToString for Conv {
    fn to_string(&self) -> String {
        match self {
            Conv::C44k48k => "44k to 48k".into(),
            Conv::C44k96k => "44k to 96k".into(),
            Conv::C48k44k => "48k to 44k".into(),
            Conv::C48k96k => "48k to 96k".into(),
            Conv::C96k44k => "96k to 44k".into(),
            Conv::C96k48k => "96k to 48k".into(),
        }
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

#[divan::bench(
    name="0. linear 1s",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn linear_1s(bencher: divan::Bencher, conv: &Conv) {
    let manager = linear::Manager::new(conv.ratio()).unwrap();
    let sample_num = conv.sample_num_10ms() * 100;
    bencher.bench_local(move || {
        let iter = (0..).map(|x| x as f64).into_iter();
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

#[divan::bench(
    name="1. init a96",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a96(conv: &Conv) -> sinc::Manager {
    sinc::Manager::new(conv.ratio(), 96.0, 128, conv.trans_width()).unwrap()
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
        let iter = (0..).map(|x| x as f64).into_iter();
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

#[divan::bench(
    name="2. init a120",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a120(conv: &Conv) -> sinc::Manager {
    sinc::Manager::new(conv.ratio(), 120.0, 512, conv.trans_width()).unwrap()
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
        let iter = (0..).map(|x| x as f64).into_iter();
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}

#[divan::bench(
    name="3. init a144",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a144(conv: &Conv) -> sinc::Manager {
    sinc::Manager::new(conv.ratio(), 144.0, 2048, conv.trans_width()).unwrap()
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
        let iter = (0..).map(|x| x as f64).into_iter();
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}
