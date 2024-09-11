use simple_src::{sinc::Manager, Convert};

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

impl Conv {
    fn sample_num_10ms(&self) -> usize {
        match self {
            Conv::C48k44k | Conv::C96k44k => 441,
            Conv::C44k48k | Conv::C96k48k => 480,
            _ => 960,
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

#[divan::bench(args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k])]
fn init_a120(conv: &Conv) -> Manager {
    match conv {
        Conv::C44k48k => Manager::new(R44K48K, 120.0, 512, TRANS44K),
        Conv::C44k96k => Manager::new(R44K96K, 120.0, 512, TRANS44K),
        Conv::C48k44k => Manager::new(R48K44K, 120.0, 512, TRANS44K),
        Conv::C48k96k => Manager::new(R48K96K, 120.0, 512, TRANS48K),
        Conv::C96k44k => Manager::new(R96K44K, 120.0, 512, TRANS44K),
        Conv::C96k48k => Manager::new(R96K48K, 120.0, 512, TRANS48K),
    }
}

#[divan::bench(
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

#[divan::bench(args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k])]
fn init_a150(conv: &Conv) -> Manager {
    match conv {
        Conv::C44k48k => Manager::new(R44K48K, 150.0, 2048, TRANS44K),
        Conv::C44k96k => Manager::new(R44K96K, 150.0, 2048, TRANS44K),
        Conv::C48k44k => Manager::new(R48K44K, 150.0, 2048, TRANS44K),
        Conv::C48k96k => Manager::new(R48K96K, 150.0, 2048, TRANS48K),
        Conv::C96k44k => Manager::new(R96K44K, 150.0, 2048, TRANS44K),
        Conv::C96k48k => Manager::new(R96K48K, 150.0, 2048, TRANS48K),
    }
}

#[divan::bench(
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
    sample_count=1000,
)]
fn proc_a150_10ms(bencher: divan::Bencher, conv: &Conv) {
    let manager = init_a150(conv);
    let sample_num = conv.sample_num_10ms();
    bencher.bench_local(move || {
        let iter = (0..).map(|x| x as f64).into_iter();
        for s in manager.converter().process(iter).take(sample_num) {
            divan::black_box(s);
        }
    })
}
