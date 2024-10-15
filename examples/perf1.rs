use std::hint::black_box;

use simple_src::{sinc, Convert};

// cargo run --profile perf --example perf1
// cargo flamegraph --profile perf --example perf1
fn main() {
    let now = std::time::Instant::now();
    let manager = sinc::Manager::new(48000.0 / 44100.0, 150.0, 2048, 2050.0 / 22050.0).unwrap();
    println!("{:?}", now.elapsed());
    let now = std::time::Instant::now();
    let iter = (0..).map(|x| x as f64).into_iter();
    for s in manager.converter().process(iter).take(48000) {
        black_box(s);
    }
    println!("{:?}", now.elapsed());
}
