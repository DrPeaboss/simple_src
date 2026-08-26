use std::hint::black_box;

use simple_src::{Convert, SrcManager};

// cargo run --profile perf --example perf1
// cargo flamegraph --profile perf --example perf1
fn main() {
    let now = std::time::Instant::now();
    let manager = SrcManager::builder()
        .ratio(48000.0 / 44100.0)
        .generic()
        .attenuation(150.0)
        .quantify(2048)
        .trans_width(2050.0 / 22050.0)
        .build()
        .unwrap();
    println!("{:?}", now.elapsed());
    let now = std::time::Instant::now();
    let iter = (0..).map(|x| x as f64);
    for s in manager.converter().process(iter).take(48000) {
        black_box(s);
    }
    println!("{:?}", now.elapsed());
}
