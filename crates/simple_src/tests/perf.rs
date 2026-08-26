use std::hint::black_box;

use simple_src::{Convert, SrcManager};

#[test]
#[ignore = "perf only"]
// cargo flamegraph --profile perf --test perf -- --show-output --ignored --exact t4448
fn t4448() {
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

#[test]
#[ignore = "perf only"]
// cargo flamegraph --profile perf --test perf -- --show-output --ignored --exact t4844
fn t4844() {
    let now = std::time::Instant::now();
    let manager = SrcManager::builder()
        .ratio(44100.0 / 48000.0)
        .generic()
        .attenuation(150.0)
        .quantify(2048)
        .trans_width(2050.0 / 22050.0)
        .build()
        .unwrap();
    println!("{:?}", now.elapsed());
    let now = std::time::Instant::now();
    let iter = (0..).map(|x| x as f64);
    for s in manager.converter().process(iter).take(44100) {
        black_box(s);
    }
    println!("{:?}", now.elapsed());
}

#[test]
#[ignore = "perf only"]
// cargo flamegraph --profile perf --test perf -- --show-output --ignored --exact t9644
fn t9644() {
    let now = std::time::Instant::now();
    let manager = SrcManager::builder()
        .ratio(44100.0 / 96000.0)
        .generic()
        .attenuation(150.0)
        .quantify(2048)
        .trans_width(2050.0 / 22050.0)
        .build()
        .unwrap();
    println!("{:?}", now.elapsed());
    let now = std::time::Instant::now();
    let iter = (0..).map(|x| x as f64);
    for s in manager.converter().process(iter).take(44100) {
        black_box(s);
    }
    println!("{:?}", now.elapsed());
}

#[test]
#[ignore = "perf only"]
// cargo flamegraph --profile perf --test perf -- --show-output --ignored --exact t9648
fn t9648() {
    let now = std::time::Instant::now();
    let manager = SrcManager::builder()
        .ratio(48000.0 / 96000.0)
        .generic()
        .attenuation(150.0)
        .quantify(2048)
        .trans_width(4000.0 / 24000.0)
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
