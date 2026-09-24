//! Profiling driver for x86 perf analysis (not committed; local tooling).
//!
//! Usage: perf_x86 <fast|generic|cubic|linear> [seconds]
//! Runs ~N seconds of audio through one conversion and prints throughput.
use std::hint::black_box;
use std::time::Instant;

use simple_src::{Convert, Kernel, SrcManager};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("fast");
    let seconds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(120);

    let manager = match path {
        "fast" | "generic" => {
            let b = SrcManager::builder()
                .ratio(48000.0 / 44100.0)
                .attenuation(96.0)
                .trans_width(0.05);
            if path == "fast" {
                b.fast().build().unwrap()
            } else {
                // quantify 128 matches the bench suite's a96 generic config
                // (129-row LUT ≈ 271 KB, L2-resident like the bench numbers).
                b.generic().quantify(128).build().unwrap()
            }
        }
        "cubic" => SrcManager::builder()
            .ratio(48000.0 / 44100.0)
            .kernel(Kernel::Cubic)
            .build()
            .unwrap(),
        _ => SrcManager::with_ratio(48000.0 / 44100.0).unwrap(),
    };

    // 10 ms output chunks through the batch (streaming) path — the
    // real-time shape of the library.
    let out_per_chunk = 480;
    let in_per_chunk = 441;
    let chunks = seconds * 100;
    let input: Vec<f64> = (0..in_per_chunk)
        .map(|i| ((i as f64) * 0.013).sin())
        .collect();
    let mut buf = vec![0.0f64; out_per_chunk];
    let mut cv = manager.converter();

    let now = Instant::now();
    let mut produced: u64 = 0;
    for _ in 0..chunks {
        let mut pos = 0;
        while pos < input.len() {
            let (consumed, n) = cv.process_block(&input[pos..], &mut buf);
            if consumed == 0 {
                break;
            }
            pos += consumed;
            produced += n as u64;
            black_box(&buf[..n]);
        }
    }
    let elapsed = now.elapsed();
    eprintln!(
        "{path}: {produced} samples in {elapsed:?} -> {:.2} Msamp/s",
        produced as f64 / elapsed.as_secs_f64() / 1e6
    );
}
