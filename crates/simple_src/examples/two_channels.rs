use std::f64::consts::PI;

use simple_src::SrcManager;

const SOURCE_FILE: &str = "two_channels_44k.wav";
const TARGET_FILE: &str = "two_channels_44k_48k.wav";

fn generate_source_file() {
    let mut samples = Vec::with_capacity(44100 * 2);
    for t in (0..44100).map(|x| x as f64 / 44100.0) {
        let sample = (t * 440.0 * 2.0 * PI).sin();
        let amplitude = i16::MAX as f64;
        let sample_to_write = (sample * amplitude) as i16;
        samples.push(sample_to_write);
        samples.push(sample_to_write);
    }
    wavers::write(SOURCE_FILE, &samples, 44100, 2).unwrap();
}

fn convert_to_48k() {
    let manager = SrcManager::builder()
        .sample_rate(44100, 48000)
        .fast()
        .quality(simple_src::Quality::Bit16Medium)
        .pass_freq(20000)
        .build()
        .unwrap();

    let (samples, source_sr) = wavers::read::<i16, _>(SOURCE_FILE).unwrap();
    assert_eq!(source_sr, 44100);

    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut iter = samples.iter().map(|&x| x as f64 / i16::MAX as f64);
    while let (Some(l), Some(r)) = (iter.next(), iter.next()) {
        left.push(l);
        right.push(r);
    }

    let out_l = manager.convert(&left);
    let out_r = manager.convert(&right);
    let mut out = Vec::with_capacity(out_l.len() * 2);
    for (s1, s2) in out_l.into_iter().zip(out_r) {
        out.push((s1 * i16::MAX as f64) as i16);
        out.push((s2 * i16::MAX as f64) as i16);
    }
    wavers::write(TARGET_FILE, &out, 48000, 2).unwrap();
}

// cargo run -r -p simple_src --example two_channels
fn main() {
    let _ = std::fs::create_dir("output");
    std::env::set_current_dir("output").unwrap();
    generate_source_file();
    convert_to_48k();
}
