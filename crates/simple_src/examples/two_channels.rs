use std::f64::consts::PI;

use simple_src::sinc;

const SOURCE_FILE: &str = "two_channels_44k.wav";
const TARGET_FILE: &str = "two_channels_44k_48k.wav";

fn generate_source_file() {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(SOURCE_FILE, spec).unwrap();
    for t in (0..44100).map(|x| x as f64 / 44100.0) {
        let sample = (t * 440.0 * 2.0 * PI).sin();
        let amplitude = i16::MAX as f64;
        let sample_to_write = (sample * amplitude) as i16;
        writer.write_sample(sample_to_write).unwrap();
        writer.write_sample(sample_to_write).unwrap();
    }
    writer.finalize().unwrap();
}

fn convert_to_48k() {
    let mut reader = hound::WavReader::open(SOURCE_FILE).unwrap();
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(TARGET_FILE, spec).unwrap();
    let manager = sinc::Manager::fast_with_sample_rate_quality(
        44100,
        48000,
        simple_src::Quality::Bit16Medium,
        20000,
    )
    .unwrap();

    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut samples = reader
        .samples::<i16>()
        .map(|x| x.unwrap() as f64 / i16::MAX as f64);
    while let (Some(l), Some(r)) = (samples.next(), samples.next()) {
        left.push(l);
        right.push(r);
    }

    let out_l = manager.convert(&left);
    let out_r = manager.convert(&right);
    for (s1, s2) in out_l.into_iter().zip(out_r) {
        writer.write_sample((s1 * i16::MAX as f64) as i16).unwrap();
        writer.write_sample((s2 * i16::MAX as f64) as i16).unwrap();
    }
    writer.finalize().unwrap();
}

// cargo run -r -p simple_src --example two_channels
fn main() {
    let _ = std::fs::create_dir("output");
    std::env::set_current_dir("output").unwrap();
    generate_source_file();
    convert_to_48k();
}
