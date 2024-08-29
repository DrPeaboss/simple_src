use std::f64::consts::PI;

use hound;
use simple_src::{sinc, Convert};

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
    let input_frames = reader.duration() as u64;
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(TARGET_FILE, spec).unwrap();
    let manager = sinc::Manager::new(48000.0 / 44100.0, 110.0, 256, 2050.0 / 22050.0);
    let latency = manager.latency();
    let mut converter1 = manager.converter();
    let mut converter2 = manager.converter();
    let mut samples = reader
        .samples::<i16>()
        .map(|x| x.unwrap() as f64 / i16::MAX as f64)
        .chain(std::iter::repeat(0.0));
    let out_frames = input_frames * 48000 / 44100;
    let mut n = 0;
    while n < out_frames {
        let mut chan1 = Vec::new();
        let mut chan2 = Vec::new();
        for _ in 0..2048 {
            chan1.push(samples.next().unwrap());
            chan2.push(samples.next().unwrap());
        }
        let result1 = converter1.process(chan1.into_iter());
        let result2 = converter2.process(chan2.into_iter());
        let num_to_skip = if n == 0 { latency } else { 0 };
        let num_to_take = (out_frames - n) as usize;
        let mut write_frames = 0;
        for (s1, s2) in result1.zip(result2).skip(num_to_skip).take(num_to_take) {
            writer.write_sample((s1 * i16::MAX as f64) as i16).unwrap();
            writer.write_sample((s2 * i16::MAX as f64) as i16).unwrap();
            write_frames += 1;
        }
        n += write_frames;
    }
    writer.finalize().unwrap();
}

// cargo run -r --example two_channels
fn main() {
    let _ = std::fs::create_dir("output");
    std::env::set_current_dir("output").unwrap();
    generate_source_file();
    convert_to_48k();
}
