use hound::{WavSpec, WavWriter};
use std::f64::consts::TAU;

struct Osc {
    phase: f64,
    omega: f64,
    freq: f64,
    sample_rate: f64,
}

impl Osc {
    fn init(sample_rate: f64) -> Self {
        Self {
            phase: 0.0,
            omega: 0.0,
            freq: 0.0,
            sample_rate,
        }
    }

    fn set_freq(&mut self, freq: f64) {
        self.freq = freq;
        self.omega = TAU * freq / self.sample_rate;
    }

    fn next(&mut self) -> f64 {
        let sample = self.phase.sin();
        self.phase += self.omega;
        while self.phase >= TAU {
            self.phase -= TAU;
        }
        sample
    }
}

fn gen_beep(sample_rate: u32) {
    let filename = format!("beep_{}k.wav", sample_rate / 1000);
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = WavWriter::create(filename, spec).unwrap();
    let mut osc = Osc::init(sample_rate as f64);
    osc.set_freq(1000.0);
    let sample_count = sample_rate * 5;
    for _ in 0..sample_count {
        let sample = osc.next() * 0.99;
        writer.write_sample(sample as f32).unwrap();
    }
    writer.finalize().unwrap();
}

fn gen_sweep(sample_rate: u32) {
    let filename = format!("sweep_{}k.wav", sample_rate / 1000);
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = WavWriter::create(filename, spec).unwrap();
    let mut osc = Osc::init(sample_rate as f64);
    let sample_count = sample_rate * 5;
    let nyquist_freq = sample_rate as f64 / 2.0;
    for i in 0..sample_count {
        osc.set_freq(nyquist_freq * (i as f64 / sample_count as f64).powi(2));
        let sample = osc.next() * 0.99;
        writer.write_sample(sample as f32).unwrap();
    }
    writer.finalize().unwrap();
}

#[test]
#[ignore = "generate files"]
fn generate() {
    let _ = std::fs::create_dir("output");
    std::env::set_current_dir("output").unwrap();
    gen_beep(44100);
    gen_beep(48000);
    gen_sweep(44100);
    gen_sweep(48000);
    gen_sweep(96000);
}
