use std::f64::consts::TAU;
use wavers::write as write_wav;

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
    let mut osc = Osc::init(sample_rate as f64);
    osc.set_freq(1000.0);
    let sample_count = sample_rate * 5;
    let samples: Vec<f32> = (0..sample_count)
        .map(|_| (osc.next() * 0.99) as f32)
        .collect();
    write_wav(filename, &samples, sample_rate as i32, 1).unwrap();
}

fn gen_sweep(sample_rate: u32) {
    let filename = format!("sweep_{}k.wav", sample_rate / 1000);
    let mut osc = Osc::init(sample_rate as f64);
    let sample_count = sample_rate * 5;
    let nyquist_freq = sample_rate as f64 / 2.0;
    let samples: Vec<f32> = (0..sample_count)
        .map(|i| {
            osc.set_freq(nyquist_freq * (i as f64 / sample_count as f64).powi(2));
            (osc.next() * 0.99) as f32
        })
        .collect();
    write_wav(filename, &samples, sample_rate as i32, 1).unwrap();
}

#[test]
#[ignore = "generate files"]
fn generate() {
    let _ = std::fs::create_dir("output");
    std::env::set_current_dir("output").unwrap();
    gen_beep(44100);
    gen_beep(48000);
    gen_beep(96000);
    gen_beep(192000);
    gen_sweep(44100);
    gen_sweep(48000);
    gen_sweep(96000);
    gen_sweep(192000);
}
