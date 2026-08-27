//! Quick alias comparison: 30 kHz tone, 96 kHz -> 48 kHz (ratio 0.5).
use simple_src::{Kernel, SrcManager};
use std::f64::consts::PI;

fn tone(n: usize, f: f64, sr: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (2.0 * PI * f * i as f64 / sr).sin())
        .collect()
}

fn rms(xs: &[f64]) -> f64 {
    (xs.iter().map(|x| x * x).sum::<f64>() / xs.len() as f64).sqrt()
}

fn main() {
    let input_sr = 96_000.0;
    let tone_hz = 30_000.0;
    let ratio = 0.5;
    let input = tone(8192, tone_hz, input_sr);

    for kernel in [Kernel::Linear, Kernel::Cubic, Kernel::Sinc] {
        let mgr = if kernel == Kernel::Sinc {
            SrcManager::builder()
                .ratio(ratio)
                .kernel(kernel)
                .generic()
                .attenuation(96.0)
                .quantify(128)
                .trans_width(0.1)
                .build()
                .unwrap()
        } else {
            SrcManager::builder()
                .ratio(ratio)
                .kernel(kernel)
                .build()
                .unwrap()
        };
        let out = mgr.convert(&input);
        let body = &out[512..out.len() - 512];
        println!(
            "{kernel:?} down 96k->48k @30kHz: out_rms={:.4} (input_rms={:.4})",
            rms(body),
            rms(&input[512..input.len() - 512])
        );
    }
    println!("(30 kHz is above 24 kHz Nyquist; unfiltered downsample folds energy into passband)");
}
