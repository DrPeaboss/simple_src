use std::f64::consts::PI;

use simple_src::{Convert, ConvertMode, Quality, linear, sinc};

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn zero_crossings(xs: &[f64]) -> usize {
    xs.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count()
}

#[test]
fn linear_dc_gain() {
    let manager = linear::Manager::new(2.0).unwrap();
    let input = vec![1.0; 64];
    let output = manager.convert(&input);
    assert_eq!(output.len(), 128);
    let body = &output[8..output.len() - 8];
    let avg = mean(body);
    assert!((avg - 1.0).abs() < 1e-9, "dc gain {avg}");
}

#[test]
fn sinc_dc_gain() {
    let manager = sinc::Manager::with_quality(2.0, Quality::Bit8Better, 0.1).unwrap();
    let input = vec![1.0; 256];
    let output = manager.convert(&input);
    assert_eq!(output.len(), manager.output_len(input.len()));
    let start = manager.latency().max(32);
    let body = &output[start..output.len() - 32];
    let avg = mean(body);
    assert!((avg - 1.0).abs() < 0.05, "dc gain {avg}");
}

#[test]
fn sinc_impulse_latency() {
    let manager = sinc::Manager::with_quality(1.0, Quality::Bit8Better, 0.1).unwrap();
    let mut input = vec![0.0; 128];
    input[0] = 1.0;
    let mut cv = manager.converter();
    let raw: Vec<f64> = cv
        .process(input.iter().copied().chain(std::iter::repeat(0.0)))
        .take(manager.latency() + 16)
        .collect();
    let peak = raw
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let latency = manager.latency();
    assert!(
        peak.abs_diff(latency) <= 2,
        "peak at {peak}, latency {latency}"
    );
    let compensated = manager.convert(&input);
    let peak2 = compensated
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert!(peak2 <= 2, "convert() should drop latency, peak at {peak2}");
}

#[test]
fn sinc_sine_keeps_frequency() {
    let sr_in = 8000.0;
    let sr_out = 16000.0;
    let freq = 500.0;
    let manager = sinc::Manager::with_quality(sr_out / sr_in, Quality::Bit8Better, 0.1).unwrap();
    let n = 800;
    let input: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * freq * i as f64 / sr_in).sin())
        .collect();
    let output = manager.convert(&input);
    let start = manager.latency().max(64);
    let body = &output[start..output.len() - 64];
    let duration = body.len() as f64 / sr_out;
    let crossings = zero_crossings(body) as f64;
    let measured = crossings / duration;
    assert!(
        (measured - freq).abs() / freq < 0.05,
        "measured {measured} Hz, expected {freq} Hz"
    );
}

#[test]
fn sample_rate_uses_rational_fast() {
    let manager =
        sinc::Manager::fast_with_sample_rate_quality(44100, 48000, Quality::Bit16Fast, 20000)
            .unwrap();
    assert_eq!(manager.mode(), ConvertMode::RationalFast);
    assert_eq!(manager.ratio_parts(), Some((160, 147)));
}

#[test]
fn process_block_roundtrip_length() {
    let manager = sinc::Manager::with_quality(2.0, Quality::Bit8Fast, 0.2).unwrap();
    let input: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let mut cv = manager.converter();
    let mut collected = Vec::new();
    let mut pos = 0;
    let mut tmp = [0.0; 32];
    while pos < input.len() {
        let (consumed, produced) = cv.process_block(&input[pos..], &mut tmp);
        if consumed == 0 && produced == 0 {
            break;
        }
        pos += consumed;
        collected.extend_from_slice(&tmp[..produced]);
    }
    let mut tail = [0.0; 256];
    let n = cv.flush(&mut tail);
    collected.extend_from_slice(&tail[..n]);
    let skipped: Vec<_> = collected.into_iter().skip(manager.latency()).collect();
    assert!(skipped.len() >= manager.output_len(input.len()));
}

#[test]
fn float_ratio_pi_does_not_use_rational() {
    let manager = sinc::Manager::with_quality(PI, Quality::Bit8Fast, 0.2).unwrap();
    assert_eq!(manager.mode(), ConvertMode::Float);
    assert_eq!(manager.ratio_parts(), None);
}

#[test]
fn flush_then_convert_length_still_matches() {
    let manager = sinc::Manager::with_quality(2.0, Quality::Bit8Better, 0.1).unwrap();
    let input: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
    let via_convert = manager.convert(&input);
    let mut cv = manager.converter();
    let mut collected = Vec::new();
    let mut pos = 0;
    let mut tmp = [0.0; 16];
    while pos < input.len() {
        let (c, p) = cv.process_block(&input[pos..], &mut tmp);
        if c == 0 && p == 0 {
            break;
        }
        pos += c;
        collected.extend_from_slice(&tmp[..p]);
    }
    let mut tail = [0.0; 4096];
    let n = cv.flush(&mut tail);
    collected.extend_from_slice(&tail[..n]);
    assert!(n < tail.len());
    assert_eq!(cv.flush(&mut tail), 0);
    let skipped: Vec<_> = collected.into_iter().skip(manager.latency()).collect();
    assert!(
        skipped.len() + 2 >= via_convert.len(),
        "stream {} convert {}",
        skipped.len(),
        via_convert.len()
    );
}
