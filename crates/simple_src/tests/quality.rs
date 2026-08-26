use std::f64::consts::PI;

use simple_src::{Convert, ConvertMode, Quality, SrcManager};

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn rms(xs: &[f64]) -> f64 {
    (xs.iter().map(|x| x * x).sum::<f64>() / xs.len() as f64).sqrt()
}

fn db(x: f64) -> f64 {
    20.0 * x.abs().max(1e-20).log10()
}

fn tone(n: usize, freq: f64, sr: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
        .collect()
}

fn steady_body(out: &[f64], skip: usize) -> &[f64] {
    let s = skip.min(out.len() / 4);
    let e = out.len().saturating_sub(s).max(s + 1);
    &out[s..e]
}

/// Absolute DC gain error (linear amplitude, not dB).
fn dc_gain_error(src: &SrcManager, n: usize) -> f64 {
    let out = src.convert(&vec![1.0; n]);
    let body = steady_body(&out, src.latency().max(64));
    (mean(body) - 1.0).abs()
}

fn tone_gain_db(src: &SrcManager, freq: f64, sr_in: f64, n_in: usize) -> f64 {
    let input = tone(n_in, freq, sr_in);
    let out = src.convert(&input);
    let skip = src.latency().max(512);
    let in_body = &input[skip.min(input.len() / 4)..input.len().saturating_sub(skip).max(skip + 1)];
    let out_body = steady_body(&out, skip);
    db(rms(out_body) / rms(in_body).max(1e-20))
}

#[test]
fn linear_dc_gain() {
    let src = SrcManager::with_ratio(2.0).unwrap();
    let input = vec![1.0; 64];
    let output = src.convert(&input);
    assert_eq!(output.len(), 128);
    let body = &output[8..output.len() - 8];
    let avg = mean(body);
    assert!((avg - 1.0).abs() < 1e-9, "dc gain {avg}");
}

#[test]
fn cubic_dc_gain() {
    let src = simple_src::SrcManager::builder()
        .ratio(2.0)
        .kernel(simple_src::Kernel::Cubic)
        .build()
        .unwrap();
    let err = dc_gain_error(&src, 1024);
    assert!(err < 1e-6, "cubic dc gain err={err}");
}

#[test]
fn sinc_dc_gain_high_and_low() {
    // High tier: < 0.001 dB ≈ 1.15e-4 linear.
    for (ratio, quality, tw) in [
        (2.0, Quality::Bit16Fast, 0.1),
        (0.5, Quality::Bit16Fast, 0.1),
        (48000.0 / 44100.0, Quality::Bit16Better, 0.1),
    ] {
        let src = SrcManager::builder()
            .ratio(ratio)
            .generic()
            .quality(quality)
            .trans_width(tw)
            .build()
            .unwrap();
        let err = dc_gain_error(&src, 1024);
        assert!(
            err < 1.15e-4,
            "high-tier dc ratio={ratio} quality={quality:?} err={err}"
        );
    }
    // Low tier: < 0.05 dB ≈ 5.8e-3 linear.
    for ratio in [2.0, 0.5] {
        let src = SrcManager::builder()
            .ratio(ratio)
            .generic()
            .quality(Quality::Bit8Fast)
            .trans_width(0.2)
            .build()
            .unwrap();
        let err = dc_gain_error(&src, 1024);
        assert!(err < 5.8e-3, "low-tier dc ratio={ratio} err={err}");
    }
}

#[test]
fn sinc_impulse_latency_within_one_sample() {
    for (ratio, quality, tw) in [
        (1.0, Quality::Bit8Better, 0.1),
        (2.0, Quality::Bit16Fast, 0.1),
        (0.5, Quality::Bit16Fast, 0.1),
    ] {
        let src = SrcManager::builder()
            .ratio(ratio)
            .generic()
            .quality(quality)
            .trans_width(tw)
            .build()
            .unwrap();
        let mut input = vec![0.0; 256];
        input[0] = 1.0;
        let mut cv = src.converter();
        let raw: Vec<f64> = cv
            .process(input.iter().copied().chain(std::iter::repeat(0.0)))
            .take(src.latency() + 32)
            .collect();
        let peak = raw
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let latency = src.latency();
        // Unity ratio stays within one sample; non-integer phase rounding can
        // shift the absolute peak by an extra sample when ratio ≠ 1.
        let tol = if (ratio - 1.0).abs() < 1e-12 { 1 } else { 2 };
        assert!(
            peak.abs_diff(latency) <= tol,
            "ratio={ratio} peak={peak} latency={latency} tol={tol}"
        );
        let compensated = src.convert(&input);
        let peak2 = compensated
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            peak2 <= tol,
            "convert() should drop latency, ratio={ratio} peak={peak2} tol={tol}"
        );
    }
}

#[test]
fn sample_rate_44100_48000_is_fast_rational() {
    let up = SrcManager::builder()
        .sample_rate(44100, 48000)
        .fast()
        .quality(Quality::Bit16Fast)
        .pass_freq(20000)
        .build()
        .unwrap();
    assert_eq!(up.mode(), ConvertMode::RationalFast);
    assert_eq!(up.ratio_parts(), Some((160, 147)));

    let down = SrcManager::builder()
        .sample_rate(48000, 44100)
        .fast()
        .quality(Quality::Bit16Fast)
        .pass_freq(20000)
        .build()
        .unwrap();
    assert_eq!(down.mode(), ConvertMode::RationalFast);
    assert_eq!(down.ratio_parts(), Some((147, 160)));
}

#[test]
fn float_ratio_pi_uses_float_phase() {
    let src = SrcManager::builder()
        .ratio(PI)
        .generic()
        .quality(Quality::Bit8Fast)
        .trans_width(0.2)
        .build()
        .unwrap();
    assert_eq!(src.mode(), ConvertMode::Float);
    assert_eq!(src.ratio_parts(), None);
    // Must complete without allocating a huge rational coef table.
    let out = src.convert(&[1.0, 0.0, -1.0, 0.0, 1.0, 0.0]);
    assert_eq!(out.len(), src.output_len(6));
}

#[test]
fn process_block_roundtrip_length() {
    let src = SrcManager::builder()
        .ratio(2.0)
        .generic()
        .quality(Quality::Bit8Fast)
        .trans_width(0.2)
        .build()
        .unwrap();
    let input: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let mut cv = src.converter();
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
    let skipped: Vec<_> = collected.into_iter().skip(src.latency()).collect();
    assert!(skipped.len() >= src.output_len(input.len()));
}

#[test]
fn flush_then_convert_length_still_matches() {
    let src = SrcManager::builder()
        .ratio(2.0)
        .generic()
        .quality(Quality::Bit8Better)
        .trans_width(0.1)
        .build()
        .unwrap();
    let input: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
    let via_convert = src.convert(&input);
    let mut cv = src.converter();
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
    let skipped: Vec<_> = collected.into_iter().skip(src.latency()).collect();
    assert!(
        skipped.len() + 2 >= via_convert.len(),
        "stream {} convert {}",
        skipped.len(),
        via_convert.len()
    );
}

#[test]
fn passband_gain_48k_to_44k1() {
    let quality = Quality::Bit16Better;
    let src = SrcManager::builder()
        .sample_rate(48000, 44100)
        .fast()
        .quality(quality)
        .pass_freq(20000)
        .build()
        .unwrap();
    let n = 48000;
    for freq in [1000.0, 18000.0] {
        let g = tone_gain_db(&src, freq, 48000.0, n);
        assert!(
            g.abs() < 0.05,
            "passband {freq} Hz gain {g} dB (want |g|<0.05)"
        );
    }
}

#[test]
fn stopband_attenuation_48k_to_44k1() {
    let quality = Quality::Bit16Better;
    let atten = quality.attenuation();
    let src = SrcManager::builder()
        .sample_rate(48000, 44100)
        .fast()
        .quality(quality)
        .pass_freq(20000)
        .build()
        .unwrap();
    // Well above the stop edge (halfway from pass to output Nyquist under P1-6).
    let g = tone_gain_db(&src, 23500.0, 48000.0, 48000);
    let limit = -(atten - 3.0);
    assert!(
        g <= limit,
        "stopband 23.5 kHz gain {g} dB, want ≤ {limit} (A={atten})"
    );
}

#[test]
fn generic_matches_fast_bit16() {
    let quality = Quality::Bit16Fast;
    let tw = 0.1;
    let ratio = 2.0;
    let generic = SrcManager::builder()
        .ratio(ratio)
        .generic()
        .quality(quality)
        .trans_width(tw)
        .build()
        .unwrap();
    let fast = SrcManager::builder()
        .ratio(ratio)
        .fast()
        .quality(quality)
        .trans_width(tw)
        .build()
        .unwrap();
    assert_eq!(generic.order(), fast.order());
    let input: Vec<f64> = (0..512)
        .map(|i| (2.0 * PI * i as f64 / 37.0).sin())
        .collect();
    let og = generic.convert(&input);
    let of = fast.convert(&input);
    let n = og.len().min(of.len());
    let mut max_err = 0.0_f64;
    let mut sum_sq = 0.0;
    for i in 0..n {
        let e = (og[i] - of[i]).abs();
        max_err = max_err.max(e);
        sum_sq += e * e;
    }
    let rms_err = (sum_sq / n as f64).sqrt();
    assert!(
        db(rms_err) < -90.0,
        "generic vs fast rms_err={rms_err} ({:.1} dB), max={max_err}",
        db(rms_err)
    );
}
