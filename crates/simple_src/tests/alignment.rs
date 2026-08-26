use simple_src::{Convert, Kernel, SrcManager};

fn stream_samples(manager: &SrcManager, input: &[f64], count: usize) -> Vec<f64> {
    let mut converter = manager.converter();
    let mut iter = input.iter().copied();
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        match converter.next_sample(&mut iter) {
            Some(sample) => out.push(sample),
            None => break,
        }
    }
    out
}

#[test]
fn linear_rational_upsample_first_outputs() {
    let manager = SrcManager::with_ratio(2.0).unwrap();
    let input: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let out = stream_samples(&manager, &input, 8);
    let expected = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
    assert_eq!(out.len(), expected.len());
    for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-12, "out[{i}]={got} expected {want}");
    }
}

#[test]
fn linear_rational_sample_rate_first_output_is_input() {
    let manager = SrcManager::builder()
        .sample_rate(44_100, 48_000)
        .kernel(Kernel::Linear)
        .build()
        .unwrap();
    let out = stream_samples(&manager, &[7.0, 1.0, 2.0, 3.0], 1);
    assert_eq!(out.len(), 1);
    assert!((out[0] - 7.0).abs() < 1e-12, "first output={}", out[0]);
}

#[test]
fn linear_float_and_rational_ratio_two_agree() {
    let rational = SrcManager::with_ratio(2.0).unwrap();
    let float = SrcManager::builder()
        .ratio(2.0)
        .kernel(Kernel::Linear)
        .build()
        .unwrap();
    assert_eq!(rational.mode(), simple_src::ConvertMode::RationalFast);
    assert_eq!(float.mode(), simple_src::ConvertMode::RationalFast);

    let input: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let out_r = stream_samples(&rational, &input, 12);
    let out_f = stream_samples(&float, &input, 12);
    assert_eq!(out_r.len(), out_f.len());
    for (i, (a, b)) in out_r.iter().zip(out_f.iter()).enumerate() {
        assert!((a - b).abs() < 1e-12, "mismatch at {i}: {a} vs {b}");
    }
}

#[test]
fn cubic_first_output_matches_input_start() {
    for ratio in [2.0, std::f64::consts::PI, 48_000.0 / 44_100.0] {
        let manager = SrcManager::builder()
            .ratio(ratio)
            .kernel(Kernel::Cubic)
            .build()
            .unwrap();
        let input: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let out = stream_samples(&manager, &input, 1);
        assert_eq!(out.len(), 1);
        assert!(
            (out[0] - input[0]).abs() < 1e-12,
            "ratio={ratio} first output={}",
            out[0]
        );
    }
}

#[test]
fn cubic_rational_upsample_first_outputs() {
    let manager = SrcManager::builder()
        .ratio(2.0)
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();
    let input: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let out = stream_samples(&manager, &input, 4);
    let expected = [0.0, 0.4375, 1.0, 1.5];
    assert_eq!(out.len(), expected.len());
    for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-12, "out[{i}]={got} expected {want}");
    }
}

#[test]
fn cubic_chunked_matches_continuous_after_alignment_fix() {
    let manager = SrcManager::builder()
        .ratio(2.0)
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let continuous = collect_cubic(&manager, &input);
    let chunked = collect_cubic_chunks(&manager, &[&input[..5], &input[5..]]);

    assert_eq!(continuous.len(), chunked.len());
    for (i, (a, b)) in continuous.iter().zip(chunked.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "chunked resume mismatch at {i}: {a} vs {b}"
        );
    }
}

fn collect_cubic(manager: &SrcManager, input: &[f64]) -> Vec<f64> {
    let mut converter = manager.converter();
    let mut out = Vec::new();
    let mut iter = input.iter().copied();
    while let Some(sample) = converter.next_sample(&mut iter) {
        out.push(sample);
    }
    drain_flush(&mut converter, &mut out);
    out
}

fn collect_cubic_chunks(manager: &SrcManager, chunks: &[&[f64]]) -> Vec<f64> {
    let mut converter = manager.converter();
    let mut out = Vec::new();
    for chunk in chunks {
        let mut iter = chunk.iter().copied();
        while let Some(sample) = converter.next_sample(&mut iter) {
            out.push(sample);
        }
    }
    drain_flush(&mut converter, &mut out);
    out
}

fn drain_flush(converter: &mut simple_src::Converter, out: &mut Vec<f64>) {
    let mut tail = [0.0; 16];
    loop {
        let n = converter.flush(&mut tail);
        if n == 0 {
            break;
        }
        out.extend_from_slice(&tail[..n]);
    }
}
