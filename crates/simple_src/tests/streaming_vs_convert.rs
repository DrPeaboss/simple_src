//! P1c: chunked streaming must reproduce `convert()` sample-for-sample
//! across every kernel and ratio shape (rational fast/generic, float-phase
//! irrational, integer rates), and chunk sizes must be transparent.
//!
//! `convert()` pads the tail with zeros and drops the latency-pending
//! prefix; a streaming caller that runs `process_block` over chunks and then
//! flushes until 0 reaches the same state machine, so the first
//! `convert().len()` samples after the latency skip must match exactly.

use simple_src::{Convert, Kernel, Quality, SrcManager};

fn manager_cases() -> Vec<(String, SrcManager)> {
    let mut out = Vec::new();
    for ratio in [0.5_f64, 2.0] {
        let out_label = format!("linear@{ratio}");
        out.push((
            out_label,
            SrcManager::builder()
                .ratio(ratio)
                .kernel(Kernel::Linear)
                .build()
                .unwrap(),
        ));
        out.push((
            format!("cubic@{ratio}"),
            SrcManager::builder()
                .ratio(ratio)
                .kernel(Kernel::Cubic)
                .build()
                .unwrap(),
        ));
        let g = SrcManager::builder()
            .ratio(ratio)
            .generic()
            .attenuation(96.0)
            .quantify(128)
            .trans_width(0.1);
        out.push((format!("sinc_g@{ratio}"), g.build().unwrap()));
        let f = SrcManager::builder()
            .ratio(ratio)
            .fast()
            .attenuation(96.0)
            .trans_width(0.1);
        out.push((format!("sinc_f@{ratio}"), f.build().unwrap()));
    }
    // Real integer rate pair on the fast path.
    out.push((
        "sinc_f@44100/48000".into(),
        SrcManager::builder()
            .sample_rate(44_100, 48_000)
            .quality(Quality::Bit16Fast)
            .pass_freq(20_000)
            .fast()
            .build()
            .unwrap(),
    ));
    // Float-phase (irrational) paths.
    out.push((
        "sinc_g@pi".into(),
        SrcManager::builder()
            .ratio(std::f64::consts::PI)
            .generic()
            .attenuation(96.0)
            .quantify(128)
            .trans_width(0.1)
            .build()
            .unwrap(),
    ));
    out.push((
        "linear@float".into(),
        SrcManager::with_ratio(1.23456789).unwrap(),
    ));
    out
}

fn stream_with_chunk(manager: &SrcManager, input: &[f64], chunk: usize) -> Vec<f64> {
    let mut cv = manager.converter();
    let mut body = Vec::new();
    let mut pos = 0;
    let mut small = vec![0.0; chunk];
    while pos < input.len() {
        let (c, p) = cv.process_block(&input[pos..], &mut small);
        if c == 0 && p == 0 {
            break;
        }
        pos += c;
        body.extend_from_slice(&small[..p]);
    }
    let mut tail = vec![0.0; (chunk / 2).max(3)];
    loop {
        let n = cv.flush(&mut tail);
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tail[..n]);
    }
    body
}

fn assert_matches_body(label: &str, manager: &SrcManager, input: &[f64], body: &[f64]) {
    let whole = manager.convert(input);
    let lat = manager.latency();
    assert!(
        body.len() >= lat + whole.len(),
        "{label}: stream len {} < latency {lat} + convert len {}",
        body.len(),
        whole.len()
    );
    for (i, (a, b)) in body
        .iter()
        .skip(lat)
        .take(whole.len())
        .zip(whole.iter())
        .enumerate()
    {
        assert!((a - b).abs() < 1e-9, "{label}: sample {i} {a} vs {b}");
    }
}

#[test]
fn chunked_streaming_matches_convert_for_all_kernels_and_ratios() {
    let input: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.011).sin()).collect();
    for (label, m) in manager_cases() {
        let body = stream_with_chunk(&m, &input, 7);
        assert_matches_body(&label, &m, &input, &body);
    }
}

#[test]
fn chunk_sizes_are_transparent() {
    let input: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.017).cos()).collect();
    for (label, m) in manager_cases() {
        // Tiny, typical, and large chunks must each reproduce convert().
        // (Trailing pad-zero counts may differ with the last stop point, so
        // the contract is convert-prefix equality, not equal total length.)
        for &c in &[1usize, 5, 16, 257] {
            let body = stream_with_chunk(&m, &input, c);
            assert_matches_body(&label, &m, &input, &body);
        }
    }
}
