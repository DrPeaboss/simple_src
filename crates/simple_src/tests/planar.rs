//! Planar multi-channel coverage.
//!
//! P0 hardening. The lockstep helpers must reject every layout mismatch as
//! documented, detect drifted converters, and — most importantly — reproduce
//! the "N independent mono converters" contract exactly for sinc, cubic, and
//! float-phase linear converters.

use simple_src::{Convert, Error, SrcManager, flush_planar, process_planar};

/// Reference mono stream: process the finite input, then flush until the
/// delay line is empty. Identical code path to `convert()` before the
/// latency skip, so it contains the latency-pending prefix. (Never pad with
/// an infinite zero chain here: linear/cubic `next_sample` does not check the
/// empty-delay condition, so an endless iterator would loop forever.)
fn stream_mono(manager: &SrcManager, input: &[f64]) -> Vec<f64> {
    let mut cv = manager.converter();
    let mut out: Vec<f64> = cv.process(input.iter().copied()).collect();
    let mut tail = [0.0; 4096];
    loop {
        let n = cv.flush(&mut tail);
        if n == 0 {
            break;
        }
        out.extend_from_slice(&tail[..n]);
    }
    out
}

/// Lockstep planar stream: one `process_planar` call over the whole input,
/// then `flush_planar` until every converter drains. Returns one buffer per
/// channel.
fn planar_stream(manager: &SrcManager, channels: &[&[f64]]) -> Vec<Vec<f64>> {
    let n = channels.len();
    let ref_len = channels[0].len();
    let mut cvs: Vec<_> = (0..n).map(|_| manager.converter()).collect();
    let mut out: Vec<Vec<f64>> = vec![Vec::new(); n];
    // Big enough for ratio <= 16 plus FIR latency and a flush margin.
    let mut outputs: Vec<Vec<f64>> = vec![vec![0.0; ref_len * 16 + 4096]; n];
    let inputs: Vec<&[f64]> = channels.to_vec();
    let mut obufs: Vec<&mut [f64]> = outputs.iter_mut().map(|v| v.as_mut_slice()).collect();
    let (consumed, produced) = process_planar(&mut cvs, &inputs, &mut obufs).unwrap();
    assert_eq!(consumed, ref_len, "planar should consume the whole input");
    for i in 0..n {
        out[i].extend_from_slice(&outputs[i][..produced]);
    }
    loop {
        let mut tails: Vec<Vec<f64>> = vec![vec![0.0; 512]; n];
        let mut tb: Vec<&mut [f64]> = tails.iter_mut().map(|v| v.as_mut_slice()).collect();
        let p = flush_planar(&mut cvs, &mut tb).unwrap();
        if p == 0 {
            break;
        }
        for i in 0..n {
            out[i].extend_from_slice(&tails[i][..p]);
        }
    }
    out
}

fn assert_streams_equal(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "stream length");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-12, "stream mismatch {x} vs {y}");
    }
}

// ---------------------------------------------------------------------------
// Layout validation
// ---------------------------------------------------------------------------

#[test]
fn planar_rejects_output_length_mismatch() {
    let manager = SrcManager::with_ratio(2.0).unwrap();
    let mut cvs = [manager.converter(), manager.converter()];
    let left = [1.0, 2.0, 3.0, 4.0];
    let right = [4.0, 3.0, 2.0, 1.0];
    let mut out_l = [0.0; 8];
    let mut out_r = [0.0; 4];
    let inputs: [&[f64]; 2] = [&left, &right];
    let mut outputs: [&mut [f64]; 2] = [&mut out_l, &mut out_r];
    let err = process_planar(&mut cvs, &inputs, &mut outputs).unwrap_err();
    assert!(
        matches!(
            err,
            Error::MismatchedLength {
                channel: 1,
                expected: 8,
                actual: 4,
                what: "output"
            }
        ),
        "{err:?}"
    );
}

#[test]
fn flush_planar_rejects_channel_count_mismatch() {
    let manager = SrcManager::with_ratio(2.0).unwrap();
    let mut cvs = [manager.converter(), manager.converter()];
    let mut out_l = [0.0; 4];
    let mut outputs: [&mut [f64]; 1] = [&mut out_l];
    let err = flush_planar(&mut cvs, &mut outputs).unwrap_err();
    assert!(
        matches!(
            err,
            Error::MismatchedChannels {
                expected: 2,
                actual: 1,
                what: "output"
            }
        ),
        "{err:?}"
    );
}

#[test]
fn flush_planar_rejects_output_length_mismatch() {
    let manager = SrcManager::with_ratio(2.0).unwrap();
    let mut cvs = [manager.converter(), manager.converter()];
    let mut out_l = [0.0; 4];
    let mut out_r = [0.0; 8];
    let mut outputs: [&mut [f64]; 2] = [&mut out_l, &mut out_r];
    let err = flush_planar(&mut cvs, &mut outputs).unwrap_err();
    assert!(
        matches!(
            err,
            Error::MismatchedLength {
                channel: 1,
                expected: 4,
                actual: 8,
                what: "output"
            }
        ),
        "{err:?}"
    );
}

#[test]
fn unaligned_converters_are_detected() {
    let manager = SrcManager::with_ratio(2.0).unwrap();
    let mut cvs = [manager.converter(), manager.converter()];
    // Input lengths are equal; channel 1 is pre-drifted by an earlier block
    // so its consume/produce counts differ once the planar call runs.
    let mut tmp = [0.0; 8];
    cvs[1].process_block(&[1.0; 4], &mut tmp);
    let eq = [1.0; 8];
    let mut ol = [0.0; 16];
    let mut os = [0.0; 16];
    let inputs: [&[f64]; 2] = [&eq, &eq];
    let mut outputs: [&mut [f64]; 2] = [&mut ol, &mut os];
    let err = process_planar(&mut cvs, &inputs, &mut outputs).unwrap_err();
    assert!(
        matches!(err, Error::UnalignedConverters { channel: 1 }),
        "{err:?}"
    );
}

// ---------------------------------------------------------------------------
// Multi-channel correctness: planar == N independent mono converters
// ---------------------------------------------------------------------------

#[test]
fn sinc_planar_matches_mono_per_channel() {
    let m = SrcManager::builder()
        .ratio(2.0)
        .generic()
        .attenuation(48.0)
        .quantify(8)
        .trans_width(0.2)
        .build()
        .unwrap();
    let a: Vec<f64> = (0..257).map(|i| (i as f64 * 0.07).sin()).collect();
    let b: Vec<f64> = (0..257).map(|i| (i as f64 * 0.13).cos()).collect();
    let channels = [a.as_slice(), b.as_slice()];
    let mono_a = stream_mono(&m, &a);
    let mono_b = stream_mono(&m, &b);
    let planar = planar_stream(&m, &channels);
    assert_eq!(planar.len(), 2);
    assert_streams_equal(&planar[0], &mono_a);
    assert_streams_equal(&planar[1], &mono_b);
}

#[test]
fn fast_sinc_planar_matches_mono_with_real_rates() {
    let m = SrcManager::builder()
        .sample_rate(44100, 48000)
        .quality(simple_src::Quality::Bit16Fast)
        .pass_freq(20000)
        .fast()
        .build()
        .unwrap();
    let a: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.05).sin()).collect();
    let b: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.11).cos()).collect();
    let channels = [a.as_slice(), b.as_slice()];
    let mono_a = stream_mono(&m, &a);
    let mono_b = stream_mono(&m, &b);
    let planar = planar_stream(&m, &channels);
    assert_streams_equal(&planar[0], &mono_a);
    assert_streams_equal(&planar[1], &mono_b);
}

#[test]
fn float_phase_linear_planar_works() {
    // 1.23456789 is not a bounded rational, so this exercises the float
    // phase accumulator through the planar path.
    let m = SrcManager::with_ratio(1.23456789).unwrap();
    assert_eq!(m.mode(), simple_src::ConvertMode::Float);
    let mut channels = Vec::new();
    for k in 0..3 {
        channels.push(
            (0..200)
                .map(|i| (i as f64 * (0.3 + k as f64 * 0.37)).sin())
                .collect::<Vec<_>>(),
        );
    }
    let refs: Vec<&[f64]> = channels.iter().map(|v| v.as_slice()).collect();
    let planar = planar_stream(&m, &refs);
    for (i, ch) in channels.iter().enumerate() {
        let mono = stream_mono(&m, ch);
        assert_streams_equal(&planar[i], &mono);
    }
}

#[test]
fn cubic_planar_three_channels_lockstep() {
    let m = SrcManager::builder()
        .ratio(1.5)
        .kernel(simple_src::Kernel::Cubic)
        .build()
        .unwrap();
    let mut channels = Vec::new();
    for k in 0..3 {
        channels.push(
            (0..200)
                .map(|i| (i as f64 * (0.2 + k as f64 * 0.5)).sin())
                .collect::<Vec<_>>(),
        );
    }
    let refs: Vec<&[f64]> = channels.iter().map(|v| v.as_slice()).collect();
    let planar = planar_stream(&m, &refs);
    // All three channels report the same lockstep counts, and each channel
    // is bit-identical to its mono stream.
    let counts = planar.iter().map(|v| v.len()).collect::<Vec<_>>();
    assert!(
        counts.windows(2).all(|w| w[0] == w[1]),
        "lockstep lengths {counts:?}"
    );
    for (i, ch) in channels.iter().enumerate() {
        let mono = stream_mono(&m, ch);
        assert_streams_equal(&planar[i], &mono);
    }
}

#[test]
fn planar_drains_flush_in_lockstep_to_zero() {
    // Feed a partial buffer so every channel still holds delay, then flush
    // with a small tail buffer until planar returns 0 — counts must stay
    // identical across channels on every call.
    let m = SrcManager::builder()
        .ratio(2.0)
        .generic()
        .attenuation(48.0)
        .quantify(8)
        .trans_width(0.2)
        .build()
        .unwrap();
    let mut cvs = vec![m.converter(), m.converter(), m.converter()];
    let a = vec![1.0f64; 100];
    let inputs: [&[f64]; 3] = [&a, &a, &a];
    let mut outputs: Vec<Vec<f64>> = vec![vec![0.0; 512]; 3];
    let mut obufs: Vec<&mut [f64]> = outputs.iter_mut().map(|v| v.as_mut_slice()).collect();
    let (_, produced) = process_planar(&mut cvs, &inputs, &mut obufs).unwrap();
    assert!(produced > 0);
    assert_eq!(obufs.len(), 3);
    let mut total = 0;
    loop {
        let mut tails: Vec<Vec<f64>> = vec![vec![0.0; 7]; 3];
        let mut tb: Vec<&mut [f64]> = tails.iter_mut().map(|v| v.as_mut_slice()).collect();
        let p = flush_planar(&mut cvs, &mut tb).unwrap();
        if p == 0 {
            break;
        }
        total += p;
        assert!(p <= 7);
        // every channel filled the same prefix of its tail buffer
        let first = &tails[0][..p];
        for t in tails.iter().take(3).skip(1) {
            assert_eq!(&t[..p], first, "channels must flush in lockstep");
        }
    }
    assert!(total > 0, "expected a drain tail");
}
