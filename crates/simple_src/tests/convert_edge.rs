//! Edge semantics of the `Convert` trait surface.
//!
//! P0 hardening. Pins the boundary behavior of `next_sample` / `process_block`
//! / `flush`: empty buffers, tiny output buffers, resume-after-`None`, the
//! default (trait-provided) implementations on custom `Convert` types, and the
//! built-in converters' promise that streaming + flush reproduces `convert()`
//! exactly. Also asserts `Send + Sync` on manager and converter, which the
//! threaded streaming usage relies on.

use simple_src::{Convert, SrcManager};

/// A `Convert` type that never consumes input and always produces the next
/// number. Exercises the trait *default* `process_block` / `flush`.
struct Ramped {
    next: f64,
}

impl Convert for Ramped {
    fn next_sample<I>(&mut self, _iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        let v = self.next;
        self.next += 1.0;
        Some(v)
    }
}

/// A `Convert` type that never produces anything. Exercises the default
/// implementations' termination behavior.
struct Never;

impl Convert for Never {
    fn next_sample<I>(&mut self, _iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        None
    }
}

fn linear() -> SrcManager {
    SrcManager::with_ratio(2.0).unwrap()
}

fn small_sinc() -> SrcManager {
    SrcManager::builder()
        .ratio(2.0)
        .generic()
        .attenuation(48.0)
        .quantify(8)
        .trans_width(0.2)
        .build()
        .unwrap()
}

// ---------------------------------------------------------------------------
// Default trait implementations (custom Convert types)
// ---------------------------------------------------------------------------

#[test]
fn default_process_block_never_consumes_input() {
    let mut c = Ramped { next: 0.0 };
    let input = [9.0; 5];
    let mut out = [0.0; 3];
    let (consumed, produced) = c.process_block(&input, &mut out);
    assert_eq!(consumed, 0);
    assert_eq!(produced, 3);
    assert_eq!(out, [0.0, 1.0, 2.0]);
}

#[test]
fn default_flush_fills_the_whole_buffer() {
    // Documented behavior: the default flush does not detect an empty delay
    // line and keeps writing for the whole buffer.
    let mut c = Ramped { next: 0.0 };
    let mut out = [0.0; 7];
    let n = c.flush(&mut out);
    assert_eq!(n, 7);
    assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn never_converter_defaults_produce_nothing() {
    let mut c = Never;
    let mut out = [0.0; 4];
    assert_eq!(c.flush(&mut out), 0);
    let input = [1.0];
    assert_eq!(c.process_block(&input, &mut out), (0, 0));
}

// ---------------------------------------------------------------------------
// Built-in converters: block/flush edge behavior
// ---------------------------------------------------------------------------

#[test]
fn flush_before_any_input_writes_nothing() {
    for m in [linear(), small_sinc()] {
        let mut cv = m.converter();
        let mut out = [0.0; 16];
        assert_eq!(cv.flush(&mut out), 0);
    }
}

#[test]
fn process_block_empty_edges() {
    let m = linear();
    let mut cv = m.converter();
    let mut out = [0.0; 4];
    assert_eq!(cv.process_block(&[], &mut out), (0, 0));
    let input = [1.0, 2.0];
    assert_eq!(cv.process_block(&input, &mut []), (0, 0));
}

#[test]
fn tiny_output_buffer_chunks_match_convert() {
    let m = linear();
    let input: Vec<f64> = (0..64).map(|i| (i as f64 * 0.37).sin()).collect();
    let whole = m.convert(&input);
    let mut cv = m.converter();
    let mut collected = Vec::new();
    let mut pos = 0;
    let mut small = [0.0; 3];
    while pos < input.len() {
        let (c, p) = cv.process_block(&input[pos..], &mut small);
        if c == 0 && p == 0 {
            break;
        }
        pos += c;
        collected.extend_from_slice(&small[..p]);
    }
    let mut tail = [0.0; 128];
    loop {
        let n = cv.flush(&mut tail);
        if n == 0 {
            break;
        }
        collected.extend_from_slice(&tail[..n]);
    }
    assert!(
        collected.len() >= whole.len(),
        "{} vs {}",
        collected.len(),
        whole.len()
    );
    for (a, b) in collected.iter().take(whole.len()).zip(whole.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn next_sample_resumes_after_none() {
    // Chunk boundaries are transparent to the phase state machine: the first
    // chunk emits what it can, and the next chunk continues from the pending
    // interval (here: 6.0), so out1 + out2 is a prefix of the full stream
    // with no samples lost or duplicated. Verified against the probe oracle;
    // the same property is asserted structurally via the prefix match below.
    let m = linear();
    let mut cv = m.converter();
    let mut out1 = Vec::new();
    let mut iter1 = [5.0, 6.0].into_iter();
    while let Some(s) = cv.next_sample(&mut iter1) {
        out1.push(s);
    }
    let mut out2 = Vec::new();
    let mut iter2 = [7.0, 8.0].into_iter();
    while let Some(s) = cv.next_sample(&mut iter2) {
        out2.push(s);
    }
    let expected1 = [5.0, 5.5];
    let expected2 = [6.0, 6.5, 7.0, 7.5];
    assert_eq!(out1.len(), expected1.len());
    for (a, b) in out1.iter().zip(expected1) {
        assert!((a - b).abs() < 1e-12, "chunk1 {a} vs {b}");
    }
    assert_eq!(out2.len(), expected2.len());
    for (a, b) in out2.iter().zip(expected2) {
        assert!((a - b).abs() < 1e-12, "chunk2 {a} vs {b}");
    }
    // out1 + out2 is a prefix of the true full stream (convert pads zeros).
    let whole = linear().convert(&[5.0, 6.0, 7.0, 8.0]);
    let mut joined = out1;
    joined.extend_from_slice(&out2);
    assert!(joined.len() <= whole.len());
    for (a, b) in joined.iter().zip(whole.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn streaming_plus_flush_reproduces_convert_sinc() {
    let m = small_sinc();
    assert!(m.latency() > 0, "sinc must have FIR latency");
    let input: Vec<f64> = (0..64).map(|i| (i as f64 * 0.21).sin()).collect();
    let whole = m.convert(&input);
    let mut cv = m.converter();
    let mut body: Vec<f64> = cv.process(input.iter().copied()).collect();
    let mut tail = [0.0; 4];
    loop {
        let n = cv.flush(&mut tail);
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tail[..n]);
    }
    // The full stream carries the latency-pending prefix and drains zeros
    // past `whole`; the first `whole.len()` samples after the latency skip
    // must match `convert()` sample-for-sample.
    assert!(body.len() >= m.latency() + whole.len());
    for (a, b) in body
        .iter()
        .skip(m.latency())
        .take(whole.len())
        .zip(whole.iter())
    {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn streaming_plus_flush_reproduces_convert_linear() {
    let m = linear();
    let input: Vec<f64> = (0..40).map(|i| (i as f64 * 0.11).cos()).collect();
    let whole = m.convert(&input);
    let mut cv = m.converter();
    let mut body: Vec<f64> = cv.process(input.iter().copied()).collect();
    let mut tail = [0.0; 2];
    loop {
        let n = cv.flush(&mut tail);
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tail[..n]);
    }
    assert!(
        body.len() >= whole.len(),
        "{} vs {}",
        body.len(),
        whole.len()
    );
    for (a, b) in body.iter().take(whole.len()).zip(whole.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn process_called_twice_continues_the_stream() {
    // Streaming continuation across two `process` calls; only the final
    // flush completes the last half-interval and the pad tail. The first
    // `whole.len()` samples must still reproduce `convert()` exactly.
    let m = linear();
    let mut cv = m.converter();
    let first: Vec<f64> = cv.process([1.0, 2.0, 3.0, 4.0].into_iter()).collect();
    let second: Vec<f64> = cv.process([5.0, 6.0, 7.0, 8.0].into_iter()).collect();
    let mut joined = first;
    joined.extend_from_slice(&second);
    let mut tail = [0.0; 8];
    loop {
        let n = cv.flush(&mut tail);
        if n == 0 {
            break;
        }
        joined.extend_from_slice(&tail[..n]);
    }
    let whole = linear().convert(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert!(
        joined.len() >= whole.len(),
        "{} vs {}",
        joined.len(),
        whole.len()
    );
    for (a, b) in joined.iter().take(whole.len()).zip(whole.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Thread-safety contracts
// ---------------------------------------------------------------------------

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn manager_and_converter_are_send_sync() {
    assert_send::<SrcManager>();
    assert_sync::<SrcManager>();
    assert_send::<simple_src::Converter>();
    assert_sync::<simple_src::Converter>();
}
