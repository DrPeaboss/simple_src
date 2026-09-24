#[cfg(feature = "internal-bench")]
#[path = "common/mod.rs"]
mod common;
// The harness macros and Criterion type must be in scope for the
// criterion_group!/criterion_main! expansions in both feature states.
#[cfg(feature = "internal-bench")]
use common::*;
#[cfg(not(feature = "internal-bench"))]
use criterion::Criterion;
#[cfg(feature = "internal-bench")]
use criterion::{BenchmarkId, Criterion, Throughput};
use criterion::{criterion_group, criterion_main};
#[cfg(feature = "internal-bench")]
use simple_src::{Convert, SrcManager};
#[cfg(feature = "internal-bench")]
use std::hint::black_box;

/// Forced dot-kernel benches (feature `internal-bench`): measure the portable
/// scalar fallback against the runtime-selected SIMD kernel on the same
/// machine. The SIMD entries are only registered when the CPU actually
/// supports the target feature, so they only measure where the kernel runs;
/// the scalar entries run everywhere.
#[cfg(feature = "internal-bench")]
mod forced {
    use super::*;

    fn simd_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        }
        #[cfg(target_arch = "aarch64")]
        {
            std::arch::is_aarch64_feature_detected!("neon")
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    fn batch_forced(m: &SrcManager, conv: &Conv, force_scalar: bool) -> f64 {
        let total_out = conv.sample_num_10ms();
        let input = input_for(conv.ratio(), total_out);
        let mut cv = m.converter_forced_kernel(force_scalar);
        let mut sink = [0.0f64; STAGE];
        let (mut cin, mut produced, mut acc) = (0usize, 0usize, 0.0f64);
        while produced < total_out {
            let fill = STAGE.min(total_out - produced);
            let (c, p) = cv.process_block(&input[cin..], &mut sink[..fill]);
            if p == 0 {
                break;
            }
            cin += c;
            acc += black_box(sink[p - 1]);
            produced += p;
        }
        acc
    }

    fn iter_forced(m: &SrcManager, conv: &Conv, force_scalar: bool) -> f64 {
        let mut acc = 0.0f64;
        let iter = (0..).map(|x| x as f64);
        for s in m
            .converter_forced_kernel(force_scalar)
            .process(iter)
            .take(conv.sample_num_10ms())
        {
            acc += black_box(s);
        }
        acc
    }

    /// One forced bench group: scalar or SIMD kernel, batch or iterator API.
    fn group(c: &mut Criterion, name: &str, generic: bool, scalar: bool, batch: bool) {
        let mut g = c.benchmark_group(name);
        for conv in CONVERT_CONVS {
            g.throughput(Throughput::Elements(conv.sample_num_10ms() as u64));
            g.bench_with_input(BenchmarkId::from_parameter(&conv), &conv, |b, conv| {
                let manager = sinc_manager(conv, 96.0, !generic);
                if batch {
                    b.iter(|| batch_forced(&manager, conv, scalar));
                } else {
                    b.iter(|| iter_forced(&manager, conv, scalar));
                }
            });
        }
        g.finish();
    }

    pub fn register(c: &mut Criterion) {
        group(c, "4. forced scalar fast batch", false, true, true);
        group(c, "4. forced scalar fast iter", false, true, false);
        group(c, "4. forced scalar generic batch", true, true, true);
        if simd_available() {
            group(c, "4. forced simd fast batch", false, false, true);
            group(c, "4. forced simd fast iter", false, false, false);
            group(c, "4. forced simd generic batch", true, false, true);
        }
    }
}

#[cfg(feature = "internal-bench")]
criterion_group! {
    name = benches;
    config = criterion_config();
    targets = forced::register
}

#[cfg(not(feature = "internal-bench"))]
fn noop(_c: &mut Criterion) {}

#[cfg(not(feature = "internal-bench"))]
criterion_group!(benches, noop);

criterion_main!(benches);
