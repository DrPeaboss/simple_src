#[cfg(feature = "internal-bench")]
#[path = "common/mod.rs"]
mod common;
#[cfg(feature = "internal-bench")]
use common::*;
#[cfg(feature = "internal-bench")]
use simple_src::{Convert, SrcManager};

fn main() {
    divan::main();
}

/// Forced dot-kernel benches (feature `internal-bench`): measure the portable
/// scalar fallback against the runtime-selected SIMD kernel on the same
/// machine. The SIMD-forced entries early-return on CPUs without the target
/// feature, so they only measure where the kernel actually runs; the scalar
/// entries run everywhere.
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
            acc += divan::black_box(sink[p - 1]);
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
            acc += divan::black_box(s);
        }
        acc
    }

    #[divan::bench(
        name = "4. forced scalar fast batch",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 500,
    )]
    fn forced_scalar_fast_batch(bencher: divan::Bencher, conv: &Conv) {
        let manager = sinc_manager(conv, 96.0, true);
        bencher.bench_local(move || batch_forced(&manager, conv, true));
    }

    #[divan::bench(
        name = "4. forced simd fast batch",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 500,
    )]
    fn forced_simd_fast_batch(bencher: divan::Bencher, conv: &Conv) {
        if !simd_available() {
            return; // SIMD not present on this CPU: nothing to measure
        }
        let manager = sinc_manager(conv, 96.0, true);
        bencher.bench_local(move || batch_forced(&manager, conv, false));
    }

    #[divan::bench(
        name = "4. forced scalar fast iter",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 500,
    )]
    fn forced_scalar_fast_iter(bencher: divan::Bencher, conv: &Conv) {
        let manager = sinc_manager(conv, 96.0, true);
        bencher.bench_local(move || iter_forced(&manager, conv, true));
    }

    #[divan::bench(
        name = "4. forced simd fast iter",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 500,
    )]
    fn forced_simd_fast_iter(bencher: divan::Bencher, conv: &Conv) {
        if !simd_available() {
            return;
        }
        let manager = sinc_manager(conv, 96.0, true);
        bencher.bench_local(move || iter_forced(&manager, conv, false));
    }

    #[divan::bench(
        name = "4. forced scalar generic batch",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 200,
    )]
    fn forced_scalar_generic_batch(bencher: divan::Bencher, conv: &Conv) {
        let manager = sinc_manager(conv, 96.0, false);
        bencher.bench_local(move || batch_forced(&manager, conv, true));
    }

    #[divan::bench(
        name = "4. forced simd generic batch",
        args = [Conv::C44k48k, Conv::C48k44k],
        sample_count = 200,
    )]
    fn forced_simd_generic_batch(bencher: divan::Bencher, conv: &Conv) {
        if !simd_available() {
            return;
        }
        let manager = sinc_manager(conv, 96.0, false);
        bencher.bench_local(move || batch_forced(&manager, conv, false));
    }
}
