# Changelog

## 0.5.0

### Added

- Unified public API: [`SrcManager`](crates/simple_src/src/manager.rs), [`SrcBuilder`](crates/simple_src/src/manager.rs), [`Kernel`](crates/simple_src/src/kernel/mod.rs), [`SincPath`](crates/simple_src/src/kernel/mod.rs).
- [`Kernel::Cubic`](crates/simple_src/src/kernel/mod.rs): Catmull-Rom cubic interpolation (ratio-only, zero latency).
- Internal [`KernelSpec`](crates/simple_src/src/kernel/spec.rs) trait and shared engine loops (`polynomial_next_sample`, `fir_next_sample`).
- CLI `--kernel cubic`.

### Changed

- **Breaking:** Removed legacy `Manager` / per-kernel manager types; use `SrcManager` and `SrcBuilder`.
- **Breaking:** `with_ratio` / `with_sample_rate` are linear-only; sinc and cubic use the builder.
- `Ratio::linear_mode` renamed to `polynomial_mode` (used by linear and cubic).
- Kernels reorganized under `kernel/` (`linear`, `cubic`, `sinc`).
- Fixed a 3-5x linear (and cubic) interpolation regression introduced by the engine
  unification: phase accumulators are monomorphic per conversion mode, linear uses
  dedicated per-mode cores, and `process_block` runs the sample loop below the kernel
  dispatch (batched) instead of re-entering it per sample.
- End-to-end test coverage for the scalar dot-kernel fallback: forced-scalar
  converters now run through the full pipeline and are compared against the
  runtime-selected kernel. A hidden `internal-bench` feature exposes
  `SrcManager::converter_forced_kernel` so benchmarks can measure both kernels;
  AVX2-forced benches skip themselves on CPUs without AVX2+FMA.
- ~9x faster sinc Generic path (float ratios and explicit generic mode): the generic
  table is now stored as `(quan + 1)` polyphase rows instead of a half-sinc table with
  per-tap interpolation, so one output sample is the lerp of two dot products
  (`(1-t) * dot(taps, row[b]) + t * dot(taps, row[b+1])`, an exact algebraic transform;
  results differ only by float reassociation). It reuses the AVX2/FMA dot kernels and a
  dedicated batch loop shared with the Fast path. a96 44100->48000 batch: 196 us -> 21 us
  per 10 ms. `lut_len()` for Generic now reports `(quan + 1) * (order + 1)`.
- ~3x faster sinc Fast (polyphase) path: the LUT is stored flat, the FIR delay line
  feeds contiguous slices, an AVX2+FMA dot-product kernel (runtime-detected once per
  converter, portable auto-vectorized fallback, x86_64) replaces the per-tap zip, and
  `process_block` runs a monomorphic phase loop with no per-sample dispatch. a96
  44100->48000 batch: 31.8 us -> 10.0 us per 10 ms; quality baselines unchanged.
- Benches cover the batch path, `convert`, planar stereo, chunked streaming, ratio
  shapes (float phase, large rational, 16x bounds), and the sinc Fast polyphase path.
- Spectral quality baselines (`tests/spectral`): FFT-based THD+N, max-spur, alias
  rejection, and passband flatness red lines per kernel, with CSV + SVG spectrum
  artifacts and a raw sweep for `plots.py` spectrograms.

### Quality ladder

Linear &lt; Cubic &lt; Sinc (default).
