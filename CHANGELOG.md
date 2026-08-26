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

### Quality ladder

Linear &lt; Cubic &lt; Sinc (default).
