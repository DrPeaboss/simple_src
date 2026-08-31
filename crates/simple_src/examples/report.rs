//! Quality report generator — standalone entry point.
//!
//! Recomputes every quality baseline in memory (full-density spectra, so the
//! overlay charts do not suffer CSV downsampling), collects the metrics into
//! summary tables, renders comparison charts, and assembles one
//! self-contained `report.html`:
//!
//! ```text
//! $CARGO_TARGET_TMPDIR/quality/report.html          (with all artifacts)
//! <repo>/output/report/index.html                   (local copy)
//! ```
//!
//! Run with:
//! ```text
//! cargo run -p simple_src --example report
//! ```
//!
//! The report is meant to be browsed as a single tab: tables up front, every
//! chart folded into a `<details>` block (click to expand), with anchors for
//! the sections. It is a build command, not a test — it shares its analysis
//! module (`tooling/report/`) with the spectral assertion tests but runs
//! independently of them, so `cargo test` does not need to have run first.
//! The trim comparison uses a 6-tone grid to stay fast.

#[path = "../tooling/report/analysis.rs"]
mod analysis;

#[path = "../tooling/report/mod.rs"]
mod report;

fn main() {
    report::generate_report();
}
