# simple_src

A simple sample rate conversion lib for audio.

## Usage

Usually use *sinc* Converter, it is flexible and high-quality.
The *linear* Converter is not recommended unless performance is really important
and quality is not cared.

Sinc converters have FIR latency. For a complete buffer, call
`Manager::convert`, which pads zeros and drops the leading delay.
For streaming, skip `manager.latency()` samples at the start and call
`Convert::flush` after the last input until it returns 0. Built-in
sinc/linear converters stop once the delay line is empty (they may still
need more than one call if the flush buffer is short). The trait default
of `Convert::flush` only fills the provided buffer and does not stop on
an empty delay.

Float ratios such as `48000.0 / 44100.0` may be reduced to a rational when
a continued-fraction fit has numerator and denominator ≤ 16384 and relative
error ≤ `1e-12` (so `0.7` becomes `7/10`, while `π` stays float phase).
Prefer `with_sample_rate` / `fast_with_sample_rate` for exact rate pairs.

Multi-channel audio is N independent mono converters. Keep planar buffers
(one slice per channel) and use a converter per channel, or
`process_planar` to keep consume/produce counts aligned. Pass the same
converter count, buffer lengths, and process history on every channel;
mismatches return `Error` instead of panicking.

### sinc

Generic interpolation uses a half Kaiser-sinc table and `quantify`. Fast
polyphase interpolation precomputes one tap set per rational phase; it
requires a rational ratio whose reduced numerator is ≤ 1024, does not take
`quantify`, and returns `Error::FastUnavailable` otherwise.

Typical 44100/48000 conversion:

```rust
use simple_src::{sinc, Quality};

let samples = vec![1.0, 2.0, 3.0, 4.0];
let manager = sinc::Manager::fast_with_sample_rate_quality(
    44100,
    48000,
    Quality::Bit16Fast,
    20000,
).unwrap();
for s in manager.convert(&samples) {
    println!("{s}");
}
```

Generic path with a quality preset (`quantify` is used):

```rust
use simple_src::{sinc, Quality};

let samples = vec![1.0, 2.0, 3.0, 4.0];
let manager = sinc::Manager::with_quality(2.0, Quality::Bit8Fast, 0.1).unwrap();
for s in manager.convert(&samples) {
    println!("{s}");
}
```

Or use builder:

```rust
use simple_src::{sinc, Convert};

let samples = vec![1.0, 2.0, 3.0, 4.0];
let manager = sinc::Manager::builder()
    .ratio(2.0)
    .attenuation(48.0)
    .quantify(8)
    .pass_width(0.9)
    .build()
    .unwrap();
let mut converter = manager.converter();
for s in converter.process(samples.into_iter()) {
    println!("{s}");
}
```

For multi-channel example see [two_channels.rs](/crates/simple_src/examples/two_channels.rs).

### linear

```rust
use simple_src::{linear, Convert};

let samples = vec![1.0, 2.0, 3.0, 4.0];
let manager = linear::Manager::new(2.0).unwrap();
let mut converter = manager.converter();
for s in converter.process(samples.into_iter()) {
    println!("{s}");
}
```

## Sinc parameters

Recommended initialization parameters for *sinc* converter, also available as
[`Quality`](https://docs.rs/simple_src) presets:

|              | attenuation | quantify | `Quality`        |
| ------------ | ----------- | -------- | ---------------- |
| 8bit fast    | 48          | 8        | `Bit8Fast`       |
| 8bit medium  | 60          | 16       | `Bit8Medium`     |
| 8bit better  | 72          | 32       | `Bit8Better`     |
| 16bit lower  | 84          | 64       | `Bit16Lower`     |
| 16bit fast   | 96          | 128      | `Bit16Fast`      |
| 16bit medium | 108         | 256      | `Bit16Medium`    |
| 16bit better | 120         | 512      | `Bit16Better`    |
| 24bit lower  | 132         | 1024     | `Bit24Lower`     |
| 24bit fast   | 144         | 2048     | `Bit24Fast`      |
| 24bit medium | 156         | 4096     | `Bit24Medium`    |
| 24bit better | 168         | 8192     | `Bit24Better`    |

The relationship between *attenuation* and *quantify* is about
*Q = 2 ^ (A / 12 - 1)*, *A = 12 + 12 * log2(Q)*.

Due to the amount of calculation and the size of LUT, A = 144 or 156 for 24bit
audio is usually fine, and for 16bit, A = 120 is enough.

`Quality::attenuation` applies to Generic and Fast; `Quality::quantify` is
Generic-only.

### Filter design notes

- **Order:** From attenuation and transition width, length uses *A + 6 dB*
  and is rounded up to an even order (capped at 2048). Explicit `order` /
  `with_raw` paths skip that margin.
- **Cutoff:** Ideal lowpass cutoff is at
  `min(1, ratio) * (1 - trans_width)`, so the transition sits entirely below
  the applicable Nyquist (output Nyquist when downsampling, input Nyquist when
  upsampling). The −6 dB point is therefore near the pass edge from
  `pass_freq` / `pass_width`; frequencies between that edge and Nyquist are
  already in the transition or stop band. Stop-band attenuation *A* applies
  beyond the stop edge (about halfway from the pass edge to Nyquist), not
  inside the transition.
- **Coefficients:** Fast phases and the Generic half-table are scaled for
  unity DC gain.

## CLI

```
cargo run -p simple_src_cli -- input.wav -r 48000 -o output.wav
```

The CLI uses Fast polyphase interpolation by default. Pass `--generic` (and
`--quantify` if needed) for half-table interpolation.

## Plots

Use [plots.py](/plots.py) to show the results of conversion. It needs *numpy*, *scipy*
and *matplotlib*.

Here is an example showing the results of a downsampling 96kHz:

```
$ cargo test -p simple_src -r --test testwav -- --ignored --exact --show-output generate
$ cargo test -p simple_src -r --test sinc -- --ignored --exact --show-output ta120_2_96k_down
$ python
>>> import plots
>>> import os
>>> os.chdir('output')
>>> plots.spectrum('beep_96k_44k_s_a120_2.wav')
>>> plots.spectrogram('sweep_96k_44k_s_a120_2.wav')
>>> plots.impulse('impulse_96k_44k_s_a120_2.wav')
>>> plots.impulse('impulse_96k_44k_s_a120_2.wav', True)
```

See code in [tests](/crates/simple_src/tests/) for more details.

## References

1. Smith, J.O. Digital Audio Resampling Home Page
    https://ccrma.stanford.edu/~jos/resample/.
2. Alan V. Oppenheim, Ronald W. Schafer.
    Discrete-Time Signal Processing, Third Edition.
