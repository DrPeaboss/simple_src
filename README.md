# simple_src

A simple sample rate conversion lib for audio.

## Usage

Usually use *sinc* Converter, it is flexible and high-quality.
The *linear* Converter is not recommended unless performance is really important.

### sinc

With new method:

```rust
use simple_src::{sinc, Convert};

let samples = vec![1.0, 2.0, 3.0, 4.0];
let manager = sinc::Manager::new(2.0, 48.0, 8, 0.1).unwrap();
let mut converter = manager.converter();
for s in converter.process(samples.into_iter()) {
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

For multi-channel example see [two_channels.rs](/examples/two_channels.rs).

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

Recommended initialization parameters for *sinc* converter:

|              | attenuation | quantify |
| ------------ | ----------- | -------- |
| 8bit fast    | 48          | 8        |
| 8bit medium  | 60          | 16       |
| 8bit better  | 72          | 32       |
| 16bit lower  | 84          | 64       |
| 16bit fast   | 96          | 128      |
| 16bit medium | 108         | 256      |
| 16bit better | 120         | 512      |
| 24bit lower  | 132         | 1024     |
| 24bit fast   | 144         | 2048     |
| 24bit medium | 156         | 4096     |
| 24bit better | 168         | 8192     |

The relationship between *attenuation* and *quantify* is about
*Q = 2 ^ (A / 12 - 1)*, *A = 12 + 12 * log2(Q)*.

Due to the amount of calculation and the size of LUT, A = 144 or 156 for 24bit
audio is usually fine, and for 16bit, A = 120 is enough.

## Plots

Use [plots.py](/plots.py) to show the results of conversion. It needs *numpy*, *scipy*
and *matplotlib*.

Here is an example showing the results of a downsampling 96kHz:

```
$ cargo test -r --test testwav -- --ignored --exact --show-output generate
$ cargo test -r --test sinc -- --ignored --exact --show-output ta120_2_96k_down
$ python
>>> import plots
>>> import os
>>> os.chdir('output')
>>> plots.spectrum('beep_96k_44k_s_a120_2.wav')
>>> plots.spectrogram('sweep_96k_44k_s_a120_2.wav')
>>> plots.impulse('impulse_96k_44k_s_a120_2.wav')
>>> plots.impulse('impulse_96k_44k_s_a120_2.wav', True)
```

See code in [tests](/tests/) for more details.

## References

1. Smith, J.O. Digital Audio Resampling Home Page
    https://ccrma.stanford.edu/~jos/resample/.
2. Alan V. Oppenheim, Ronald W. Schafer.
    Discrete-Time Signal Processing, Thrid Edition.
