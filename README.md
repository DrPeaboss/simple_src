# simple_src

A simple sample rate conversion lib for audio.

Usage:

```rust
use simple_src::{sinc, Convert};

let samples = vec![1.0, 2.0, 3.0, 4.0];
let manager = sinc::Manager::new(2.0, 48.0, 8, 0.1);
let mut converter = manager.converter();
for s in converter.process(samples.into_iter()) {
    println!("{s}");
}
```

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

with `sinc::Manager::new` or `sinc::Manager::with_order`, or use
`sinc::Manager::with_raw` with the raw parameters calculated by self.

The relationship between *attenuation* and *quantify* is about *Q = 2 ^ (A / 12 - 1)*.

Due to the amount of calculation and the size of LUT, A = 144 or 156 for 24bit
audio is usually fine, and for 16bit, A = 120 is enough.

For multi-channel example see [examples/two_channels.rs](/examples/two_channels.rs).

The *linear* Converter is not recommended unless performance is really important.

Use [plots.py](/plots.py) to show the result of tests. It need *numpy*, *scipy*
and *matplotlib*.

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

See [tests](/tests/) for more details.

Reference <https://ccrma.stanford.edu/~jos/resample/resample.html>
