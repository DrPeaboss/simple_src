# simple_src

A simple sample rate conversion lib.

Usage:

```rust
use simple_src::{linear, Convert};

let samples = vec![1.0, 2.0, 3.0, 4.0];
let manager = linear::Manager::new(2.0);
let mut converter = manager.converter();
for s in converter.process(samples.into_iter()) {
    println("{s}");
}
```

`linear` is not recommended unless performance is really important.

Recommended initialization parameters for `sinc` converter:

|              | atten | quan | remark             |
| ------------ | ----- | ---- | ------------------ |
| 16bit fast   | 100   | 128  |                    |
| 16bit better | 110   | 256  |                    |
| 16bit best   | 120   | 512  |                    |
| 24bit fast   | 140   | 1024 | actually about 135 |
| 24bit better | 150   | 2048 | actually about 145 |
| 24bit best   | 160   | 4096 | actually about 155 |

with `sinc::Manager::new` or `sinc::Manager::with_order`, or use
`sinc::Manager::with_raw` with the raw parameters calculated by self.

```rust
use simple_src::{sinc, Convert};

let samples = vec![1.0, 2.0, 3.0, 4.0];
let manager = sinc::Manager::new(2.0, 30.0, 32, 0.1);
let mut converter = manager.converter();
for s in converter.process(samples.into_iter()) {
    println("{s}");
}
```

For multi channels see [examples/two_channels.rs](/examples/two_channels.rs).

Use [plots.py](/plots.py) to show the result of tests.

```python
>>> import os
>>> os.chdir('output')
>>> import plots
>>> plots.show_wav_spectrogram('sweep_96k_44k_xxx.wav')
```

See [tests](/tests/) for more details.

Reference <https://ccrma.stanford.edu/~jos/resample/resample.html>
