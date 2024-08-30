# simple_src

A simple sample rate converter.

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

For multi channels:

```rust
use simple_src::{sinc, Convert};

let channel1 = vec![1.0, 2.0, 3.0, 4.0];
let channel2 = vec![1.5, 2.5, 3.5, 4.5];
let manager = sinc::Manager::new(2.0, 30.0, 32, 0.1);
let mut converter1 = manager.converter();
let mut converter2 = manager.converter();
let result1 = converter1.process(channel1.into_iter());
let result2 = converter2.process(channel2.into_iter());
// ...
```

See [examples/two_channels.rs](/examples/two_channels.rs) for more information.

Use [plots.py](/plots.py) to show the result of tests.

```python
>>> import os
>>> os.chdir('output')
>>> import plots
>>> plots.show_wav_spectrogram('sweep_96k_44k_xxx.wav')
```

See [tests](/tests/) for more details.

Reference <https://ccrma.stanford.edu/~jos/resample/resample.html>
