//! Finer overshoot analysis: map each output sample to its bracketing input pair.
use simple_src::{Convert, Kernel, SrcManager};

fn convert_stream(manager: &SrcManager, input: &[f64]) -> Vec<f64> {
    let mut cv = manager.converter();
    let mut out = Vec::new();
    let mut iter = input.iter().copied();
    while let Some(s) = cv.next_sample(&mut iter) {
        out.push(s);
    }
    let mut tail = [0.0; 64];
    loop {
        let n = cv.flush(&mut tail);
        if n == 0 {
            break;
        }
        out.extend_from_slice(&tail[..n]);
    }
    out
}

/// For each output index i, find input phase `t = i/ratio` and bracket [k, k+1].
/// Report if output exceeds [min(yk,yk+1), max(yk,yk+1)].
fn bracket_overshoot(input: &[f64], output: &[f64], ratio: f64) -> (f64, f64, usize) {
    let mut max_above = 0.0_f64;
    let mut max_below = 0.0_f64;
    let mut worst_i = 0;
    for (i, &y) in output.iter().enumerate() {
        let t = i as f64 / ratio;
        let k = t.floor() as usize;
        if k + 1 >= input.len() {
            continue;
        }
        let y0 = input[k];
        let y1 = input[k + 1];
        let lo = y0.min(y1);
        let hi = y0.max(y1);
        if y > hi + 1e-15 {
            let d = y - hi;
            if d > max_above {
                max_above = d;
                worst_i = i;
            }
        }
        if y < lo - 1e-15 {
            let d = lo - y;
            if d > max_below {
                max_below = d;
                worst_i = i;
            }
        }
    }
    (max_above, max_below, worst_i)
}

fn main() {
    let step: Vec<f64> = (0..128).map(|i| if i < 64 { 0.0 } else { 1.0 }).collect();
    let impulse = {
        let mut v = vec![0.0; 256];
        v[64] = 1.0;
        v
    };

    println!("=== Bracket overshoot (output vs local input segment) ===\n");
    for ratio in [2.0, 0.5, 48000.0 / 44100.0] {
        println!("ratio={ratio}");
        for kernel in [Kernel::Linear, Kernel::Cubic] {
            let mgr = SrcManager::builder()
                .ratio(ratio)
                .kernel(kernel)
                .build()
                .unwrap();
            for (name, input) in [("step", &step), ("impulse", &impulse)] {
                let out = convert_stream(&mgr, input);
                let (above, below, wi) = bracket_overshoot(input, &out, ratio);
                println!(
                    "  {kernel:?} {name:8} above={above:.6} below={below:.6} worst_idx={wi}"
                );
            }
        }
        println!();
    }

    // Zoom into step edge for ratio=2 cubic
    println!("=== Step edge detail (ratio=2, cubic) ===");
    let mgr = SrcManager::builder()
        .ratio(2.0)
        .kernel(Kernel::Cubic)
        .build()
        .unwrap();
    let out = convert_stream(&mgr, &step);
    let ratio = 2.0;
    for i in 120..140.min(out.len()) {
        let t = i as f64 / ratio;
        let k = t.floor() as usize;
        let frac = t - k as f64;
        let bracket = if k + 1 < step.len() {
            format!("[{}, {}]", step[k], step[k + 1])
        } else {
            "?".into()
        };
        println!(
            "  out[{i}]={:.6} t={t:.3} k={k} frac={frac:.3} bracket={bracket}",
            out[i]
        );
    }

    // Startup: zero-padded left edge
    println!("\n=== Startup edge (input DC=1, ratio=2) ===");
    let dc = vec![1.0; 32];
    for kernel in [Kernel::Linear, Kernel::Cubic] {
        let mgr = SrcManager::builder()
            .ratio(2.0)
            .kernel(kernel)
            .build()
            .unwrap();
        let out = convert_stream(&mgr, &dc);
        let (above, below, _) = bracket_overshoot(&dc, &out, 2.0);
        println!(
            "  {kernel:?} first4={:?} bracket_above={above:.6} bracket_below={below:.6}",
            &out[..4]
        );
    }
}
