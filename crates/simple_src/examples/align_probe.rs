use simple_src::{Convert, Kernel, SrcManager};

fn probe(label: &str, kernel: Kernel, ratio: f64) {
    let mgr = SrcManager::builder()
        .ratio(ratio)
        .kernel(kernel)
        .build()
        .unwrap();
    println!(
        "=== {label} kernel={kernel:?} ratio={ratio} mode={:?} parts={:?}",
        mgr.mode(),
        mgr.ratio_parts()
    );
    let input: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let mut cv = mgr.converter();
    let mut iter = input.iter().copied();
    for i in 0..8 {
        if let Some(s) = cv.next_sample(&mut iter) {
            println!("  out[{i}]={s}");
        }
    }
}

fn main() {
    for kernel in [Kernel::Linear, Kernel::Cubic] {
        probe("ratio=2.0 (rational)", kernel, 2.0);
        probe("ratio=PI (float)", kernel, std::f64::consts::PI);
        probe("ratio=48000/44100", kernel, 48000.0 / 44100.0);
        println!();
    }
}
