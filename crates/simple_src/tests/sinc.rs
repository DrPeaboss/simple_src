use std::io::Write;

use simple_src::{Convert, ConvertMode, SrcBuilder, SrcManager};

struct Src {
    sr_old: u32,
    sr_new: u32,
    manager: SrcManager,
}

impl Src {
    fn new_by_order(sr_old: u32, sr_new: u32, atten: f64, quan: u32, order: u32) -> Self {
        Self {
            sr_old,
            sr_new,
            manager: SrcManager::builder()
                .ratio(sr_new as f64 / sr_old as f64)
                .generic()
                .attenuation(atten)
                .quantify(quan)
                .order(order)
                .build()
                .unwrap(),
        }
    }

    fn new_by_trans_width(
        sr_old: u32,
        sr_new: u32,
        atten: f64,
        quan: u32,
        trans_width: f64,
    ) -> Self {
        Self {
            sr_old,
            sr_new,
            manager: SrcManager::builder()
                .ratio(sr_new as f64 / sr_old as f64)
                .generic()
                .attenuation(atten)
                .quantify(quan)
                .trans_width(trans_width)
                .build()
                .unwrap(),
        }
    }

    fn new_by_builder(sr_old: u32, sr_new: u32, builder: SrcBuilder) -> Self {
        Self {
            sr_old,
            sr_new,
            manager: builder.build().unwrap(),
        }
    }
}

fn convert(file_prefix: &str, src: &Src, remark: &str) {
    let ratio = src.sr_new as f64 / src.sr_old as f64;
    let source_file = format!("{file_prefix}_{}k.wav", src.sr_old / 1000);
    let target_file = format!(
        "{file_prefix}_{}k_{}k_s_{remark}.wav",
        src.sr_old / 1000,
        src.sr_new / 1000
    );
    let (source, source_sr) = wavers::read::<f32, _>(source_file).unwrap();
    assert_eq!(source_sr as u32, src.sr_old);
    let out_duration = (ratio * (source.len() as f64)) as usize;
    let in_iter = source.iter().map(|&s| s as f64);
    let out: Vec<f32> = src
        .manager
        .converter()
        .process(in_iter)
        .skip(src.manager.latency())
        .take(out_duration)
        .map(|s| s as f32)
        .collect();
    wavers::write(target_file, &out, src.sr_new as i32, 1).unwrap();
}

fn impulse(src: &Src, remark: &str) {
    let filename = format!(
        "impulse_{}k_{}k_s_{remark}.wav",
        src.sr_old / 1000,
        src.sr_new / 1000
    );
    let count = src.sr_old as usize;
    let in_iter = (0..count)
        .enumerate()
        .map(|(i, _)| if i == count / 2 { 1.0 } else { 0.0 });
    let out: Vec<f32> = src
        .manager
        .converter()
        .process(in_iter)
        .skip(src.manager.latency())
        .take(src.sr_new as usize)
        .map(|s| s as f32)
        .collect();
    wavers::write(filename, &out, src.sr_new as i32, 1).unwrap();
}

fn impulse_raw(src: &Src, remark: &str) {
    let filename = format!(
        "impulse_{}k_{}k_s_{remark}.f64",
        src.sr_old / 1000,
        src.sr_new / 1000
    );
    let mut file = std::fs::File::create(filename).unwrap();
    let count = src.sr_old as usize;
    let in_iter = (0..count)
        .enumerate()
        .map(|(i, _)| if i == count / 2 { 1.0 } else { 0.0 });
    let mut cvrt = src.manager.converter();
    cvrt.process(in_iter)
        .skip(src.manager.latency())
        .take(src.sr_new as usize)
        .for_each(|s| {
            file.write_all(&s.to_ne_bytes()).unwrap();
        });
    file.flush().unwrap();
}

fn cwd() {
    std::env::set_current_dir("output").unwrap();
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact ta100_1
fn ta100_1() {
    cwd();
    let remark = "a100_1";
    let src = Src::new_by_order(44100, 48000, 100.0, 128, 128);
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let src = Src::new_by_order(48000, 44100, 100.0, 128, 128);
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta100_2
fn ta100_2() {
    cwd();
    let remark = "a100_2";
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(44100, 48000, 100.0, 128, trans_width);
    println!(
        "order of 44k to 48k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 100.0, 128, trans_width);
    println!(
        "order of 48k to 44k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta120_1
fn ta120_1() {
    cwd();
    let remark = "a120_1";
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(44100, 48000, 120.0, 128, trans_width);
    println!(
        "order of 44k to 48k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 120.0, 128, trans_width);
    println!(
        "order of 48k to 44k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta110_1
fn ta110_1() {
    cwd();
    let remark = "a110_1";
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(44100, 48000, 110.0, 128, trans_width);
    println!(
        "order of 44k to 48k {remark} is {}",
        src.manager.order().unwrap()
    );
    impulse(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 110.0, 128, trans_width);
    println!(
        "order of 48k to 44k {remark} is {}",
        src.manager.order().unwrap()
    );
    impulse(&src, remark);
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta120_2
fn ta120_2() {
    cwd();
    let remark = "a120_2";
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(44100, 48000, 120.0, 512, trans_width);
    println!(
        "order of 44k to 48k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 120.0, 512, trans_width);
    println!(
        "order of 48k to 44k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
    impulse(&src, remark);
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta144_1
fn ta144_1() {
    cwd();
    let remark = "a144_1";
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(44100, 48000, 144.0, 2048, trans_width);
    println!(
        "order of 44k to 48k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 144.0, 2048, trans_width);
    println!(
        "order of 48k to 44k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
    impulse(&src, remark);
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta156_1
fn ta156_1() {
    cwd();
    let remark = "a156_1";
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(44100, 48000, 156.0, 4096, trans_width);
    println!(
        "order of 44k to 48k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
    impulse(&src, remark);
    impulse_raw(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 156.0, 4096, trans_width);
    println!(
        "order of 48k to 44k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
    impulse(&src, remark);
    impulse_raw(&src, remark);
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta168_1
fn ta168_1() {
    cwd();
    let remark = "a168_1";
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(44100, 48000, 168.0, 8192, trans_width);
    println!(
        "order of 44k to 48k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
    impulse(&src, remark);
    impulse_raw(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 168.0, 8192, trans_width);
    println!(
        "order of 48k to 44k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
    impulse(&src, remark);
    impulse_raw(&src, remark);
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta150_1
fn ta150_1() {
    cwd();
    let remark = "a150_1";
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(44100, 48000, 150.0, 2048, trans_width);
    println!(
        "order of 44k to 48k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
    impulse(&src, remark);
    impulse_raw(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 150.0, 2048, trans_width);
    println!(
        "order of 48k to 44k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
    impulse(&src, remark);
    impulse_raw(&src, remark);
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta150_1_96k_down
fn ta150_1_96k_down() {
    cwd();
    let remark = "a150_1";
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(96000, 44100, 150.0, 2048, trans_width);
    println!(
        "order of 96k to 44k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
    let trans_width = 4000.0 / 24000.0;
    let src = Src::new_by_trans_width(96000, 48000, 150.0, 2048, trans_width);
    println!(
        "order of 96k to 48k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("sweep", &src, remark);
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta120_2_96k_down
fn ta120_2_96k_down() {
    cwd();
    let remark = "a120_2";
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(96000, 44100, 120.0, 512, trans_width);
    println!(
        "order of 96 to 44k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let trans_width = 4000.0 / 24000.0;
    let src = Src::new_by_trans_width(96000, 48000, 120.0, 512, trans_width);
    println!(
        "order of 96k to 48k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta120_2_192k_down_order
fn ta120_2_192k_down_order() {
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(192000, 44100, 120.0, 512, trans_width);
    println!(
        "order of 192k to 44k a120 is {}",
        src.manager.order().unwrap()
    );
    let trans_width = 4000.0 / 24000.0;
    let src = Src::new_by_trans_width(192000, 48000, 120.0, 512, trans_width);
    println!(
        "order of 192k to 48k a120 is {}",
        src.manager.order().unwrap()
    );
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta150_1_192k_down_order
fn ta150_1_192k_down_order() {
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(192000, 44100, 150.0, 2048, trans_width);
    println!(
        "order of 192k to 44k a150 is {}",
        src.manager.order().unwrap()
    );
    let trans_width = 4000.0 / 24000.0;
    let src = Src::new_by_trans_width(192000, 48000, 150.0, 2048, trans_width);
    println!(
        "order of 192k to 48k a150 is {}",
        src.manager.order().unwrap()
    );
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta96_1
fn ta96_1() {
    cwd();
    let remark = "a96_1";
    let builder = SrcManager::builder()
        .sample_rate(44100, 48000)
        .attenuation(96)
        .quantify(128)
        .pass_freq(20000);
    let src = Src::new_by_builder(44100, 48000, builder);
    println!(
        "order of 44k to 48k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let builder = SrcManager::builder()
        .sample_rate(48000, 44100)
        .attenuation(96)
        .quantify(128)
        .pass_freq(20000);
    let src = Src::new_by_builder(48000, 44100, builder);
    println!(
        "order of 48k to 44k {remark} is {}",
        src.manager.order().unwrap()
    );
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
}

#[test]
#[ignore = "display only"]
fn tmultithread() {
    let manager = SrcManager::builder()
        .ratio(2.0)
        .generic()
        .attenuation(30.0)
        .quantify(16)
        .trans_width(0.1)
        .build()
        .unwrap();
    let manager2 = manager.clone();
    let h1 = std::thread::spawn(move || {
        let mut converter = manager.converter();
        let samples = (0..10).map(|x| x as f64);
        for s in converter.process(samples) {
            println!("{s}");
        }
    });
    let h2 = std::thread::spawn(move || {
        let mut converter = manager2.converter();
        let samples = (-10..0).map(|x| x as f64);
        for s in converter.process(samples) {
            println!("{s}");
        }
    });
    h1.join().unwrap();
    h2.join().unwrap();
}

/// Skipping `latency()` must align the output timebase with the input: the
/// FIR group delay has to be exactly `order / 2` input samples. For a
/// symmetric, band-limited kernel the impulse-response centroid of the
/// sampled output equals the continuous one, so the test reads the alignment
/// error directly. The engine historically anchored its window one input
/// sample late (group delay `order / 2 + 1`), leaving a constant fractional
/// misalignment of up to one input sample after the integer latency skip.
#[test]
fn fir_latency_aligns_group_delay() {
    let r = 44100.0 / 96000.0;
    let builders = [
        SrcManager::builder()
            .sample_rate(96000, 44100)
            .attenuation(96)
            .trans_width(0.05),
        SrcManager::builder()
            .sample_rate(96000, 44100)
            .attenuation(96)
            .quantify(128)
            .trans_width(0.05)
            .generic(),
    ];
    for builder in builders {
        let manager = builder.build().unwrap();
        let order = manager.order().unwrap() as f64;
        let latency = manager.latency();
        let p = order as usize;
        let n = 4 * p + 1;
        let input = (0..n).map(|i| if i == p { 1.0 } else { 0.0 });
        let mut converter = manager.converter();
        let mut out: Vec<f64> = converter.process(input).skip(latency).collect();
        let mut buf = [0.0f64; 4096];
        loop {
            let produced = converter.flush(&mut buf);
            if produced == 0 {
                break;
            }
            out.extend_from_slice(&buf[..produced]);
        }
        let (num, den) = out
            .iter()
            .enumerate()
            .fold((0.0f64, 0.0f64), |(a, b), (j, &s)| {
                (a + j as f64 * s, b + s)
            });
        let centroid = num / den;
        // Contract differs by mode. Fast designs quantize the half-order to
        // a multiple of the reduced ratio's denominator, making the group
        // delay r*order/2 an exact integer == `latency` and landing the
        // content exactly on the output grid (centroid == p*r). Generic
        // keeps the designed order, so its content carries the usual
        // rounding residual r*order/2 - latency.
        let expected = match manager.mode() {
            ConvertMode::RationalFast => {
                let group_delay = r * order / 2.0;
                assert!(
                    (group_delay - latency as f64).abs() < 1e-9,
                    "group delay {group_delay} != integer latency {latency} \
                     (order {order} not denominator-aligned)"
                );
                p as f64 * r
            }
            _ => (p as f64 + order / 2.0) * r - latency as f64,
        };
        assert!(
            (centroid - expected).abs() < 1e-4,
            "IR centroid {centroid:.6} != expected {expected:.6} \
             (order {order}, latency {latency}, {} samples)",
            out.len()
        );
    }
}
