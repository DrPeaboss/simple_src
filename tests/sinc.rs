use std::io::Write;

use simple_src::{sinc, Convert};

struct Src {
    sr_old: u32,
    sr_new: u32,
    manager: sinc::Manager,
}

impl Src {
    fn new_by_order(sr_old: u32, sr_new: u32, atten: f64, quan: u32, order: u32) -> Self {
        Self {
            sr_old,
            sr_new,
            manager: sinc::Manager::with_order(sr_new as f64 / sr_old as f64, atten, quan, order)
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
            manager: sinc::Manager::new(sr_new as f64 / sr_old as f64, atten, quan, trans_width)
                .unwrap(),
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
    let mut reader = hound::WavReader::open(source_file).unwrap();
    let out_duration = (ratio * (reader.duration() as f64)) as usize;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: src.sr_new,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(target_file, spec).unwrap();
    let in_iter = reader
        .samples::<f32>()
        .map(|s| s.unwrap() as f64)
        .chain(std::iter::repeat(0.0));
    let mut cvtr = src.manager.converter();
    cvtr.process(in_iter)
        .skip(src.manager.latency())
        .take(out_duration)
        .for_each(|s| writer.write_sample(s as f32).unwrap());
    writer.finalize().unwrap();
}

fn impulse(src: &Src, remark: &str) {
    let filename = format!(
        "impulse_{}k_{}k_s_{remark}.wav",
        src.sr_old / 1000,
        src.sr_new / 1000
    );
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: src.sr_new,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(filename, spec).unwrap();
    let count = src.sr_old as usize;
    let in_iter = (0..count)
        .enumerate()
        .map(|(i, _)| if i == count / 2 { 1.0 } else { 0.0 });
    let mut cvrt = src.manager.converter();
    cvrt.process(in_iter)
        .skip(src.manager.latency())
        .take(src.sr_new as usize)
        .for_each(|s| writer.write_sample(s as f32).unwrap());
    writer.finalize().unwrap();
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
            file.write(&s.to_ne_bytes()).unwrap();
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
    println!("order of 44k to 48k {remark} is {}", src.manager.order());
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 100.0, 128, trans_width);
    println!("order of 48k to 44k {remark} is {}", src.manager.order());
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
    println!("order of 44k to 48k {remark} is {}", src.manager.order());
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 120.0, 128, trans_width);
    println!("order of 48k to 44k {remark} is {}", src.manager.order());
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
    println!("order of 44k to 48k {remark} is {}", src.manager.order());
    impulse(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 110.0, 128, trans_width);
    println!("order of 48k to 44k {remark} is {}", src.manager.order());
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
    println!("order of 44k to 48k {remark} is {}", src.manager.order());
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 120.0, 512, trans_width);
    println!("order of 48k to 44k {remark} is {}", src.manager.order());
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
    println!("order of 44k to 48k {remark} is {}", src.manager.order());
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 144.0, 2048, trans_width);
    println!("order of 48k to 44k {remark} is {}", src.manager.order());
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
    println!("order of 44k to 48k {remark} is {}", src.manager.order());
    convert("sweep", &src, remark);
    impulse(&src, remark);
    impulse_raw(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 156.0, 4096, trans_width);
    println!("order of 48k to 44k {remark} is {}", src.manager.order());
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
    println!("order of 44k to 48k {remark} is {}", src.manager.order());
    convert("sweep", &src, remark);
    impulse(&src, remark);
    impulse_raw(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 168.0, 8192, trans_width);
    println!("order of 48k to 44k {remark} is {}", src.manager.order());
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
    println!("order of 44k to 48k {remark} is {}", src.manager.order());
    convert("sweep", &src, remark);
    impulse(&src, remark);
    impulse_raw(&src, remark);
    let src = Src::new_by_trans_width(48000, 44100, 150.0, 2048, trans_width);
    println!("order of 48k to 44k {remark} is {}", src.manager.order());
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
    println!("order of 96k to 44k {remark} is {}", src.manager.order());
    convert("sweep", &src, remark);
    let trans_width = 4000.0 / 24000.0;
    let src = Src::new_by_trans_width(96000, 48000, 150.0, 2048, trans_width);
    println!("order of 96k to 48k {remark} is {}", src.manager.order());
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
    println!("order of 96 to 44k {remark} is {}", src.manager.order());
    convert("beep", &src, remark);
    convert("sweep", &src, remark);
    impulse(&src, remark);
    let trans_width = 4000.0 / 24000.0;
    let src = Src::new_by_trans_width(96000, 48000, 120.0, 512, trans_width);
    println!("order of 96k to 48k {remark} is {}", src.manager.order());
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
    println!("order of 192k to 44k a120 is {}", src.manager.order());
    let trans_width = 4000.0 / 24000.0;
    let src = Src::new_by_trans_width(192000, 48000, 120.0, 512, trans_width);
    println!("order of 192k to 48k a120 is {}", src.manager.order());
}

#[test]
#[ignore = "slow"]
// cargo test -r --test sinc -- --ignored --exact --show-output ta150_1_192k_down_order
fn ta150_1_192k_down_order() {
    let trans_width = 2050.0 / 22050.0;
    let src = Src::new_by_trans_width(192000, 44100, 150.0, 2048, trans_width);
    println!("order of 192k to 44k a150 is {}", src.manager.order());
    let trans_width = 4000.0 / 24000.0;
    let src = Src::new_by_trans_width(192000, 48000, 150.0, 2048, trans_width);
    println!("order of 192k to 48k a150 is {}", src.manager.order());
}

#[test]
#[ignore = "display only"]
fn tmultithread() {
    let manager = sinc::Manager::new(2.0, 30.0, 16, 0.1).unwrap();
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
