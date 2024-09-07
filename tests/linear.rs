use simple_src::{linear::Manager, Convert};

fn convert(file_prefix: &str, sr_old: u32, sr_new: u32) {
    let ratio = sr_new as f64 / sr_old as f64;
    let source_file = format!("{file_prefix}_{}k.wav", sr_old / 1000);
    let target_file = format!(
        "{file_prefix}_{}k_{}k_linear.wav",
        sr_old / 1000,
        sr_new / 1000
    );
    let mut reader = hound::WavReader::open(source_file).unwrap();
    let out_duration = (ratio * (reader.duration() as f64)) as usize;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr_new,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(target_file, spec).unwrap();
    let in_iter = reader
        .samples::<f32>()
        .map(|s| s.unwrap() as f64)
        .chain(std::iter::repeat(0.0));
    Manager::new(ratio)
        .converter()
        .process(in_iter)
        .take(out_duration)
        .for_each(|s| writer.write_sample(s as f32).unwrap());
    writer.finalize().unwrap();
}

#[test]
#[ignore = "generate files"]
// cargo test -r --test linear -- --ignored --exact tlinear
fn tlinear() {
    std::env::set_current_dir("output").unwrap();
    convert("beep", 44100, 48000);
    convert("beep", 48000, 44100);
    convert("sweep", 44100, 48000);
    convert("sweep", 48000, 44100);
    convert("sweep", 48000, 96000);
    convert("sweep", 96000, 48000);
}

#[test]
#[ignore = "display only"]
fn tmultithread() {
    let manager = Manager::new(2.0);
    let h1 = std::thread::spawn(move || {
        let mut converter = manager.converter();
        let samples = (0..10).map(|x| x as f64);
        for s in converter.process(samples) {
            println!("{s}");
        }
    });
    let h2 = std::thread::spawn(move || {
        let mut converter = manager.converter();
        let samples = (-10..0).map(|x| x as f64);
        for s in converter.process(samples) {
            println!("{s}");
        }
    });
    h1.join().unwrap();
    h2.join().unwrap();
}
