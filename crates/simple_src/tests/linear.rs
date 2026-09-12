use simple_src::{Convert, SrcManager};

fn convert(file_prefix: &str, sr_old: u32, sr_new: u32) {
    let ratio = sr_new as f64 / sr_old as f64;
    let source_file = format!("{file_prefix}_{}k.wav", sr_old / 1000);
    let target_file = format!(
        "{file_prefix}_{}k_{}k_linear.wav",
        sr_old / 1000,
        sr_new / 1000
    );
    let (source, source_sr) = wavers::read::<f32, _>(source_file).unwrap();
    assert_eq!(source_sr, sr_old as i32);
    let out_duration = (ratio * (source.len() as f64)) as usize;
    let in_iter = source.iter().map(|&s| s as f64);
    let out: Vec<f32> = SrcManager::with_ratio(ratio)
        .unwrap()
        .converter()
        .process(in_iter)
        .take(out_duration)
        .map(|s| s as f32)
        .collect();
    wavers::write(target_file, &out, sr_new as i32, 1).unwrap();
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
    let manager = SrcManager::with_ratio(2.0).unwrap();
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
