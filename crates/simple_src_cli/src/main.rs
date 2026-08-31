use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use simple_src::{Convert, Kernel, SincPath, SrcManager, process_planar};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "simple-src-cli")]
#[command(version, about, long_about = None)]
struct Args {
    /// input wav file path
    input: PathBuf,

    /// output wav file path
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// target sample rate
    #[arg(short = 'r', long, default_value_t = 44100)]
    target_rate: u32,

    #[arg(short, long, default_value_t = 144)]
    attenuation: u32,

    /// LUT quantify for generic interpolation (ignored unless --generic)
    #[arg(short, long, default_value_t = 2048)]
    quantify: u32,

    #[arg(short, long, default_value_t = 0.95)]
    pass_width: f64,

    /// Use generic half-table interpolation instead of polyphase Fast (sinc only)
    #[arg(long)]
    generic: bool,

    /// Conversion kernel: linear, cubic, or sinc
    #[arg(long, default_value = "sinc")]
    kernel: String,
}

fn main() {
    let args = Args::parse();
    let now = std::time::Instant::now();
    match run(&args) {
        Ok(_) => {
            println!("conversion completed, time elapsed: {:?}", now.elapsed());
        }
        Err(e) => {
            eprintln!("conversion failed: {e}");
            std::process::exit(1);
        }
    }
}

type SampleIter<'a> = Box<dyn Iterator<Item = Result<f64>> + 'a>;
type NormFn = Box<dyn Fn(f64) -> f64>;

fn run(args: &Args) -> Result<()> {
    let mut reader = hound::WavReader::open(&args.input)?;
    let input_spec = reader.spec();
    check_input_spec(&input_spec)?;
    let channels = input_spec.channels as usize;
    let input_sr = input_spec.sample_rate;
    let output_sr = args.target_rate;
    let output_frames = get_output_frames(reader.duration(), input_sr, output_sr)?;
    let manager = create_manager(
        input_sr,
        output_sr,
        args.attenuation,
        args.quantify,
        args.pass_width,
        args.generic,
        &args.kernel,
    )?;
    let output_spec = hound::WavSpec {
        channels: input_spec.channels,
        sample_rate: output_sr,
        bits_per_sample: input_spec.bits_per_sample,
        sample_format: input_spec.sample_format,
    };
    let output_file = get_output_file(&args.input, &args.output, output_sr);
    println!("output file is {output_file:?}");
    println!("mode {:?} ratio {}", manager.mode(), manager.ratio());
    let mut writer = hound::WavWriter::create(output_file, output_spec)?;
    let latency = manager.latency();

    let (samples_iter, norm_fn): (SampleIter<'_>, NormFn) = match input_spec.sample_format {
        hound::SampleFormat::Float => {
            let iter = reader.samples::<f32>().map(|s| {
                s.context("failed to read sample from wav")
                    .map(|s| s as f64)
            });
            (Box::new(iter), Box::new(|s: f64| s))
        }
        hound::SampleFormat::Int => match input_spec.bits_per_sample {
            16 => {
                let iter = reader.samples::<i16>().map(|s| {
                    s.context("failed to read sample from wav")
                        .map(|s| s as f64 / 32767.0)
                });
                (
                    Box::new(iter),
                    Box::new(|s: f64| (s * 32767.0).clamp(-32767.0, 32767.0)),
                )
            }
            24 => {
                let iter = reader.samples::<i32>().map(|s| {
                    s.context("failed to read sample from wav").map(|s| {
                        if s < 0 {
                            s as f64 / 8388608.0
                        } else {
                            s as f64 / 8388607.0
                        }
                    })
                });
                (
                    Box::new(iter),
                    Box::new(|s: f64| {
                        (if s < 0.0 {
                            s * 8388608.0
                        } else {
                            s * 8388607.0
                        })
                        .clamp(-8388608.0, 8388607.0)
                    }),
                )
            }
            32 => {
                let iter = reader.samples::<i32>().map(|s| {
                    s.context("failed to read sample from wav").map(|s| {
                        if s < 0 {
                            s as f64 / 2147483648.0
                        } else {
                            s as f64 / 2147483647.0
                        }
                    })
                });
                (
                    Box::new(iter),
                    Box::new(|s: f64| {
                        (if s < 0.0 {
                            s * 2147483648.0
                        } else {
                            s * 2147483647.0
                        })
                        .clamp(-2147483648.0, 2147483647.0)
                    }),
                )
            }
            _ => bail!("unsupported integer bit depth"),
        },
    };

    let mut samples = samples_iter.chain(std::iter::repeat_with(|| Ok(0.0)));
    let mut converters: Vec<_> = (0..channels).map(|_| manager.converter()).collect();

    let buf_len = (2 * latency).max(2048);
    let mut n = 0u64;
    let mut pending_skip = latency;

    while n < output_frames {
        let mut channel_samples: Vec<Vec<f64>> =
            (0..channels).map(|_| Vec::with_capacity(buf_len)).collect();
        for _ in 0..buf_len {
            for chan in channel_samples.iter_mut() {
                let sample = samples
                    .next()
                    .ok_or_else(|| anyhow!("unexpected end of samples"))??;
                chan.push(sample);
            }
        }

        let remaining = (output_frames - n) as usize + pending_skip;
        let mut channel_out: Vec<Vec<f64>> = (0..channels)
            .map(|_| vec![0.0; remaining.min(buf_len * 16)])
            .collect();
        let inputs: Vec<&[f64]> = channel_samples.iter().map(|v| v.as_slice()).collect();
        let mut outputs: Vec<&mut [f64]> =
            channel_out.iter_mut().map(|v| v.as_mut_slice()).collect();
        let (_, produced) = process_planar(&mut converters, &inputs, &mut outputs)?;
        if produced == 0 {
            let mut flushed = 0;
            for (cv, out) in converters.iter_mut().zip(channel_out.iter_mut()) {
                flushed = cv.flush(out);
            }
            if flushed == 0 {
                break;
            }
            write_planar_frames(
                &channel_out,
                flushed,
                &mut pending_skip,
                &mut n,
                output_frames,
                &norm_fn,
                &input_spec,
                &mut writer,
            )?;
            continue;
        }
        write_planar_frames(
            &channel_out,
            produced,
            &mut pending_skip,
            &mut n,
            output_frames,
            &norm_fn,
            &input_spec,
            &mut writer,
        )?;
    }
    writer.finalize()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_planar_frames<W: std::io::Write + std::io::Seek>(
    channel_out: &[Vec<f64>],
    produced: usize,
    pending_skip: &mut usize,
    n: &mut u64,
    output_frames: u64,
    norm_fn: &dyn Fn(f64) -> f64,
    input_spec: &hound::WavSpec,
    writer: &mut hound::WavWriter<W>,
) -> Result<()> {
    let start = (*pending_skip).min(produced);
    *pending_skip -= start;
    let take = (produced - start).min((output_frames - *n) as usize);
    for i in start..start + take {
        for channel in channel_out {
            let normalized = norm_fn(channel[i]);
            match input_spec.sample_format {
                hound::SampleFormat::Float => writer.write_sample(normalized as f32)?,
                hound::SampleFormat::Int => match input_spec.bits_per_sample {
                    16 => writer.write_sample(normalized as i16)?,
                    24 | 32 => writer.write_sample(normalized as i32)?,
                    _ => bail!("unsupported integer bit depth"),
                },
            }
        }
    }
    *n += take as u64;
    Ok(())
}

fn check_input_spec(spec: &hound::WavSpec) -> Result<()> {
    match spec.sample_format {
        hound::SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                bail!("unsupported floating point bit depth, only 32-bit float is supported");
            }
        }
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 | 24 | 32 => {}
            _ => {
                bail!("unsupported integer bit depth, only 16-bit, 24-bit and 32-bit are supported")
            }
        },
    }
    if spec.channels == 0 {
        bail!("bad wav file, which channels is 0");
    }
    Ok(())
}

fn get_output_frames(input_frames: u32, input_sr: u32, output_sr: u32) -> Result<u64> {
    if input_sr == output_sr {
        bail!("sample rate is same, no need to convert");
    }
    Ok(input_frames as u64 * output_sr as u64 / input_sr as u64)
}

fn create_manager(
    input_sr: u32,
    output_sr: u32,
    atten: u32,
    quan: u32,
    pass_width: f64,
    generic: bool,
    kernel_name: &str,
) -> Result<SrcManager> {
    let kernel = match kernel_name {
        "linear" => Kernel::Linear,
        "cubic" => Kernel::Cubic,
        "sinc" => Kernel::Sinc,
        other => bail!("unknown kernel {other}, expected linear, cubic, or sinc"),
    };
    let mut builder = SrcManager::builder()
        .kernel(kernel)
        .sample_rate(input_sr, output_sr);
    if kernel == Kernel::Sinc {
        builder = builder.pass_width(pass_width);
        builder = if generic {
            builder
                .sinc_path(SincPath::Generic)
                .attenuation(atten)
                .quantify(quan)
        } else {
            builder.sinc_path(SincPath::Fast).attenuation(atten)
        };
    }
    builder.build().map_err(|e| {
        anyhow!(
            "failed to initialize SRC converter: {e} (use --generic for half-table interpolation)"
        )
    })
}

fn get_output_file(input: &Path, output: &Option<PathBuf>, output_sr: u32) -> PathBuf {
    if let Some(output_path) = output {
        if output_path.is_dir() {
            let input_parent = input.parent().unwrap_or_else(|| Path::new(""));
            let is_same_dir = match (
                std::fs::canonicalize(input_parent),
                std::fs::canonicalize(output_path),
            ) {
                (Ok(input_dir), Ok(output_dir)) => input_dir == output_dir,
                _ => input_parent == output_path,
            };

            let file_name = input
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new(""));
            let file_stem = input
                .file_stem()
                .unwrap_or_else(|| std::ffi::OsStr::new(""));
            let extension = input.extension();

            if is_same_dir {
                let new_file_name = if let Some(ext) = extension {
                    format!(
                        "{}_{}.{}",
                        file_stem.to_string_lossy(),
                        output_sr,
                        ext.to_string_lossy()
                    )
                } else {
                    format!("{}_{}", file_stem.to_string_lossy(), output_sr)
                };
                return output_path.join(new_file_name);
            } else {
                return output_path.join(file_name);
            }
        }
        return output_path.clone();
    }

    let parent = input.parent().unwrap_or_else(|| Path::new(""));
    let file_stem = input
        .file_stem()
        .unwrap_or_else(|| std::ffi::OsStr::new(""));
    let extension = input.extension();

    let new_file_name = if let Some(ext) = extension {
        format!(
            "{}_{}.{}",
            file_stem.to_string_lossy(),
            output_sr,
            ext.to_string_lossy()
        )
    } else {
        format!("{}_{}", file_stem.to_string_lossy(), output_sr)
    };

    parent.join(new_file_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("simple_src_cli_test_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_tone_wav(path: &Path, sr: u32, frames: u32, bits: u16, float: bool) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: bits,
            sample_format: if float {
                hound::SampleFormat::Float
            } else {
                hound::SampleFormat::Int
            },
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        if float {
            for i in 0..frames {
                w.write_sample(((i as f64 * 0.01).sin()) as f32).unwrap();
            }
        } else if bits == 16 {
            for i in 0..frames {
                w.write_sample((((i as f64 * 0.01).sin()) * 10_000.0) as i16)
                    .unwrap();
            }
        } else {
            for i in 0..frames {
                w.write_sample((((i as f64 * 0.01).sin()) * 1_000_000.0) as i32)
                    .unwrap();
            }
        }
        w.finalize().unwrap();
    }

    fn args(input: PathBuf, output: Option<PathBuf>, kernel: &str, generic: bool) -> Args {
        Args {
            input,
            output,
            target_rate: 48000,
            attenuation: 96,
            quantify: 2048,
            pass_width: 0.95,
            generic,
            kernel: kernel.to_string(),
        }
    }

    #[test]
    fn create_manager_supports_all_kernels() {
        let ok = |k: &str| create_manager(44100, 48000, 96, 128, 0.95, false, k).is_ok();
        assert!(ok("sinc"));
        assert!(ok("linear"));
        assert!(ok("cubic"));
        let err = create_manager(44100, 48000, 96, 128, 0.95, false, "bogus")
            .err()
            .unwrap();
        assert!(err.to_string().contains("unknown kernel"), "{err}");
    }

    #[test]
    fn create_manager_generic_vs_fast_modes() {
        let fast = create_manager(44100, 48000, 96, 128, 0.95, false, "sinc").unwrap();
        assert_eq!(fast.mode(), simple_src::ConvertMode::RationalFast);
        let generic = create_manager(44100, 48000, 96, 128, 0.95, true, "sinc").unwrap();
        assert_eq!(generic.mode(), simple_src::ConvertMode::Rational);
    }

    #[test]
    fn check_input_spec_accepts_supported_formats() {
        let spec = |bits, format| hound::WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: bits,
            sample_format: format,
        };
        check_input_spec(&spec(16, hound::SampleFormat::Int)).unwrap();
        check_input_spec(&spec(24, hound::SampleFormat::Int)).unwrap();
        check_input_spec(&spec(32, hound::SampleFormat::Int)).unwrap();
        check_input_spec(&spec(32, hound::SampleFormat::Float)).unwrap();
    }

    #[test]
    fn check_input_spec_rejects_unsupported_formats() {
        let spec = |bits, format, channels| hound::WavSpec {
            channels,
            sample_rate: 44100,
            bits_per_sample: bits,
            sample_format: format,
        };
        assert!(check_input_spec(&spec(64, hound::SampleFormat::Float, 1)).is_err());
        assert!(check_input_spec(&spec(8, hound::SampleFormat::Int, 1)).is_err());
        assert!(check_input_spec(&spec(16, hound::SampleFormat::Int, 0)).is_err());
    }

    #[test]
    fn get_output_frames_rejects_same_rate() {
        let err = get_output_frames(44100, 44100, 44100).unwrap_err();
        assert!(err.to_string().contains("same"), "{err}");
    }

    #[test]
    fn get_output_frames_computes_truncated_ratio() {
        // 800 frames @ 44100 -> 48000: floor(800 * 48000 / 44100) = 870.
        assert_eq!(get_output_frames(800, 44100, 48000).unwrap(), 870);
        assert_eq!(get_output_frames(1000, 44100, 16000).unwrap(), 362);
    }

    #[test]
    fn output_file_naming_rules() {
        let dir = temp_dir("naming_same");
        let input = dir.join("in.wav");
        let same_dir_expected = dir.join("in_48000.wav");
        assert_eq!(get_output_file(&input, &None, 48000), same_dir_expected);

        let explicit = dir.join("renamed.wav");
        assert_eq!(
            get_output_file(&input, &Some(explicit.clone()), 48000),
            explicit
        );
        assert_eq!(
            get_output_file(&input, &Some(dir.clone()), 48000),
            same_dir_expected
        );

        let other = temp_dir("naming_other");
        let expected_other = other.join("in.wav");
        assert_eq!(
            get_output_file(&input, &Some(other.clone()), 48000),
            expected_other
        );

        let noext = dir.join("in");
        let expected_noext = dir.join("in_48000");
        assert_eq!(get_output_file(&noext, &None, 48000), expected_noext);

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&other).ok();
    }

    #[test]
    fn run_end_to_end_linear_int16() {
        let dir = temp_dir("e2e_linear");
        let input = dir.join("in.wav");
        write_tone_wav(&input, 44100, 800, 16, false);
        let output = dir.join("out.wav");
        let a = args(input, Some(output.clone()), "linear", false);
        run(&a).unwrap();

        let reader = hound::WavReader::open(&output).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 48000);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
        // 800 * 48000 / 44100 truncated to 870 frames.
        assert_eq!(reader.duration(), 870);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_end_to_end_sinc_fast() {
        let dir = temp_dir("e2e_sinc");
        let input = dir.join("in.wav");
        write_tone_wav(&input, 44100, 800, 16, false);
        let output = dir.join("out.wav");
        let a = args(input, Some(output.clone()), "sinc", false);
        run(&a).unwrap();

        let reader = hound::WavReader::open(&output).unwrap();
        assert_eq!(reader.spec().sample_rate, 48000);
        assert_eq!(reader.duration(), 870);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_end_to_end_float32_preserves_format() {
        let dir = temp_dir("e2e_float");
        let input = dir.join("in.wav");
        write_tone_wav(&input, 44100, 500, 32, true);
        let output = dir.join("out.wav");
        let a = args(input, Some(output.clone()), "sinc", false);
        run(&a).unwrap();

        let mut reader = hound::WavReader::open(&output).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 48000);
        assert_eq!(spec.sample_format, hound::SampleFormat::Float);
        assert_eq!(spec.bits_per_sample, 32);
        // 500 * 48000 / 44100 truncated to 544 frames.
        assert_eq!(reader.duration(), 544);
        let max_abs = reader
            .samples::<f32>()
            .map(|s| s.unwrap().abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs.is_finite() && max_abs <= 2.0, "level {max_abs}"); // sinc ripple ~5% over 1.0
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_rejects_same_rate_and_missing_input() {
        let dir = temp_dir("e2e_errors");
        let input = dir.join("in.wav");
        write_tone_wav(&input, 48000, 100, 16, false);
        let output = dir.join("out.wav");
        let mut a = args(input.clone(), Some(output.clone()), "linear", false);
        a.target_rate = 48000;
        let err = run(&a).unwrap_err();
        assert!(err.to_string().contains("same"), "{err}");

        let missing = dir.join("nope.wav");
        let a2 = args(missing, Some(output.clone()), "linear", false);
        assert!(run(&a2).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
