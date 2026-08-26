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

    /// Conversion kernel: linear or sinc
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
        "sinc" => Kernel::Sinc,
        other => bail!("unknown kernel {other}, expected linear or sinc"),
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
