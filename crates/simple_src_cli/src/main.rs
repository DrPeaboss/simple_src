use anyhow::{Result, anyhow, bail};
use clap::Parser;
use i24::i24;
use simple_src::{Convert, Kernel, SincPath, SrcManager, process_planar};
use std::path::{Path, PathBuf};
use wavers::{Wav, WavType};

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

/// Sample formats the CLI can read and write. The output file always
/// keeps the input format.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    Int16,
    Int24,
    Int32,
    Float32,
    Float64,
}

fn detect_format(wav_type: WavType) -> Result<Format> {
    match wav_type {
        WavType::Pcm16 | WavType::EPcm16 => Ok(Format::Int16),
        WavType::Pcm24 | WavType::EPcm24 => Ok(Format::Int24),
        WavType::Pcm32 | WavType::EPcm32 => Ok(Format::Int32),
        WavType::Float32 | WavType::EFloat32 => Ok(Format::Float32),
        WavType::Float64 | WavType::EFloat64 => Ok(Format::Float64),
    }
}

#[derive(Clone, Copy, Debug)]
struct InputSpec {
    channels: usize,
    sample_rate: u32,
    /// total interleaved frames
    frames: usize,
    format: Format,
}

/// Reads interleaved samples chunk by chunk in the file's native format and
/// normalizes them to f64. Reading in native types (instead of asking wavers
/// to convert) keeps the historical normalization constants and avoids the
/// library's i24 -> f64 path, which scales by i32::MAX.
struct SourceReader {
    inner: NativeInner,
    /// interleaved samples left unread
    remaining: usize,
}

enum NativeInner {
    I16(Wav<i16>),
    I24(Wav<i24>),
    I32(Wav<i32>),
    F32(Wav<f32>),
    F64(Wav<f64>),
}

impl SourceReader {
    fn open(path: &Path) -> Result<(Self, InputSpec)> {
        // The type parameter only affects data reads, so a probe through
        // Wav<f64> inspects the header without touching the data chunk.
        let probe = Wav::<f64>::from_path(path)
            .map_err(|e| anyhow!("failed to open {}: {e}", path.display()))?;
        let channels = probe.n_channels() as usize;
        let sample_rate = probe.sample_rate();
        let format = detect_format(probe.encoding())?;
        check_spec(format, channels)?;
        let remaining = probe.n_samples();
        let inner = match format {
            Format::Int16 => NativeInner::I16(
                Wav::<i16>::from_path(path).map_err(|e| anyhow!("failed to open input: {e}"))?,
            ),
            Format::Int24 => NativeInner::I24(
                Wav::<i24>::from_path(path).map_err(|e| anyhow!("failed to open input: {e}"))?,
            ),
            Format::Int32 => NativeInner::I32(
                Wav::<i32>::from_path(path).map_err(|e| anyhow!("failed to open input: {e}"))?,
            ),
            Format::Float32 => NativeInner::F32(
                Wav::<f32>::from_path(path).map_err(|e| anyhow!("failed to open input: {e}"))?,
            ),
            Format::Float64 => NativeInner::F64(
                Wav::<f64>::from_path(path).map_err(|e| anyhow!("failed to open input: {e}"))?,
            ),
        };
        Ok((
            SourceReader { inner, remaining },
            InputSpec {
                channels,
                sample_rate: sample_rate as u32,
                frames: remaining / channels,
                format,
            },
        ))
    }

    /// Reads up to `frames` frames of interleaved samples, normalized to f64.
    /// The request is clamped to the remaining data because wavers errors on
    /// short reads instead of returning a short block.
    fn read_frames(&mut self, frames: usize, spec: &InputSpec) -> Result<Vec<f64>> {
        let n = (frames * spec.channels).min(self.remaining);
        self.remaining -= n;
        if n == 0 {
            return Ok(Vec::new());
        }
        let samples = match &mut self.inner {
            NativeInner::I16(w) => w
                .read_samples(n)?
                .iter()
                .map(|&s| s as f64 / 32767.0)
                .collect::<Vec<f64>>(),
            NativeInner::I24(w) => w
                .read_samples(n)?
                .iter()
                .map(|&s| {
                    let v = s.to_i32();
                    if v < 0 {
                        v as f64 / 8388608.0
                    } else {
                        v as f64 / 8388607.0
                    }
                })
                .collect::<Vec<f64>>(),
            NativeInner::I32(w) => w
                .read_samples(n)?
                .iter()
                .map(|&s| {
                    if s < 0 {
                        s as f64 / 2147483648.0
                    } else {
                        s as f64 / 2147483647.0
                    }
                })
                .collect::<Vec<f64>>(),
            NativeInner::F32(w) => w
                .read_samples(n)?
                .iter()
                .map(|&s| s as f64)
                .collect::<Vec<f64>>(),
            NativeInner::F64(w) => w.read_samples(n)?.iter().copied().collect::<Vec<f64>>(),
        };
        Ok(samples)
    }
}

/// Accumulates normalized output samples in the target format and writes the
/// whole file at once (wavers has no streaming writer).
///
/// Integer variants carry a deterministic PRNG state for TPDF dither.
enum Sink {
    I16 { v: Vec<i16>, rng: u64 },
    I24 { v: Vec<i24>, rng: u64 },
    I32 { v: Vec<i32>, rng: u64 },
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// Seed for the output dither PRNG (splitmix64). The dither sequence is a
/// function of this constant and the sample order, so conversions are
/// bit-reproducible.
const DITHER_SEED: u64 = 0x243F_6A88_85A3_08D4;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform `[0, 1)` from the high 53 bits of a splitmix64 draw.
fn splitmix_uniform(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Triangular Probability Density Function dither in LSB units, peak ±1:
/// the sum of two independent uniforms centered at zero. Standard practice
/// when reducing floating-point material to a fixed integer depth.
fn tpdf_lsb(rng: &mut u64) -> f64 {
    splitmix_uniform(rng) + splitmix_uniform(rng) - 1.0
}

impl Sink {
    fn new(format: Format) -> Self {
        match format {
            Format::Int16 => Sink::I16 { v: Vec::new(), rng: DITHER_SEED },
            Format::Int24 => Sink::I24 { v: Vec::new(), rng: DITHER_SEED },
            Format::Int32 => Sink::I32 { v: Vec::new(), rng: DITHER_SEED },
            Format::Float32 => Sink::F32(Vec::new()),
            Format::Float64 => Sink::F64(Vec::new()),
        }
    }

    fn push(&mut self, s: f64) {
        // Round (not truncate): HA's bit-depth probe feeds signals as small as
        // ±1 LSB, where as-casting zeroes ~99% of samples and adds a DC bias
        // on every integer output. TPDF dither decouples the rounding error
        // from the signal: the probe's sources sit at the quantization floor
        // (24-bit source RMS ≈ 0.5 LSB, with a −0.5 LSB truncation bias from
        // the MATLAB generator), where an un-dithered rounding pattern is a
        // threshold image of the signal. Its 256-point FFT then takes DC and
        // Nyquist as exact integer sums, which can cancel to precisely 0.0 —
        // log10(0.0) = −Inf poisons the band-average and the depth is
        // reported as unsupported even though the audio is correct.
        match self {
            Sink::I16 { v, rng } => {
                let d = tpdf_lsb(rng);
                v.push((s * 32767.0 + d).clamp(-32767.0, 32767.0).round() as i16);
            }
            Sink::I24 { v, rng } => {
                let scaled = if s < 0.0 {
                    s * 8388608.0
                } else {
                    s * 8388607.0
                };
                v.push(i24::from_i32(
                    (scaled + tpdf_lsb(rng))
                        .clamp(-8388608.0, 8388607.0)
                        .round() as i32,
                ));
            }
            Sink::I32 { v, rng } => {
                let scaled = if s < 0.0 {
                    s * 2147483648.0
                } else {
                    s * 2147483647.0
                };
                v.push((scaled + tpdf_lsb(rng))
                    .clamp(-2147483648.0, 2147483647.0)
                    .round() as i32);
            }
            Sink::F32(v) => v.push(s as f32),
            Sink::F64(v) => v.push(s),
        }
    }

    fn save(self, path: &Path, sample_rate: u32, channels: usize) -> Result<()> {
        match self {
            Sink::I16 { v, .. } => wavers::write(path, &v, sample_rate as i32, channels as u16)
                .map_err(|e| anyhow!("failed to write output: {e}"))?,
            Sink::I24 { v, .. } => wavers::write(path, &v, sample_rate as i32, channels as u16)
                .map_err(|e| anyhow!("failed to write output: {e}"))?,
            Sink::I32 { v, .. } => wavers::write(path, &v, sample_rate as i32, channels as u16)
                .map_err(|e| anyhow!("failed to write output: {e}"))?,
            Sink::F32(v) => wavers::write(path, &v, sample_rate as i32, channels as u16)
                .map_err(|e| anyhow!("failed to write output: {e}"))?,
            Sink::F64(v) => wavers::write(path, &v, sample_rate as i32, channels as u16)
                .map_err(|e| anyhow!("failed to write output: {e}"))?,
        }
        Ok(())
    }
}

fn check_spec(format: Format, channels: usize) -> Result<()> {
    let _ = format;
    if channels == 0 {
        bail!("bad wav file, which channels is 0");
    }
    Ok(())
}

fn run(args: &Args) -> Result<()> {
    let (mut source, input_spec) = SourceReader::open(&args.input)?;
    let channels = input_spec.channels;
    let input_sr = input_spec.sample_rate;
    let output_sr = args.target_rate;
    let output_frames = get_output_frames(input_spec.frames as u64, input_sr, output_sr)?;
    let manager = create_manager(
        input_sr,
        output_sr,
        args.attenuation,
        args.quantify,
        args.pass_width,
        args.generic,
        &args.kernel,
    )?;
    let output_file = get_output_file(&args.input, &args.output, output_sr);
    println!("output file is {output_file:?}");
    println!("mode {:?} ratio {}", manager.mode(), manager.ratio());
    let mut sink = Sink::new(input_spec.format);
    let latency = manager.latency();
    let mut converters: Vec<_> = (0..channels).map(|_| manager.converter()).collect();

    let buf_len = (2 * latency).max(2048);
    let mut n = 0u64;
    let mut pending_skip = latency;

    while n < output_frames {
        let interleaved = source.read_frames(buf_len, &input_spec)?;
        let frames_read = interleaved.len() / channels;
        let mut channel_samples: Vec<Vec<f64>> =
            (0..channels).map(|_| Vec::with_capacity(buf_len)).collect();
        for frame in 0..buf_len {
            for (chan, buf) in channel_samples.iter_mut().enumerate() {
                // once the input runs out, feed zeros like the old
                // iterator + repeat(0.0) chain did
                let sample = if frame < frames_read {
                    interleaved[frame * channels + chan]
                } else {
                    0.0
                };
                buf.push(sample);
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
                &mut sink,
            )?;
            continue;
        }
        write_planar_frames(
            &channel_out,
            produced,
            &mut pending_skip,
            &mut n,
            output_frames,
            &mut sink,
        )?;
    }
    sink.save(&output_file, output_sr, channels)?;
    Ok(())
}

fn write_planar_frames(
    channel_out: &[Vec<f64>],
    produced: usize,
    pending_skip: &mut usize,
    n: &mut u64,
    output_frames: u64,
    sink: &mut Sink,
) -> Result<()> {
    let start = (*pending_skip).min(produced);
    *pending_skip -= start;
    let take = (produced - start).min((output_frames - *n) as usize);
    for i in start..start + take {
        for channel in channel_out {
            sink.push(channel[i]);
        }
    }
    *n += take as u64;
    Ok(())
}

fn get_output_frames(input_frames: u64, input_sr: u32, output_sr: u32) -> Result<u64> {
    if input_sr == output_sr {
        bail!("sample rate is same, no need to convert");
    }
    Ok(input_frames * output_sr as u64 / input_sr as u64)
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

    fn write_tone_wav(path: &Path, sr: u32, frames: u32, format: Format) {
        match format {
            Format::Float32 => {
                let samples: Vec<f32> = (0..frames)
                    .map(|i| ((i as f64 * 0.01).sin()) as f32)
                    .collect();
                wavers::write(path, &samples, sr as i32, 1).unwrap();
            }
            Format::Float64 => {
                let samples: Vec<f64> = (0..frames).map(|i| (i as f64 * 0.01).sin()).collect();
                wavers::write(path, &samples, sr as i32, 1).unwrap();
            }
            Format::Int16 => {
                let samples: Vec<i16> = (0..frames)
                    .map(|i| (((i as f64 * 0.01).sin()) * 10_000.0) as i16)
                    .collect();
                wavers::write(path, &samples, sr as i32, 1).unwrap();
            }
            Format::Int32 => {
                let samples: Vec<i32> = (0..frames)
                    .map(|i| (((i as f64 * 0.01).sin()) * 1_000_000.0) as i32)
                    .collect();
                wavers::write(path, &samples, sr as i32, 1).unwrap();
            }
            Format::Int24 => {
                let samples: Vec<i24> = (0..frames)
                    .map(|i| i24::from_i32((((i as f64 * 0.01).sin()) * 1_000_000.0) as i32))
                    .collect();
                wavers::write(path, &samples, sr as i32, 1).unwrap();
            }
        }
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
    fn detect_format_maps_all_wav_types() {
        assert_eq!(detect_format(WavType::Pcm16).unwrap(), Format::Int16);
        assert_eq!(detect_format(WavType::EPcm16).unwrap(), Format::Int16);
        assert_eq!(detect_format(WavType::Pcm24).unwrap(), Format::Int24);
        assert_eq!(detect_format(WavType::Pcm32).unwrap(), Format::Int32);
        assert_eq!(detect_format(WavType::Float32).unwrap(), Format::Float32);
        assert_eq!(detect_format(WavType::Float64).unwrap(), Format::Float64);
        assert_eq!(detect_format(WavType::EFloat64).unwrap(), Format::Float64);
    }

    #[test]
    fn check_spec_rejects_zero_channels() {
        assert!(check_spec(Format::Int16, 2).is_ok());
        assert!(check_spec(Format::Float64, 0).is_err());
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
        write_tone_wav(&input, 44100, 800, Format::Int16);
        let output = dir.join("out.wav");
        let a = args(input, Some(output.clone()), "linear", false);
        run(&a).unwrap();

        let reader = Wav::<i16>::from_path(&output).unwrap();
        assert_eq!(reader.sample_rate(), 48000);
        assert_eq!(reader.encoding(), WavType::Pcm16);
        // 800 * 48000 / 44100 truncated to 870 frames.
        assert_eq!(reader.n_samples(), 870);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_end_to_end_sinc_fast() {
        let dir = temp_dir("e2e_sinc");
        let input = dir.join("in.wav");
        write_tone_wav(&input, 44100, 800, Format::Int16);
        let output = dir.join("out.wav");
        let a = args(input, Some(output.clone()), "sinc", false);
        run(&a).unwrap();

        let reader = Wav::<i16>::from_path(&output).unwrap();
        assert_eq!(reader.sample_rate(), 48000);
        assert_eq!(reader.n_samples(), 870);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_end_to_end_float32_preserves_format() {
        let dir = temp_dir("e2e_float");
        let input = dir.join("in.wav");
        write_tone_wav(&input, 44100, 500, Format::Float32);
        let output = dir.join("out.wav");
        let a = args(input, Some(output.clone()), "sinc", false);
        run(&a).unwrap();

        let reader = Wav::<f32>::from_path(&output).unwrap();
        assert_eq!(reader.sample_rate(), 48000);
        assert_eq!(reader.encoding(), WavType::Float32);
        // 500 * 48000 / 44100 truncated to 544 frames.
        assert_eq!(reader.n_samples(), 544);
        let (samples, _) = wavers::read::<f32, _>(&output).unwrap();
        let max_abs = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(max_abs.is_finite() && max_abs <= 2.0, "level {max_abs}"); // sinc ripple ~5% over 1.0
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_end_to_end_float64_preserves_format() {
        let dir = temp_dir("e2e_float64");
        let input = dir.join("in.wav");
        write_tone_wav(&input, 44100, 500, Format::Float64);
        let output = dir.join("out.wav");
        let a = args(input, Some(output.clone()), "sinc", false);
        run(&a).unwrap();

        let reader = Wav::<f64>::from_path(&output).unwrap();
        assert_eq!(reader.sample_rate(), 48000);
        assert_eq!(reader.encoding(), WavType::Float64);
        assert_eq!(reader.n_samples(), 544);
        let (samples, _) = wavers::read::<f64, _>(&output).unwrap();
        let max_abs = samples.iter().fold(0.0f64, |m, &s| m.max(s.abs()));
        assert!(max_abs.is_finite() && max_abs <= 2.0, "level {max_abs}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_end_to_end_int24_round_trip() {
        let dir = temp_dir("e2e_int24");
        let input = dir.join("in.wav");
        write_tone_wav(&input, 44100, 500, Format::Int24);
        let output = dir.join("out.wav");
        let a = args(input, Some(output.clone()), "linear", false);
        run(&a).unwrap();

        let reader = Wav::<i24>::from_path(&output).unwrap();
        assert_eq!(reader.sample_rate(), 48000);
        assert_eq!(reader.encoding(), WavType::Pcm24);
        assert_eq!(reader.n_samples(), 544);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_end_to_end_int32_lsb_survives_round_trip() {
        // HA's bit-depth probe feeds signals as small as ±1 LSB. Truncating
        // on write zeroes almost all of them; rounding must keep them.
        let dir = temp_dir("e2e_lsb32");
        let input = dir.join("in.wav");
        let samples: Vec<i32> = (0..44100)
            .map(|i| (2.0 * (i as f64 * 0.01).sin()).round() as i32)
            .collect();
        let nonzero_in = samples.iter().filter(|&&s| s != 0).count();
        wavers::write(&input, &samples, 48000, 1).unwrap();
        let output = dir.join("out.wav");
        let mut a = args(input, Some(output.clone()), "linear", false);
        a.target_rate = 44100;
        run(&a).unwrap();

        let (out, _) = wavers::read::<i32, _>(&output).unwrap();
        let nonzero_out = out.iter().filter(|&&s| s != 0).count();
        assert!(nonzero_out > 0, "all samples collapsed to zero");
        // a nonzero fraction comparable to the input's (±30% for edge effects)
        let expected = nonzero_in * 44100 / 48000;
        assert!(
            nonzero_out * 100 > expected * 70,
            "nonzero {nonzero_out} vs expected ~{expected}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_rejects_same_rate_and_missing_input() {
        let dir = temp_dir("e2e_errors");
        let input = dir.join("in.wav");
        write_tone_wav(&input, 48000, 100, Format::Int16);
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
