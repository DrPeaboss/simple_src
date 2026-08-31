"""Plots for simple_src conversion results.

Interactive usage (also importable):
    python plots.py spectrum.wav
    python plots.py spectrogram sweep.wav
    import plots; plots.spectrum('out.wav')

Headless batch reporting (Agg backend, writes PNGs, no window):
    python plots.py --report [--quality-dir DIR] [--wav-dir DIR] [--out DIR]

The --report mode renders the comparison charts (spectrum overlays, filter
frequency response, passband zoom, spectrograms) as PNGs under
``<out>/extras/`` and writes a tiny ``extras.html`` index so the results can
be browsed from a single page alongside the Rust-generated ``report.html``
from the spectral tests.

Needs numpy, scipy and matplotlib.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import kaiser

# ---------------------------------------------------------------------------
# Existing interactive helpers (kept byte-for-byte compatible)
# ---------------------------------------------------------------------------


def _spectrum(fs, data, name, impulse: None | str = None):
    passband = impulse == 'passband'
    N = len(data)
    half_N = N // 2
    fft_data = abs(np.fft.fft(data))
    fft_data = fft_data / half_N if impulse is None else fft_data / max(fft_data)
    fft_dBFS = 20 * np.log10(fft_data)
    freqs = np.fft.fftfreq(N, 1 / fs)
    plt.figure(figsize=(6, 4))
    xticks = np.arange(0, fs // 2 + 1, 2000)
    xticklabels = [f'{int(tick / 1000)}' for tick in xticks]
    ymin, ymax, ystep = (-3, 1, 0.5) if passband else (-200, 10, 20)
    ax = plt.gca()
    ax.set(xlabel='Frequency in kHz', ylabel='Magnitude in dBFS',
           xlim=(0, fs // 2), ylim=(ymin, ymax),
           xticks=xticks, yticks=np.arange(ymin, ymax, ystep),
           xticklabels=xticklabels, facecolor='black')
    ax.plot(freqs[:half_N], fft_dBFS[:half_N], color='white')
    ax.grid()
    prefix = 'Passband of ' if passband else 'Spectrum of '
    plt.title(prefix + name)
    plt.show()


def spectrum(filename):
    fs, data = wavfile.read(filename)
    _spectrum(fs, data, filename)


def impulse(filename, passband=False):
    fs, data = wavfile.read(filename)
    _spectrum(fs, data, filename, impulse='passband' if passband else '')


def raw_impulse(filename, fs, passband=False):
    data = np.fromfile(filename, np.float64)
    _spectrum(fs, data, filename, impulse='passband' if passband else '')


def spectrogram(filename):
    fs, data = wavfile.read(filename)
    _spectrogram(fs, data, filename)


def raw_spectrogram(filename, fs):
    """Spectrogram of a raw little-endian f64 file (e.g. from the spectral
    baseline tests in target/tmp/quality/)."""
    data = np.fromfile(filename, np.float64)
    _spectrogram(fs, data, filename)


def _spectrogram(fs, data, name, out=None):
    N = len(data)
    window_size = 2048
    hop = window_size // 2
    win = kaiser(window_size, 20)
    SFT = ShortTimeFFT(win, hop, fs, scale_to='magnitude')
    Sx = SFT.stft(data)
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    yticks = np.arange(0, fs // 2 + 1, 2000)
    yticklabels = [f'{int(tick / 1000)}' for tick in yticks]
    ax.set(xlabel='Time in seconds', ylabel='Frequency in kHz',
           yticks=yticks, yticklabels=yticklabels)
    im = ax.imshow(20 * np.log10(abs(Sx)), origin='lower', aspect='auto',
                   extent=SFT.extent(N), cmap='inferno',
                   vmin=-200, vmax=1, interpolation='sinc')
    fig.colorbar(im, label="Magnitude in dBFS", ticks=np.arange(-200, 1, 20))
    plt.title(f'Spectrogram of {name}')
    if out:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# New report helpers (headless; used by --report)
# ---------------------------------------------------------------------------


def _load_csv(path, stride=1):
    """Read a spectral baseline CSV (freq_hz,dbfs) as (freqs, db)."""
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # header
        rows = [(float(r[0]), float(r[1])) for r in reader][::stride]
    xs = np.array([r[0] for r in rows])
    ys = np.array([r[1] for r in rows])
    return xs, ys


def overlay_spectra(csv_files, out=None, title=None):
    """Overlay several spectrum CSVs (e.g. generic vs fast) on one axis."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in csv_files:
        xs, ys = _load_csv(f)
        label = Path(f).stem
        ax.plot(xs / 1000.0, ys, lw=1.0, label=label)
    ax.set(xlabel='Frequency in kHz', ylabel='Magnitude in dBFS',
           xlim=(0, xs[-1] / 1000.0), ylim=(-200, 10))
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_title(title or 'Spectrum overlay')
    if out:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def passband_zoom(csv_file, out=None):
    """Zoom into the audible band of a flatness spectrum CSV."""
    xs, ys = _load_csv(csv_file)
    mask = xs < 22000
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs[mask] / 1000.0, ys[mask], lw=1.0)
    ax.set(xlabel='Frequency in kHz', ylabel='Magnitude in dBFS',
           xlim=(0, 22), ylim=(-40, -5))
    ax.grid(alpha=0.3)
    ax.set_title(f'Passband zoom · {Path(csv_file).stem}')
    if out:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def frequency_response(impulse_wav, fs, out=None):
    """|H(f)| of the conversion from an impulse-response wav (32-bit float,
    e.g. output/32bitfloat/impulse-32bitfloat.wav)."""
    _, data = wavfile.read(impulse_wav)
    x = np.asarray(data, dtype=np.float64)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    mag /= mag.max()
    db = 20 * np.log10(np.clip(mag, 1e-12, None))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs / 1000.0, db, lw=1.0)
    ax.set(xlabel='Frequency in kHz', ylabel='Magnitude in dB',
           xlim=(0, fs / 2 / 1000.0), ylim=(-240, 6))
    ax.grid(alpha=0.3)
    ax.set_title(f'|H(f)| · {Path(impulse_wav).stem} (fs={fs})')
    if out:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def report(quality_dir, wav_dir, out_dir):
    """Batch-render the comparison PNGs under ``out_dir/extras`` and write a
    small ``extras.html`` index page. Quality CSVs come from
    ``quality_dir`` (the spectral tests' artifact folder); impulse/sweep wavs
    come from ``wav_dir``."""
    out_dir = Path(out_dir)
    extras = out_dir / 'extras'
    extras.mkdir(parents=True, exist_ok=True)

    written = []

    def save(name):
        p = extras / name
        written.append(p.name)
        return str(p)

    # 1. generic vs fast THD+N overlay
    g96 = Path(quality_dir) / 'sinc_g96_thdn.csv'
    f96 = Path(quality_dir) / 'sinc_f96_thdn.csv'
    if g96.exists() and f96.exists():
        overlay_spectra([g96, f96], out=save('overlay_thdn.png'),
                        title='generic vs fast · THD+N spectrum (44100 → 48000)')

    # 2. generic vs fast alias overlay
    ag = Path(quality_dir) / 'sinc_g96_alias.csv'
    af = Path(quality_dir) / 'sinc_f96_alias.csv'
    if ag.exists() and af.exists():
        overlay_spectra([ag, af], out=save('overlay_alias.png'),
                        title='generic vs fast · alias spectrum (48000 → 44100)')

    # 3. passband zoom of the flatness spectra
    for stem in ['sinc_f96_flat', 'sinc_g96_flat', 'cubic_flat']:
        csvf = Path(quality_dir) / f'{stem}.csv'
        if csvf.exists():
            passband_zoom(csvf, out=save(f'passband_{stem}.png'))

    # 4. filter frequency responses from impulse wavs
    imp_dir = Path(wav_dir)
    for f in sorted(imp_dir.glob('impulse*.wav')):
        try:
            sr, _ = wavfile.read(f)
        except Exception:
            continue
        frequency_response(f, sr, out=save(f'fresponse_{f.stem}.png'))

    # 5. sweep spectrograms
    sweep_out = Path(quality_dir) / 'sweep_441_480_out.f64'
    if sweep_out.exists():
        data = np.fromfile(sweep_out, np.float64)
        _spectrogram(48000, data, 'sweep 44100 → 48000', out=save('spectrogram_sweep_441_480.png'))
    sweep_in = Path(quality_dir) / 'sweep_441_480_in.f64'
    if sweep_in.exists():
        data = np.fromfile(sweep_in, np.float64)
        _spectrogram(44100, data, 'sweep input 44100', out=save('spectrogram_sweep_441_480_in.png'))

    # 6. extras index page (single-file view of the PNGs)
    if written:
        rows = '\n'.join(
            f'<figure><figcaption>{name}</figcaption>'
            f'<img src="extras/{name}" style="max-width:100%"></figure>'
            for name in written
        )
        html = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<title>simple_src · plots extras</title></head><body>'
            '<h1>simple_src · plots extras</h1>'
            f'<p>Rendered by <code>plots.py --report</code>. '
            f'See also <a href="report.html">report.html</a> from the Rust '
            f'spectral tests.</p>{rows}</body></html>'
        )
        (out_dir / 'extras.html').write_text(html)

    print(f'wrote {len(written)} PNGs to {extras}')
    for name in written:
        print('  ' + name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', action='store_true',
                    help='headless batch mode: render comparison PNGs + extras.html')
    ap.add_argument('--quality-dir', default=None,
                    help='spectral CSV/SVG folder (default: $CARGO_TARGET_TMPDIR/quality, '
                         'falling back to ./quality)')
    ap.add_argument('--wav-dir', default=None,
                    help='wav folder with impulse-*.wav (default: ./output/32bitfloat)')
    ap.add_argument('--out', default='output/report',
                    help='output folder for the report + extras (default: output/report)')
    args = ap.parse_args(argv)

    if not args.report:
        # legacy interactive single-shot: plots.py spectrum.wav
        if len(sys.argv) < 2:
            ap.print_help()
            return 1
        kind, path = sys.argv[1], sys.argv[2:]
        if kind == 'spectrogram' and path:
            spectrogram(path[0])
        elif path:
            spectrum(path[0])
        return 0

    import matplotlib
    matplotlib.use('Agg')

    quality_dir = args.quality_dir
    if quality_dir is None:
        candidate = os.environ.get('CARGO_TARGET_TMPDIR')
        if candidate:
            quality_dir = os.path.join(candidate, 'quality')
        else:
            quality_dir = './quality'
    wav_dir = args.wav_dir or './output/32bitfloat'
    report(quality_dir, wav_dir, args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())