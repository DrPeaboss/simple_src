import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import kaiser

def _spectrum(fs, data, name, impulse: None | str = None):
    passband = impulse == 'passband'
    N = len(data)
    half_N = N // 2
    fft_data = abs(np.fft.fft(data))
    fft_data = fft_data / half_N if impulse is None else fft_data / max(fft_data)
    fft_dBFS = 20 * np.log10(fft_data)
    freqs = np.fft.fftfreq(N, 1/fs)
    plt.figure(figsize=(6, 4))
    xticks = np.arange(0, fs // 2 + 1, 2000)
    xticklabels = [f'{int(tick/1000)}' for tick in xticks]
    ymin, ymax, ystep = (-3, 1, 0.5) if passband else (-200, 10, 20)
    ax = plt.gca()
    ax.set(xlabel='Frequency in kHz', ylabel='Magnitude in dBFS',
           xlim=(0, fs//2), ylim=(ymin, ymax),
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
    N = len(data)
    window_size = 2048
    hop = window_size // 2
    win = kaiser(window_size, 20)
    SFT = ShortTimeFFT(win, hop, fs, scale_to='magnitude')
    Sx = SFT.stft(data)
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    yticks = np.arange(0, fs // 2 + 1, 2000)
    yticklabels = [f'{int(tick/1000)}' for tick in yticks]
    ax.set(xlabel='Time in seconds', ylabel='Frequency in kHz',
        yticks=yticks, yticklabels=yticklabels)
    im = ax.imshow(20*np.log10(abs(Sx)), origin='lower', aspect='auto',
                     extent=SFT.extent(N), cmap='inferno',
                     vmin=-200, vmax=1,
                     interpolation='sinc')
    fig.colorbar(im, label="Magnitude in dBFS", ticks=np.arange(-200,1,20))
    plt.title(f'Spectrogram of {filename}')
    plt.show()
