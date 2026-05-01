from pathlib import Path
import winsound

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


def play_wav_file(wav_path: str | Path):
	"""Play a WAV file using Windows' built-in sound player."""
	winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)


def read_wav_file(wav_path: str | Path):
	"""Read a WAV file and return sample rate and data array."""
	sample_rate, data = wavfile.read(wav_path)
	return sample_rate, data


def to_mono_float(data: np.ndarray) -> np.ndarray:
	"""Convert audio to mono float64."""
	if data.ndim > 1:
		signal = data.mean(axis=1)
	else:
		signal = data

	return signal.astype(np.float64)


def iq_demodulate(sample_rate, data, carrier_hz=144_000.0,
                  lowpass_hz=15_000.0, filter_order=6):
    signal = to_mono_float(data)
    signal = signal - np.mean(signal)

    t = np.arange(signal.size) / sample_rate
    lo = np.exp(-1j * 2 * np.pi * carrier_hz * t)
    baseband_complex = signal * lo

    nyquist = sample_rate / 2
    if lowpass_hz <= 0 or lowpass_hz >= nyquist:
        raise ValueError(
            f"Low-pass cutoff måste vara mellan 0 och Nyquist ({nyquist:.1f} Hz)."
        )

    sos = butter(filter_order, lowpass_hz / nyquist, btype="low", output="sos")
    i_hat = sosfiltfilt(sos, np.real(baseband_complex))
    q_hat = sosfiltfilt(sos, np.imag(baseband_complex))

    return i_hat, q_hat

def rotate_iq(i_hat: np.ndarray, q_hat: np.ndarray, delta: float):
    """
    Implementerar (A-88):
    [x_I; x_Q] = R(delta) [i_hat; q_hat]
    där R = [[cos, -sin], [sin, cos]].
    """
    c = np.cos(delta)
    s = np.sin(delta)

    x_I = c * i_hat - s * q_hat
    x_Q = s * i_hat + c * q_hat
    return x_I, x_Q

def save_audio_wav(audio: np.ndarray, sample_rate: int, output_path: Path) -> Path:
    audio = np.asarray(audio, dtype=np.float64)
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wavfile.write(output_path, sample_rate, audio_int16)
    return output_path


def plot_fft(sample_rate: int, signal: np.ndarray, max_plot_hz: float = 20_000.0):
    """Plotta FFT-spektrum för en given signal."""
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)

    n = len(signal)
    window = np.hanning(n)
    fft_vals = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(n, d=1 / sample_rate)
    magnitude = np.abs(fft_vals)

    mask = freqs <= max_plot_hz

    plt.figure(figsize=(10, 5))
    plt.plot(freqs[mask], magnitude[mask], linewidth=1)
    plt.title("FFT för signalen vid delta = 0.8")
    plt.xlabel("Frekvens (Hz)")
    plt.ylabel("Magnitud")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    wav_path = Path(__file__).with_name("signal-NiBl88.wav")

    if not wav_path.exists():
        raise FileNotFoundError(f"File not found: {wav_path}")

    sample_rate, data = read_wav_file(wav_path)

    i_hat, q_hat = iq_demodulate(
        sample_rate,
        data,
        carrier_hz=144_000.0,
        lowpass_hz=15_000.0,
    )

    d = 0.8
    x_I, x_Q = rotate_iq(i_hat, q_hat, d)

    # Välj x_I som signal att spara och analysera.
    audio1 = x_I
    audio2 = x_Q
    out_name1 = f"Idemod_delta_{d:.2f}.wav"
    out_name2 = f"Qdemod_delta_{d:.2f}.wav"
    
    out_path1 = Path(__file__).with_name(out_name1)
    out_path2 = Path(__file__).with_name(out_name2)
    
    save_audio_wav(audio1, sample_rate, out_path1)
    save_audio_wav(audio2, sample_rate, out_path2)
    
    print(f"delta={d:.2f} rad sparad som {out_name1}")
    print(f"delta={d:.2f} rad sparad som {out_name2}")

    plot_fft(sample_rate, audio1, max_plot_hz=20_000.0)
    plot_fft(sample_rate, audio2, max_plot_hz=20_000.0)
