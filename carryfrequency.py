from pathlib import Path
import winsound

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def play_wav_file(wav_path: str | Path):
	"""Play a WAV file using Windows' built-in sound player."""
	winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)


def read_wav_file(wav_path: str | Path):
	"""Read a WAV file and return the sample rate and audio data."""
	return wavfile.read(wav_path)


def to_mono_float(data: np.ndarray) -> np.ndarray:
	"""Convert audio data to mono float64."""
	if data.ndim > 1:
		data = data.mean(axis=1)
	return data.astype(np.float64)


def plot_fft(sample_rate: int, data: np.ndarray, max_plot_hz: float | None = None):
	"""Plot the FFT magnitude spectrum of an audio signal."""
	signal = to_mono_float(data)
	signal = signal - np.mean(signal)

	n = signal.size
	window = np.hanning(n)
	windowed = signal * window

	fft_vals = np.fft.rfft(windowed)
	freqs = np.fft.rfftfreq(n, d=1 / sample_rate)
	magnitude = np.abs(fft_vals)

	if max_plot_hz is not None:
		mask = freqs <= max_plot_hz
		freqs = freqs[mask]
		magnitude = magnitude[mask]

	plt.figure(figsize=(10, 5))
	plt.plot(freqs, magnitude, linewidth=1)
	plt.title("FFT-spektrum för ljudfilen")
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
	print(f"Sample rate: {sample_rate} Hz")
	print(f"Data shape: {data.shape}")
	print(f"Data type: {data.dtype}")

	plot_fft(sample_rate, data, max_plot_hz=200_000.0)
