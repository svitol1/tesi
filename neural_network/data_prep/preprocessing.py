"""
Modulo di preprocessing per segnali ECG.

Fornisce funzioni per:
- rimozione del baseline wander (mediana o passa-alto)
- rimozione dell'interferenza di rete (filtro notch)
- filtraggio passa-banda (default 0.5-40 Hz)
"""

from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt

__all__ = [
	"remove_baseline",
	"notch_filter",
	"bandpass_filter",
	"preprocess_ecg",
]


def _ensure_odd(n: int) -> int:
	return int(n) if int(n) % 2 == 1 else int(n) + 1


def _apply_per_channel(signal: np.ndarray, fn) -> np.ndarray:
	x = np.asarray(signal)
	if x.ndim == 1:
		return fn(x)
	if x.ndim == 2:
		return np.stack([fn(channel) for channel in x], axis=0)
	raise ValueError(f"Expected a 1D or 2D ECG array, got shape={x.shape}")


def remove_baseline(signal: np.ndarray, fs: float, method: str = "median") -> np.ndarray:
	"""Rimuove il baseline wander dal segnale.

	Args:
		signal: segnale 1D
		fs: frequenza di campionamento (Hz)
		method: 'median' (default) o 'highpass'

	Returns:
		Segnale con baseline rimosso.
	"""
	x = np.asarray(signal)
	if method == "median":
		def _median_baseline(x1d: np.ndarray) -> np.ndarray:
			# Approccio simile a Pan-Tompkins: due median filter con finestre diverse
			w1 = _ensure_odd(max(3, int(0.2 * fs)))
			w2 = _ensure_odd(max(3, int(0.6 * fs)))
			if x1d.size < w2:
				# segnale troppo corto: sottrai mediana globale
				return x1d - np.median(x1d)
			baseline = medfilt(x1d, kernel_size=w1)
			baseline = medfilt(baseline, kernel_size=w2)
			return x1d - baseline

		return _apply_per_channel(x, _median_baseline)
	if method == "highpass":
		# High-pass Butterworth a 0.5 Hz per rimuovere variazioni lente
		cutoff = 0.5
		nyq = 0.5 * fs
		b, a = butter(2, cutoff / nyq, btype="highpass")
		return filtfilt(b, a, x, axis=-1)
	raise ValueError("method must be 'median' or 'highpass'")


def notch_filter(signal: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
	"""Applica un filtro notch per rimuovere l'interferenza di rete.

	Args:
		signal: segnale 1D
		fs: frequenza di campionamento (Hz)
		freq: frequenza di rete (50 o 60 Hz)
		q: fattore di qualità del notch (più alto => banda più stretta)

	Returns:
		Segnale filtrato.
	"""
	x = np.asarray(signal)
	if freq <= 0 or freq >= fs / 2:
		return x
	w0 = freq / (0.5 * fs)
	b, a = iirnotch(w0, q)
	return filtfilt(b, a, x, axis=-1)


def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 4) -> np.ndarray:
	"""Filtra il segnale con un passa-banda Butterworth.

	Args:
		signal: segnale 1D
		fs: frequenza di campionamento (Hz)
		low: cut-in (Hz)
		high: cut-off (Hz)
		order: ordine del filtro

	Returns:
		Segnale filtrato.
	"""
	x = np.asarray(signal)
	nyq = 0.5 * fs
	low = max(low, 0.0)
	high = min(high, nyq - 1e-6)
	if low >= high:
		raise ValueError("low must be < high and within (0, fs/2)")
	b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
	return filtfilt(b, a, x, axis=-1)


def preprocess_ecg(
	signal: np.ndarray,
	fs: float,
	remove_baseline_method: Optional[str] = "median",
	apply_notch: bool = True,
	notch_freq: float = 50.0,
	apply_bandpass: bool = True,
	band: Tuple[float, float] = (0.5, 40.0),
) -> np.ndarray:
	"""Pipeline di preprocessing per ECG.

	Ordine di esecuzione:
	1. Rimozione del baseline wander
	2. Rimozione dell'interferenza di rete (notch) se `apply_notch` è True
	3. Filtraggio passa-banda se `apply_bandpass` è True

	Args:
		signal: segnale 1D
		fs: frequenza di campionamento (Hz)
		remove_baseline_method: 'median', 'highpass' o None per saltare
		apply_notch: applicare filtro notch (default True)
		notch_freq: 50 o 60 Hz tipicamente
		apply_bandpass: applicare passa-banda (default True)
		band: tupla (low, high) in Hz

	Returns:
		Segnale preprocessato (numpy array float64)
	"""
	x = np.asarray(signal, dtype=float)
	if x.ndim not in (1, 2):
		raise ValueError(f"Expected a 1D or 2D ECG array, got shape={x.shape}")
	if remove_baseline_method is not None:
		x = remove_baseline(x, fs, method=remove_baseline_method)
	if apply_notch and notch_freq is not None and notch_freq > 0:
		x = notch_filter(x, fs, freq=notch_freq)
	if apply_bandpass and band is not None:
		low, high = band
		x = bandpass_filter(x, fs, low=low, high=high)
	return x