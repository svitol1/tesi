"""
Modulo di rilevamento dei picchi R usando l'algoritmo Pan-Tompkins.
"""

import numpy as np

__all__ = [
	"detect_r_peaks_pan_tompkins",
]


def _lowpass_filter(x: np.ndarray) -> np.ndarray:
	"""Filtro passa-basso Pan-Tompkins (ordine 2).

	H(z) = (1 - z^-6)^2 / (1 - z^-1)^2
	Implementazione come equazione alle differenze:
	y[n] = 2y[n-1] - y[n-2] + x[n] - 2x[n-6] + x[n-12]
	"""
	y = np.zeros_like(x)
	for n in range(len(x)):
		y[n] = (2 * y[n-1] if n >= 1 else 0) \
		     - (y[n-2] if n >= 2 else 0) \
		     + x[n] \
		     - (2 * x[n-6] if n >= 6 else 0) \
		     + (x[n-12] if n >= 12 else 0)
	return y


def _highpass_filter(x: np.ndarray) -> np.ndarray:
	"""Filtro passa-alto Pan-Tompkins.

	Ottenuto sottraendo il passa-basso da un segnale ritardato:
	y[n] = x[n-16] - (1/32) * (passa_basso)[n]
	"""
	lp = _lowpass_filter(x)
	y = np.zeros_like(x)
	for n in range(len(x)):
		delayed = x[n-16] if n >= 16 else 0
		y[n] = delayed - lp[n] / 32.0
	return y


def _derivative_filter(x: np.ndarray, fs: float) -> np.ndarray:
	"""Derivata (approssimazione a 5 punti).

	y[n] = (1/(8/fs)) * (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])
	"""
	y = np.zeros_like(x)
	for n in range(2, len(x) - 2):
		y[n] = (1 / (8 / fs)) * (
			-x[n-2] - 2*x[n-1] + 2*x[n+1] + x[n+2]
		)
	return y


def _moving_window_integration(x: np.ndarray, fs: float, window_ms: float = 150.0) -> np.ndarray:
	"""Integrazione a finestra mobile (MWI).

	Finestra di 150 ms → W = int(0.15 * fs) campioni
	y[n] = (1/W) * sum_{k=0}^{W-1} x[n-k]
	"""
	W = int(window_ms * fs / 1000)
	y = np.convolve(x, np.ones(W) / W, mode='same')
	return y


def _find_r_peaks(integrated: np.ndarray, fs: float) -> np.ndarray:
	"""Rilevamento picchi R con soglia adattiva.

	La soglia viene aggiornata dinamicamente basandosi sui livelli
	di segnale e rumore stimati.
	"""
	# Stima iniziale della soglia sul primo secondo
	init_window = integrated[:fs]
	signal_level = np.max(init_window)
	noise_level = np.mean(init_window)
	threshold = noise_level + 0.25 * (signal_level - noise_level)

	r_peaks = []
	min_distance = int(0.2 * fs)  # refrattarietà minima: 200 ms
	last_peak = -min_distance

	i = 1
	while i < len(integrated) - 1:
		# Ricerca di massimi locali
		if integrated[i] > integrated[i-1] and integrated[i] > integrated[i+1]:
			if integrated[i] > threshold and (i - last_peak) > min_distance:
				r_peaks.append(i)
				last_peak = i
				# Aggiornamento soglia adattiva
				signal_level = 0.125 * integrated[i] + 0.875 * signal_level
			else:
				noise_level = 0.125 * integrated[i] + 0.875 * noise_level
			threshold = noise_level + 0.25 * (signal_level - noise_level)
		i += 1

	return np.asarray(r_peaks, dtype=int)


def _refine_peaks(signal: np.ndarray, peaks: np.ndarray, window: int = 10) -> np.ndarray:
	"""Raffinamento dei picchi: ricerca del massimo nel segnale originale.

	Ricerca in una finestra di ±window campioni attorno a ogni picco rilevato.
	"""
	refined = []
	for p in peaks:
		start = max(0, p - window)
		end = min(len(signal), p + window)
		local_max = np.argmax(signal[start:end]) + start
		refined.append(local_max)
	return np.asarray(refined, dtype=int)


def detect_r_peaks_pan_tompkins(signal: np.ndarray, fs: float) -> np.ndarray:
	"""Rileva i picchi R usando l'algoritmo Pan-Tompkins.

	Pipeline completa:
	1. Filtraggio passa-banda (passa-basso + passa-alto)
	2. Derivata
	3. Elevazione al quadrato
	4. Integrazione a finestra mobile
	5. Rilevamento picchi con soglia adattiva
	6. Raffinamento

	Args:
		signal: segnale 1D (ECG)
		fs: frequenza di campionamento (Hz)

	Returns:
		Array di indici campione dei picchi R rilevati.
	"""
	x = np.asarray(signal, dtype=float)

	# Passo 1: filtraggio passa-banda
	bandpass = _highpass_filter(x)

	# Passo 2: derivata
	derived = _derivative_filter(bandpass, fs)

	# Passo 3: elevazione al quadrato
	squared = derived ** 2

	# Passo 4: integrazione a finestra mobile
	integrated = _moving_window_integration(squared, fs)

	# Passo 5: rilevamento picchi R
	r_peaks = _find_r_peaks(integrated, fs)

	# Passo 6: raffinamento
	r_peaks_refined = _refine_peaks(x, r_peaks)

	return r_peaks_refined
