"""
Utility condivise tra precompute_cache.py (fase 1) e
dataset_builder.py (fase 2).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

try:
    from .pan_tompkins_algo import detect_r_peaks_pan_tompkins
    from .preprocessing import preprocess_ecg
except ImportError:
    from pan_tompkins_algo import detect_r_peaks_pan_tompkins
    from preprocessing import preprocess_ecg

SAMPLING_RATE = 500
LEAD_NAMES_PTBXL = ["I", "II", "III", "AVR", "AVL", "AVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAMES_GEORGIA = ["I", "II", "III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"]
PRE_MS = 400.0
POST_MS = 600.0


# --------------------------------------------------------------------------
# Rilevamento picchi R (invariato rispetto all'originale)
# --------------------------------------------------------------------------

def filter_r_peaks_by_v1_v5_window(r_peaks_v1, r_peaks_v5, fs, tolerance_ms=50.0):
    if len(r_peaks_v1) == 0 or len(r_peaks_v5) == 0:
        return np.array([], dtype=int)
    tolerance_samples = max(1, int(round(tolerance_ms / 1000.0 * fs)))
    r_peaks_v1 = np.asarray(r_peaks_v1, dtype=int)
    r_peaks_v5 = np.asarray(r_peaks_v5, dtype=int)
    paired, used_v5 = [], set()
    for peak_v1 in r_peaks_v1:
        candidates = np.where(np.abs(r_peaks_v5 - peak_v1) <= tolerance_samples)[0]
        if len(candidates) == 0:
            continue
        nearest_idx = int(candidates[np.argmin(np.abs(r_peaks_v5[candidates] - peak_v1))])
        if nearest_idx in used_v5:
            continue
        used_v5.add(nearest_idx)
        paired.append(int(round((peak_v1 + r_peaks_v5[nearest_idx]) / 2.0)))
    return np.unique(np.asarray(paired, dtype=int))


def detect_r_peaks_v1_v5_average(signal_12lead, lead_names, fs=SAMPLING_RATE):
    try:
        v1_idx = lead_names.index("V1")
        v5_idx = lead_names.index("V5")
        r_v1 = detect_r_peaks_pan_tompkins(signal_12lead[:, v1_idx], fs)
        r_v5 = detect_r_peaks_pan_tompkins(signal_12lead[:, v5_idx], fs)
        return filter_r_peaks_by_v1_v5_window(r_v1, r_v5, fs)
    except Exception:
        return np.array([], dtype=int)


def segment_beats(signal_12lead, r_peaks, fs=SAMPLING_RATE, pre_ms=PRE_MS, post_ms=POST_MS):
    pre_samples = int(round(pre_ms / 1000.0 * fs))
    post_samples = int(round(post_ms / 1000.0 * fs))
    window_len = pre_samples + post_samples
    beats, used_peaks = [], []
    for r in r_peaks:
        start, end = int(r) - pre_samples, int(r) + post_samples
        if start < 0 or end > signal_12lead.shape[0]:
            continue
        beats.append(signal_12lead[start:end, :])
        used_peaks.append(int(r))
    if not beats:
        return np.empty((0, window_len, 12), dtype=np.float32), np.array([], dtype=int)
    return np.stack(beats).astype(np.float32), np.asarray(used_peaks, dtype=int)


# --------------------------------------------------------------------------
# Caricamento registrazioni grezze (usato SOLO da precompute_cache.py)
# --------------------------------------------------------------------------

def load_ptbxl_metadata(ptbxl_root: str) -> pd.DataFrame:
    path = os.path.join(ptbxl_root, "ptbxl_database.csv")
    return pd.read_csv(path, index_col="ecg_id")


def discover_records_georgia(georgia_root: str):
    records_files = sorted(Path(georgia_root).rglob("RECORDS"))
    if not records_files:
        raise FileNotFoundError(f"Nessun file RECORDS trovato sotto {georgia_root}")
    records = []
    seen = set()
    for records_file in records_files:
        with records_file.open(encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if not name or name.startswith("#"):
                    continue
                record_path = records_file.parent / name
                key = str(record_path)
                if key not in seen:
                    seen.add(key)
                    records.append((Path(name).name, key))
    return records


def load_and_preprocess_ptbxl(ptbxl_root, row, notch_freq=50.0):
    record_path = os.path.join(ptbxl_root, row["filename_hr"])
    signal, meta = wfdb.rdsamp(record_path)
    if meta["sig_name"] != LEAD_NAMES_PTBXL:
        raise ValueError(f"Ordine derivazioni sbagliato: {meta['sig_name']}")
    signal = signal.astype(np.float32)
    for lead_idx in range(signal.shape[1]):
        signal[:, lead_idx] = preprocess_ecg(signal[:, lead_idx], fs=SAMPLING_RATE, notch_freq=notch_freq)
    return signal


def load_and_preprocess_georgia(record_path, notch_freq=60.0):
    signal, meta = wfdb.rdsamp(record_path)
    if int(meta["fs"]) != SAMPLING_RATE:
        raise ValueError(f"Frequenza WFDB inattesa: {meta['fs']}")
    if meta["sig_name"] != LEAD_NAMES_GEORGIA:
        raise ValueError(f"Ordine derivazioni sbagliato: {meta['sig_name']}")
    signal = signal.astype(np.float32)
    for lead_idx in range(signal.shape[1]):
        signal[:, lead_idx] = preprocess_ecg(signal[:, lead_idx], fs=SAMPLING_RATE, notch_freq=notch_freq)
    return signal


# --------------------------------------------------------------------------
# Split train/val/test (usato SOLO da dataset_builder.py)
# --------------------------------------------------------------------------

def split_ids(ids: list, train_frac: float, val_frac: float, seed: int):
    """Split casuale riproducibile di una lista di identificativi
    (pazienti/registrazioni) in train/val/test."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(ids)))  # sort per riproducibilita' indipendente dall'ordine di scoperta
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()


# --------------------------------------------------------------------------
# Registrazioni da escludere (confermate malposizionate da un giro precedente)
# --------------------------------------------------------------------------

def load_excluded_records(csv_dir: str, filename_pattern: str = "mislabels.csv") -> set:
    """Cerca ricorsivamente dentro `csv_dir` tutti i file che si chiamano
    `filename_pattern` (colonne 'ecg_id' e 'confirmed') e ritorna l'insieme
    dei record_key con confirmed == True, unendo tutti i file trovati."""
    csv_paths = sorted(Path(csv_dir).rglob(filename_pattern))
    if not csv_paths:
        raise FileNotFoundError(
            f"Nessun file '{filename_pattern}' trovato ricorsivamente in {csv_dir}"
        )

    excluded = set()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        missing = {"ecg_id", "confirmed"} - set(df.columns)
        if missing:
            raise ValueError(f"Il file {csv_path} non contiene le colonne richieste: {missing}")

        confirmed = df["confirmed"]
        if confirmed.dtype != bool:
            confirmed = confirmed.astype(str).str.strip().str.lower().isin(["true", "1"])

        excluded.update(df.loc[confirmed, "ecg_id"].astype(str).tolist())

    return excluded


# --------------------------------------------------------------------------
# Estrazione battiti train/val a partire da picchi GIA' rilevati
# --------------------------------------------------------------------------

def select_train_val_peaks(r_peaks):
    """Seleziona il 3° e 5° picco R (indici 2 e 4) dall'array di picchi
    gia' rilevato in fase di caching. Ritorna None se non disponibili
    (< 5 picchi rilevati in totale)."""
    if len(r_peaks) < 5:
        return None
    return np.array([r_peaks[2], r_peaks[4]], dtype=int)


# --------------------------------------------------------------------------
# Assegnazione classi bilanciate ai battiti di train/val (2 per registrazione)
# --------------------------------------------------------------------------

def assign_classes_per_recording(record_keys: list, classes: list, weights: list, seed: int):
    """Per ogni registrazione (record_key univoco), assegna 2 classi
    DISTINTE (una per battito) tra quelle disponibili, con probabilita'
    proporzionali a `weights`. Ritorna record_key -> [classe_0, classe_1]."""
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    assignment = {}
    for key in sorted(record_keys):  # ordine deterministico per riproducibilita'
        chosen = rng.choice(classes, size=2, replace=False, p=weights)
        assignment[key] = chosen.tolist()
    return assignment
