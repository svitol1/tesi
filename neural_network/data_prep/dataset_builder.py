"""
Costruzione del dataset combinato PTB-XL + Georgia per l'addestramento
e la valutazione del modello di rilevamento malposizionamento elettrodi.

Split (per registrazione/paziente, seed fisso), applicato SEPARATAMENTE
a PTB-XL e Georgia e poi unito:
    - train : 80% delle registrazioni di ciascun dataset
    - val   : 10% delle registrazioni di ciascun dataset
    - test  : 10% delle registrazioni di ciascun dataset (il resto)

Per PTB-XL lo split viene fatto per PATIENT_ID (non per ecg_id), perché
uno stesso paziente puo' avere piu' registrazioni: split per ecg_id
causerebbe data leakage tra train/val/test. Per Georgia patient_id ==
ecg_id (una registrazione per paziente), quindi split per ecg_id e'
equivalente e sicuro.

TRAIN / VAL
-----------
Stessa logica di beat_dataset_builder.py + misplacement_dataset_builder.py,
applicata sia a PTB-XL sia a Georgia: per ogni registrazione si estraggono
ESATTAMENTE 2 battiti (3° e 5° picco R medio V1/V5), e si assegna una
classe diversa a ciascuno dei 2 battiti (bilanciata, seed fisso), tra
"normal" e i tipi di malposizionamento definiti in lead_misplacement.py.
Viene salvata solo la versione del battito corrispondente alla classe
assegnata (mai sia originale che trasformata dello stesso segmento).
L'assegnazione delle classi avviene su TUTTE le registrazioni (PTB-XL +
Georgia) insieme, cosi' che le proporzioni richieste in --class_weights
valgano sul dataset combinato.

TEST
----
Costruito in modo diverso, perché la valutazione avviene per INTERA
REGISTRAZIONE con majority voting (vedi test/test.py): si estraggono TUTTI
i battiti validi di ogni registrazione test (non solo 2), etichettati
"normal" senza alcuna trasformazione: sono registrazioni reali,
presumibilmente correttamente posizionate.

Output
------
<output_dir>/train/segments/*.npy + <output_dir>/train/final_dataset_index.csv
<output_dir>/val/segments/*.npy   + <output_dir>/val/final_dataset_index.csv
<output_dir>/test/segments/*.npy  + <output_dir>/test/beats_index.csv

Ogni file .npy ha shape (12, window_len), leads-first, pronto per
ECGGraphDataset senza bisogno di trasposizioni successive.

Uso
---
python dataset_builder.py \
    --ptbxl_root /path/to/ptbxl \
    --georgia_root /path/to/georgia \
    --output_dir /path/to/combined_dataset \
    --classes normal RA_LA LA_LL RA_LL V1_V2 \
    --class_weights 0.5 0.125 0.125 0.125 0.125
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from pan_tompkins_algo import detect_r_peaks_pan_tompkins
from preprocessing import preprocess_ecg
from lead_misplacement import apply_transform, MISPLACEMENT_TRANSFORMS

SAMPLING_RATE = 500
LEAD_NAMES_PTBXL = ["I", "II", "III", "AVR", "AVL", "AVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAMES_GEORGIA = ["I", "II", "III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"]
FILENAME_DIGITS = 7
PRE_MS = 400.0
POST_MS = 600.0


# --------------------------------------------------------------------------
# Utility condivise (rilevamento picchi, segmentazione)
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
# Caricamento registrazioni + split train/val/test
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
# Caricamento + preprocessing per singola registrazione
# --------------------------------------------------------------------------

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
# Estrazione battiti: 2 per registrazione (train/val) o tutti (test)
# --------------------------------------------------------------------------

def extract_train_val_beats(signal, lead_names):
    """Estrae ESATTAMENTE 2 battiti (3° e 5° picco), come in
    build_ptbxl_beat_dataset.py. Ritorna None se non disponibili."""
    r_peaks = detect_r_peaks_v1_v5_average(signal, lead_names)
    if len(r_peaks) < 5:
        return None
    selected = np.array([r_peaks[2], r_peaks[4]], dtype=int)
    beats, used_peaks = segment_beats(signal, selected)
    if beats.shape[0] != 2:
        return None
    return beats, used_peaks  # beats: (2, window_len, 12) leads-last


def extract_test_beats(signal, lead_names):
    """Estrae TUTTI i battiti validi della registrazione (per majority
    voting a livello di registrazione in fase di test)."""
    r_peaks = detect_r_peaks_v1_v5_average(signal, lead_names)
    if len(r_peaks) == 0:
        return None
    beats, used_peaks = segment_beats(signal, r_peaks)
    if beats.shape[0] == 0:
        return None
    return beats, used_peaks


# --------------------------------------------------------------------------
# Assegnazione classi bilanciate ai battiti di train/val (2 per registrazione)
# --------------------------------------------------------------------------

def assign_classes_per_recording(record_keys: list, classes: list, weights: list, seed: int):
    """Per ogni registrazione (identificata da record_key univoco),
    assegna 2 classi DISTINTE (una per battito) tra quelle disponibili,
    con probabilita' proporzionali a `weights`. Ritorna un dict
    record_key -> [classe_battito_0, classe_battito_1]."""
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    assignment = {}
    for key in sorted(record_keys):  # ordine deterministico per riproducibilita'
        chosen = rng.choice(classes, size=2, replace=False, p=weights)
        assignment[key] = chosen.tolist()
    return assignment


# --------------------------------------------------------------------------
# Pipeline principale
# --------------------------------------------------------------------------

def process_split(split_name, ptbxl_root, ptbxl_meta, ptbxl_ecg_ids,
                   georgia_records, georgia_ids_set,
                   output_dir, classes=None, class_weights=None, seed=42):
    """Processa un intero split (train / val / test) per entrambi i dataset
    e salva segmenti + indice CSV."""
    out_root = os.path.join(output_dir, split_name)
    segments_dir = os.path.join(out_root, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    is_test = split_name == "test"
    rows = []
    beat_counter = 0
    skipped = []

    georgia_subset = [(name, path) for name, path in georgia_records if name in georgia_ids_set]

    if not is_test:
        # ---- Prima fase: raccogliamo 2 battiti grezzi per registrazione
        # (PTB-XL + Georgia insieme), poi assegnamo le classi in modo
        # bilanciato su TUTTE le registrazioni di questo split.
        raw_beats = {}  # record_key -> (beats, used_peaks, source, patient_id)

        for ecg_id in tqdm(ptbxl_ecg_ids, desc=f"[{split_name}] Lettura PTB-XL"):
            row = ptbxl_meta.loc[ecg_id]
            record_key = f"ptbxl_{ecg_id}"
            try:
                signal = load_and_preprocess_ptbxl(ptbxl_root, row, notch_freq=50.0)
                if np.isnan(signal).any():
                    skipped.append((record_key, "segnale_contiene_NaN"))
                    continue
                result = extract_train_val_beats(signal, LEAD_NAMES_PTBXL)
            except Exception as e:
                skipped.append((record_key, f"errore: {e}"))
                continue
            if result is None:
                skipped.append((record_key, "picchi_insufficienti"))
                continue
            beats, used_peaks = result
            raw_beats[record_key] = (beats, used_peaks, "ptbxl", row["patient_id"])

        for name, record_path in tqdm(georgia_subset, desc=f"[{split_name}] Lettura Georgia"):
            record_key = f"georgia_{name}"
            try:
                signal = load_and_preprocess_georgia(record_path, notch_freq=60.0)
                if np.isnan(signal).any():
                    skipped.append((record_key, "segnale_contiene_NaN"))
                    continue
                result = extract_train_val_beats(signal, LEAD_NAMES_GEORGIA)
            except Exception as e:
                skipped.append((record_key, f"errore: {e}"))
                continue
            if result is None:
                skipped.append((record_key, "picchi_insufficienti"))
                continue
            beats, used_peaks = result
            raw_beats[record_key] = (beats, used_peaks, "georgia", name)

        assignment = assign_classes_per_recording(
            list(raw_beats.keys()), classes, class_weights, seed=seed
        )

        for record_key, (beats, used_peaks, source, patient_id) in tqdm(
            sorted(raw_beats.items()), desc=f"[{split_name}] Applicazione trasformazioni"
        ):
            labels = assignment[record_key]
            for local_idx in range(2):
                label = labels[local_idx]
                transformed = apply_transform(beats[local_idx], label)  # (T, 12)
                filename = f"seg{beat_counter:0{FILENAME_DIGITS}d}.npy"
                np.save(os.path.join(segments_dir, filename), transformed.T)  # -> (12, T)
                rows.append({
                    "filename": filename,
                    "ecg_id": record_key,
                    "patient_id": f"{source}_{patient_id}",
                    "source": source,
                    "beat_idx_in_record": local_idx,
                    "r_peak_sample": int(used_peaks[local_idx]),
                    "label": label,
                })
                beat_counter += 1

        index_df = pd.DataFrame(rows)
        index_csv_path = os.path.join(out_root, "final_dataset_index.csv")
        index_df.to_csv(index_csv_path, index=False)

    else:
        # ---- Test: tutti i battiti validi, etichetta "normal", nessuna
        # trasformazione (registrazioni reali, presumibilmente corrette).
        for ecg_id in tqdm(ptbxl_ecg_ids, desc=f"[{split_name}] Lettura PTB-XL"):
            row = ptbxl_meta.loc[ecg_id]
            record_key = f"ptbxl_{ecg_id}"
            try:
                signal = load_and_preprocess_ptbxl(ptbxl_root, row, notch_freq=50.0)
                if np.isnan(signal).any():
                    skipped.append((record_key, "segnale_contiene_NaN"))
                    continue
                result = extract_test_beats(signal, LEAD_NAMES_PTBXL)
            except Exception as e:
                skipped.append((record_key, f"errore: {e}"))
                continue
            if result is None:
                skipped.append((record_key, "nessun_battito_valido"))
                continue
            beats, used_peaks = result
            for local_idx, r_sample in enumerate(used_peaks):
                filename = f"seg{beat_counter:0{FILENAME_DIGITS}d}.npy"
                np.save(os.path.join(segments_dir, filename), beats[local_idx].T)
                rows.append({
                    "filename": filename,
                    "ecg_id": record_key,
                    "patient_id": f"ptbxl_{row['patient_id']}",
                    "source": "ptbxl",
                    "beat_idx_in_record": local_idx,
                    "r_peak_sample": int(r_sample),
                    "label": "normal",
                })
                beat_counter += 1

        for name, record_path in tqdm(georgia_subset, desc=f"[{split_name}] Lettura Georgia"):
            record_key = f"georgia_{name}"
            try:
                signal = load_and_preprocess_georgia(record_path, notch_freq=60.0)
                if np.isnan(signal).any():
                    skipped.append((record_key, "segnale_contiene_NaN"))
                    continue
                result = extract_test_beats(signal, LEAD_NAMES_GEORGIA)
            except Exception as e:
                skipped.append((record_key, f"errore: {e}"))
                continue
            if result is None:
                skipped.append((record_key, "nessun_battito_valido"))
                continue
            beats, used_peaks = result
            for local_idx, r_sample in enumerate(used_peaks):
                filename = f"seg{beat_counter:0{FILENAME_DIGITS}d}.npy"
                np.save(os.path.join(segments_dir, filename), beats[local_idx].T)
                rows.append({
                    "filename": filename,
                    "ecg_id": record_key,
                    "patient_id": f"georgia_{name}",
                    "source": "georgia",
                    "beat_idx_in_record": local_idx,
                    "r_peak_sample": int(r_sample),
                    "label": "normal",
                })
                beat_counter += 1

        index_df = pd.DataFrame(rows)
        index_csv_path = os.path.join(out_root, "beats_index.csv")
        index_df.to_csv(index_csv_path, index=False)

    skipped_csv_path = os.path.join(out_root, "skipped_records.csv")
    pd.DataFrame(skipped, columns=["record_key", "motivo"]).to_csv(skipped_csv_path, index=False)

    print(f"\n[{split_name}] Completato.")
    print(f"  Battiti salvati       : {beat_counter}")
    print(f"  Registrazioni saltate : {len(skipped)}")
    print(f"  Cartella segmenti     : {segments_dir}")
    print(f"  Indice (CSV)          : {index_csv_path}")
    if not is_test:
        print("  Distribuzione classi:")
        for cls, count in index_df["label"].value_counts().items():
            print(f"    {cls:10s}: {count}")
    else:
        print(f"  Registrazioni totali (per majority voting): {index_df['patient_id'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ptbxl_root", required=True)
    parser.add_argument("--georgia_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--classes", nargs="+",
                         default=["normal", "RA_LA", "LA_LL", "RA_LL", "V1_V2"])
    parser.add_argument("--class_weights", nargs="+", type=float, default=None)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights = args.class_weights if args.class_weights else [1.0] * len(args.classes)
    if len(weights) != len(args.classes):
        raise ValueError("--class_weights deve avere la stessa lunghezza di --classes")
    unknown = set(args.classes) - set(MISPLACEMENT_TRANSFORMS.keys())
    if unknown:
        raise ValueError(f"Classi sconosciute: {unknown}")

    print("--> Caricamento metadati PTB-XL...")
    ptbxl_meta = load_ptbxl_metadata(args.ptbxl_root)

    print("--> Scoperta registrazioni Georgia...")
    georgia_records = discover_records_georgia(args.georgia_root)  # [(name, path), ...]
    georgia_all_ids = [name for name, _ in georgia_records]

    print("--> Split train/val/test per PAZIENTE (PTB-XL) e per registrazione (Georgia)...")
    unique_patients = ptbxl_meta["patient_id"].unique().tolist()
    patients_train, patients_val, patients_test = split_ids(
        unique_patients, args.train_frac, args.val_frac, seed=args.seed
    )
    ptbxl_train = ptbxl_meta[ptbxl_meta["patient_id"].isin(patients_train)].index.tolist()
    ptbxl_val = ptbxl_meta[ptbxl_meta["patient_id"].isin(patients_val)].index.tolist()
    ptbxl_test = ptbxl_meta[ptbxl_meta["patient_id"].isin(patients_test)].index.tolist()

    georgia_train, georgia_val, georgia_test = split_ids(
        georgia_all_ids, args.train_frac, args.val_frac, seed=args.seed
    )

    print(f"  PTB-XL  : train={len(ptbxl_train)} val={len(ptbxl_val)} test={len(ptbxl_test)} "
          f"(pazienti: {len(patients_train)}/{len(patients_val)}/{len(patients_test)})")
    print(f"  Georgia : train={len(georgia_train)} val={len(georgia_val)} test={len(georgia_test)}")

    for split_name, ptbxl_ids, georgia_ids in [
        ("train", ptbxl_train, set(georgia_train)),
        ("val", ptbxl_val, set(georgia_val)),
        ("test", ptbxl_test, set(georgia_test)),
    ]:
        process_split(
            split_name=split_name,
            ptbxl_root=args.ptbxl_root,
            ptbxl_meta=ptbxl_meta,
            ptbxl_ecg_ids=ptbxl_ids,
            georgia_records=georgia_records,
            georgia_ids_set=georgia_ids,
            output_dir=args.output_dir,
            classes=args.classes,
            class_weights=weights,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()