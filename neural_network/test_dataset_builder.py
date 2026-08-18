"""
Preprocessing del database Georgia (PhysioNet/CinC Challenge 2020) per creare
un dataset di singoli battiti segmentati, nello stesso formato usato per
PTB-XL, da usare come TEST SET per un modello CNN/GCN che rileva il
posizionamento errato degli elettrodi nell'ECG a 12 derivazioni.

Differenze principali rispetto a PTB-XL
----------------------------------------
Il database Georgia (cartella .../training/georgia/g1 ... g11) NON contiene
un file ptbxl_database.csv con i metadati. Ogni registrazione è invece una
coppia di file WFDB:
    E00001.hea   -> header di testo con: sampling rate, numero di campioni,
                    nomi delle derivazioni, e metadati extra in righe che
                    iniziano con '#' (Age, Sex, Dx, ecc.)
    E00001.mat   -> segnale grezzo (MATLAB array)

Non esiste inoltre un campo 'patient_id' ufficiale: qui uso l'ecg_id stesso
come proxy del patient_id.

Questo intero dataset Georgia viene usato SOLO come test set (nessuno split
train/val/test, nessun mescolamento dei battiti): le registrazioni vengono
lette ed elaborate nell'ordine in cui vengono trovate sul disco (ordine
alfabetico per cartella g1..g11 e nome file), e ogni battito segmentato
viene salvato con un contatore progressivo che rispecchia quell'ordine.

Uso
---
Prima (opzionale) costruisco solo l'indice dei metadati leggendo tutti gli
.hea, per ispezionarlo prima di lanciare tutta la segmentazione:
    python georgia_preprocessing.py --georgia_root .../training/georgia \
        --output_dir .../out_georgia --build_index_only

Oppure faccio tutto in un colpo solo (indice + segmentazione battiti):
    python georgia_preprocessing.py --georgia_root .../training/georgia \
        --output_dir .../out_georgia
"""

import argparse
import glob
import os

from data_prep.preprocessing import preprocess_ecg
from data_prep.pan_tompkins_algo import detect_r_peaks_pan_tompkins

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

FILENAME_DIGITS = 6


# --------------------------------------------------------------------------
# 1. Scansione della cartella Georgia e parsing degli header .hea
# --------------------------------------------------------------------------


def find_hea_files(georgia_root: str):
    """Trova ricorsivamente tutti i file .hea sotto georgia_root (g1..g11)."""
    pattern = os.path.join(georgia_root, "**", "*.hea")
    return sorted(glob.glob(pattern, recursive=True))


def parse_hea_header(hea_path: str) -> dict:
    """
    Legge un file .hea in formato WFDB (Challenge 2020) e ne estrae i
    metadati utili. Ritorna un dizionario con: ecg_id, record_path (senza
    estensione, per wfdb.rdsamp), fs, n_samples, n_leads, lead_names, age,
    sex, dx (codici SNOMED, come stringa così come compaiono nell'header).
    """
    with open(hea_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    # prima riga: <nome_record> <n_derivazioni> <fs> <n_campioni> ...
    header_parts = lines[0].split()
    record_name = header_parts[0]
    n_leads = int(header_parts[1])
    fs = int(float(header_parts[2]))
    n_samples = int(header_parts[3])

    # righe 1..n_leads: una per ogni derivazione, l'ultimo campo è il nome
    lead_names = []
    for i in range(1, n_leads + 1):
        parts = lines[i].split()
        lead_names.append(parts[-1])

    # righe di metadati, iniziano con '#'
    age, sex, dx = None, None, None
    for line in lines[n_leads + 1:]:
        if line.startswith("#Age"):
            val = line.split(":", 1)[1].strip()
            age = None if val in ("NaN", "Unknown", "") else val
        elif line.startswith("#Sex"):
            sex = line.split(":", 1)[1].strip()
        elif line.startswith("#Dx"):
            dx = line.split(":", 1)[1].strip()

    record_dir = os.path.dirname(hea_path)
    record_path = os.path.join(record_dir, record_name)

    return {
        "ecg_id": record_name,
        "record_path": record_path,
        "source_dir": os.path.basename(record_dir),  # es. 'g1'
        "fs": fs,
        "n_samples": n_samples,
        "n_leads": n_leads,
        "lead_names": ",".join(lead_names),
        "age": age,
        "sex": sex,
        "dx": dx,
    }


def build_georgia_index(georgia_root: str) -> pd.DataFrame:
    """Costruisce il DataFrame di metadati equivalente a ptbxl_database.csv."""
    hea_files = find_hea_files(georgia_root)
    if len(hea_files) == 0:
        raise FileNotFoundError(
            f"Nessun file .hea trovato sotto {georgia_root}. "
            "Controlla il percorso (dovrebbe contenere le cartelle g1..g11)."
        )

    # find_hea_files ritorna già i path in ordine alfabetico,
    # quindi l'ordine delle righe qui rispecchia l'ordine su disco: nessun
    # mescolamento, nessuno split, dato che l'intero dataset è solo test.
    rows = [parse_hea_header(p) for p in tqdm(hea_files, desc="Lettura header .hea")]
    df = pd.DataFrame(rows).set_index("ecg_id")

    # uso l'ecg_id stesso come proxy del patient_id (nessun paziente
    # ripetuto noto nel sottoinsieme Georgia)
    df["patient_id"] = df.index

    return df


# --------------------------------------------------------------------------
# 2. Caricamento del segnale grezzo
# --------------------------------------------------------------------------


def load_raw_signal(row: pd.Series) -> np.ndarray:
    """
    Carica il segnale grezzo per una registrazione Georgia.
    Ritorna un array shape (n_samples, 12), riordinando le derivazioni
    secondo LEAD_NAMES se necessario.
    """
    signal, meta = wfdb.rdsamp(row["record_path"])
    sig_names = [s.upper() for s in meta["sig_name"]]

    if sig_names == LEAD_NAMES:
        ordered = signal
    else:
        # riordina le colonne secondo LEAD_NAMES (case-insensitive)
        try:
            idx_map = [sig_names.index(name) for name in LEAD_NAMES]
        except ValueError:
            raise ValueError(
                f"Derivazioni inattese in {row['record_path']}: {meta['sig_name']}"
            )
        ordered = signal[:, idx_map]

    return ordered.astype(np.float32)


# --------------------------------------------------------------------------
# 3. Rilevamento picchi R (identico a PTB-XL)
# --------------------------------------------------------------------------

def filter_r_peaks_by_v1_v5_window(r_peaks_v1: np.ndarray,
                                  r_peaks_v5: np.ndarray,
                                  fs: int,
                                  tolerance_ms: float = 50.0) -> np.ndarray:
    """
    Accetta un picco R solo se è stato rilevato anche nell'altra derivazione
    all'interno di una finestra temporale di +-50 ms. In questo modo si
    scartano picchi isolati in V1 o V5 che non hanno un corrispondente
    nella derivazione opposta.
    """
    if len(r_peaks_v1) == 0 or len(r_peaks_v5) == 0:
        return np.array([], dtype=int)

    tolerance_samples = max(1, int(round(tolerance_ms / 1000.0 * fs)))
    r_peaks_v1 = np.asarray(r_peaks_v1, dtype=int)
    r_peaks_v5 = np.asarray(r_peaks_v5, dtype=int)

    paired_v1 = []
    used_v5 = set()

    for peak_v1 in r_peaks_v1:
        candidates = np.where(np.abs(r_peaks_v5 - peak_v1) <= tolerance_samples)[0]
        if len(candidates) == 0:
            continue

        nearest_idx = int(candidates[np.argmin(np.abs(r_peaks_v5[candidates] - peak_v1))])
        if nearest_idx in used_v5:
            continue

        used_v5.add(nearest_idx)
        peak_v5 = r_peaks_v5[nearest_idx]
        paired_v1.append(int(round((peak_v1 + peak_v5) / 2.0)))

    return np.unique(np.asarray(paired_v1, dtype=int))


def detect_r_peaks_v1_v5_average(signal_12lead: np.ndarray, fs: int) -> np.ndarray:
    try:
        v1_idx = LEAD_NAMES.index("V1")
        v5_idx = LEAD_NAMES.index("V5")

        v1_signal = preprocess_ecg(signal_12lead[:, v1_idx], fs, notch_freq=60.0)
        v5_signal = preprocess_ecg(signal_12lead[:, v5_idx], fs, notch_freq=60.0)

        r_peaks_v1 = detect_r_peaks_pan_tompkins(v1_signal, fs)
        r_peaks_v5 = detect_r_peaks_pan_tompkins(v5_signal, fs)

        if len(r_peaks_v1) == 0 or len(r_peaks_v5) == 0:
            return np.array([], dtype=int)

        return filter_r_peaks_by_v1_v5_window(r_peaks_v1, r_peaks_v5, fs, tolerance_ms=50.0)
    except Exception:
        return np.array([], dtype=int)


# --------------------------------------------------------------------------
# 4. Segmentazione dei battiti (identico a PTB-XL)
# --------------------------------------------------------------------------


def segment_beats(signal_12lead: np.ndarray, r_peaks: np.ndarray, fs: int,
                   pre_ms: float = 400.0, post_ms: float = 600.0):
    n_samples = signal_12lead.shape[0]
    pre_samples = int(round(pre_ms / 1000.0 * fs))
    post_samples = int(round(post_ms / 1000.0 * fs))
    window_len = pre_samples + post_samples

    beats = []
    used_peaks = []
    for r in r_peaks:
        start = r - pre_samples
        end = r + post_samples
        if start < 0 or end > n_samples:
            continue
        beats.append(signal_12lead[start:end, :])
        used_peaks.append(r)

    if len(beats) == 0:
        return (np.empty((0, window_len, 12), dtype=np.float32),
                np.array([], dtype=int))

    return np.stack(beats).astype(np.float32), np.array(used_peaks, dtype=int)


# --------------------------------------------------------------------------
# 5. Pipeline completa
# --------------------------------------------------------------------------


def build_dataset(georgia_root: str, output_dir: str,
                   pre_ms: float = 400.0, post_ms: float = 600.0,
                   limit: int = None, index_csv: str = None):
    os.makedirs(output_dir, exist_ok=True)
    segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    # se ho già un indice salvato da una run precedente lo riuso, altrimenti
    # lo costruisco leggendo tutti i .hea
    if index_csv is not None and os.path.exists(index_csv):
        meta_df = pd.read_csv(index_csv, index_col="ecg_id")
    else:
        meta_df = build_georgia_index(georgia_root)
        meta_df.to_csv(os.path.join(output_dir, "georgia_metadata.csv"))

    if limit is not None:
        meta_df = meta_df.iloc[:limit]

    index_rows = []
    skipped_records = []
    global_beat_counter = 0

    for ecg_id, row in tqdm(meta_df.iterrows(), total=len(meta_df),
                             desc="Processing registrazioni"):
        fs = int(row["fs"])
        try:
            signal = load_raw_signal(row)
        except Exception as e:
            skipped_records.append((ecg_id, f"errore_lettura: {e}"))
            continue

        r_peaks = detect_r_peaks_v1_v5_average(signal, fs)
        if len(r_peaks) < 5:
            skipped_records.append((ecg_id, "picchi_R_insufficienti_per_5o"))
            continue

        beats, used_peaks = segment_beats(signal, r_peaks, fs, pre_ms, post_ms)

        for local_idx, r_sample in enumerate(used_peaks):
            filename = f"seg{global_beat_counter:0{FILENAME_DIGITS}d}.npy"
            filepath = os.path.join(segments_dir, filename)
            np.save(filepath, beats[local_idx].T)

            index_rows.append({
                "filename": filename,
                "ecg_id": ecg_id,
                "patient_id": row["patient_id"],
                "beat_idx_in_record": local_idx,
                "r_peak_sample": int(r_sample),
                "split": "test",
                "label": "normal",
                "sampling_rate": fs,
            })
            global_beat_counter += 1

    index_df = pd.DataFrame(index_rows)
    index_csv_path = os.path.join(output_dir, "beats_index.csv")
    index_df.to_csv(index_csv_path, index=False)

    skipped_df = pd.DataFrame(skipped_records, columns=["ecg_id", "motivo"])
    skipped_csv_path = os.path.join(output_dir, "skipped_records.csv")
    skipped_df.to_csv(skipped_csv_path, index=False)

    print(f"\nCompletato.")
    print(f"  Battiti totali salvati : {global_beat_counter}")
    print(f"  Registrazioni saltate  : {len(skipped_records)}")
    print(f"  Cartella segmenti      : {segments_dir}")
    print(f"  Indice battiti (CSV)   : {index_csv_path}")
    print(f"  Registrazioni saltate  : {skipped_csv_path}")

    return segments_dir, index_csv_path


# --------------------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--georgia_root", required=True,
                         help="Cartella radice del database Georgia "
                              "(quella che contiene g1, g2, ..., g11)")
    parser.add_argument("--output_dir", required=True,
                         help="Cartella dove salvare segments/, beats_index.csv "
                              "e georgia_metadata.csv")
    parser.add_argument("--pre_ms", type=float, default=400.0)
    parser.add_argument("--post_ms", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=None,
                         help="Solo per debug: processa solo le prime N registrazioni")
    parser.add_argument("--build_index_only", action="store_true",
                         help="Costruisce solo georgia_metadata.csv e si ferma "
                              "(utile per ispezionare i metadati prima di lanciare "
                              "tutta la segmentazione)")
    args = parser.parse_args()

    if args.build_index_only:
        os.makedirs(args.output_dir, exist_ok=True)
        meta_df = build_georgia_index(args.georgia_root)
        out_path = os.path.join(args.output_dir, "georgia_metadata.csv")
        meta_df.to_csv(out_path)
        print(f"Indice salvato in: {out_path}  ({len(meta_df)} registrazioni)")
        return

    index_csv = os.path.join(args.output_dir, "georgia_metadata.csv")
    build_dataset(
        georgia_root=args.georgia_root,
        output_dir=args.output_dir,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        limit=args.limit,
        index_csv=index_csv,
    )


if __name__ == "__main__":
    main()