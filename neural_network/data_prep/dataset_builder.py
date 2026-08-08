"""
build_ptbxl_beat_dataset.py

Preprocessing del database PTB-XL (PhysioNet) per creare un dataset di
"battiti" segmentati, da usare per un modello CNN/GCN che rileva
il posizionamento errato degli elettrodi nell'ECG a 12 derivazioni.

COSA FA
-------
Per ogni registrazione ECG a 12 derivazioni (10 secondi):
  1. Carica il segnale grezzo (wfdb).
  2. Rileva i picchi R su una derivazione "guida" (default: lead II).
  3. Per ogni picco R trovato, estrae una finestra sincrona su TUTTE le
     12 derivazioni: [R - 300 ms, R + 600 ms].
  4. Scarta i battiti troppo vicini all'inizio/fine della registrazione
     (finestra che uscirebbe dai 10s).
  5. Salva OGNI battito come file .npy separato (seg000000.npy,
     seg000001.npy, ...) in <output_dir>/segments/, shape (window_len, 12).
     Un CSV di indice collega ogni file ai metadati (ecg_id, patient_id,
     strat_fold, posizione del picco R, ecc.).

NOTE IMPORTANTI PER IL TUO CASO D'USO (misplacement detection)
----------------------------------------------------------------
- Il rilevamento dei picchi R viene fatto su UNA sola derivazione guida
  e la stessa posizione temporale viene poi applicata a TUTTE le 12
  derivazioni. Questo è intenzionale: se rilevassi i picchi
  indipendentemente su ogni derivazione, uno scambio di elettrodi
  potrebbe disallineare le finestre tra derivazioni, "nascondendo" col
  preprocessing proprio l'anomalia che vuoi far riconoscere al modello.
- Lo split train/val/test va fatto per PAZIENTE (patient_id), non per
  battito né per registrazione, altrimenti hai data leakage. PTB-XL
  fornisce già 'strat_fold' pensato per questo: fold 1-8 = train,
  9 = val, 10 = test (convenzione standard del paper originale di PTB-XL).

REQUISITI
---------
pip install wfdb neurokit2 tqdm pandas numpy --break-system-packages

USO
---
python build_ptbxl_beat_dataset.py \
    --ptbxl_root /path/to/ptb-xl \
    --output_dir /path/to/output \
    --sampling_rate 500 \
    --guide_lead II
"""

import argparse
import os

import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

# Le 12 derivazioni standard, nell'ordine in cui compaiono nei file PTB-XL
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Numero di cifre usate nel nome dei file seg000000.npy -> fino a 10^7 - 1 battiti
FILENAME_DIGITS = 7


# --------------------------------------------------------------------------
# 1. Caricamento metadati e segnali PTB-XL
# --------------------------------------------------------------------------

def load_ptbxl_metadata(ptbxl_root: str) -> pd.DataFrame:
    """
    Carica ptbxl_database.csv.

    NOTA: la colonna 'scp_codes' contiene un dizionario scritto come
    stringa (es. "{'NORM': 100.0}"), utile solo se serve la diagnosi
    clinica. Per la segmentazione dei battiti non serve, quindi qui non
    viene parsata (si evita così la dipendenza da 'ast'/'eval' su dati
    esterni). Se in futuro ti servisse, puoi parsarla con json dopo aver
    sostituito gli apici singoli con doppi, oppure con ast.literal_eval.
    """
    path = os.path.join(ptbxl_root, "ptbxl_database.csv")
    df = pd.read_csv(path, index_col="ecg_id")
    return df


def load_raw_signal(ptbxl_root: str, row: pd.Series, sampling_rate: int) -> np.ndarray:
    """
    Carica il segnale grezzo per una singola registrazione.
    Ritorna un array shape (n_samples, 12), unità mV.
    """
    if sampling_rate == 100:
        rel_path = row["filename_lr"]
    elif sampling_rate == 500:
        rel_path = row["filename_hr"]
    else:
        raise ValueError("sampling_rate deve essere 100 o 500 (come da file PTB-XL)")

    record_path = os.path.join(ptbxl_root, rel_path)
    signal, meta = wfdb.rdsamp(record_path)
    assert meta["sig_name"] == LEAD_NAMES, (
        f"Ordine derivazioni inatteso: {meta['sig_name']}"
    )
    return signal.astype(np.float32)


# --------------------------------------------------------------------------
# 2. Rilevamento picchi R
# --------------------------------------------------------------------------

def detect_r_peaks(signal_12lead: np.ndarray, fs: int, guide_lead: str = "II"):
    """
    Rileva i picchi R sulla derivazione guida.
    Ritorna un array di indici campione (int). Vuoto se il rilevamento fallisce.
    """
    lead_idx = LEAD_NAMES.index(guide_lead)
    lead_signal = signal_12lead[:, lead_idx]

    try:
        _, info = nk.ecg_peaks(lead_signal, sampling_rate=fs, method="neurokit")
        r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        r_peaks = np.array([], dtype=int)

    return r_peaks


# --------------------------------------------------------------------------
# 3. Segmentazione dei battiti
# --------------------------------------------------------------------------

def segment_beats(signal_12lead: np.ndarray, r_peaks: np.ndarray, fs: int,
                   pre_ms: float = 300.0, post_ms: float = 600.0):
    """
    Estrae, per ogni picco R, una finestra [R-pre_ms, R+post_ms] su tutte
    le 12 derivazioni.

    Ritorna:
        beats: array shape (n_beats_validi, window_len, 12)
        used_peaks: indici campione dei picchi R effettivamente usati
    """
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
# 4. Pipeline completa: costruzione del dataset
# --------------------------------------------------------------------------

def build_dataset(ptbxl_root: str, output_dir: str, sampling_rate: int = 500,
                   guide_lead: str = "II", pre_ms: float = 300.0,
                   post_ms: float = 600.0, min_beats_per_record: int = 1,
                   limit: int = None):
    """
    Processa l'intero database PTB-XL e salva:
      - <output_dir>/segments/seg0000000.npy, seg0000001.npy, ...
      - <output_dir>/beats_index.csv    : indice con metadati per ogni battito
      - <output_dir>/skipped_records.csv: registrazioni escluse e motivo
    """
    segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    meta_df = load_ptbxl_metadata(ptbxl_root)
    if limit is not None:
        meta_df = meta_df.iloc[:limit]

    fs = sampling_rate
    index_rows = []
    skipped_records = []
    global_beat_counter = 0

    for ecg_id, row in tqdm(meta_df.iterrows(), total=len(meta_df),
                             desc="Processamento registrazioni"):
        try:
            signal = load_raw_signal(ptbxl_root, row, sampling_rate)
        except Exception as e:
            skipped_records.append((ecg_id, f"errore_lettura: {e}"))
            continue

        r_peaks = detect_r_peaks(signal, fs, guide_lead=guide_lead)
        if len(r_peaks) < min_beats_per_record:
            skipped_records.append((ecg_id, "picchi_R_insufficienti"))
            continue

        beats, used_peaks = segment_beats(signal, r_peaks, fs, pre_ms, post_ms)
        if beats.shape[0] == 0:
            skipped_records.append((ecg_id, "nessun_battito_valido_dopo_bordo"))
            continue

        for local_idx, r_sample in enumerate(used_peaks):
            filename = f"seg{global_beat_counter:0{FILENAME_DIGITS}d}.npy"
            filepath = os.path.join(segments_dir, filename)
            np.save(filepath, beats[local_idx])

            index_rows.append({
                "filename": filename,
                "ecg_id": ecg_id,
                "patient_id": row["patient_id"],
                "beat_idx_in_record": local_idx,
                "r_peak_sample": int(r_sample),
                "strat_fold": row["strat_fold"],
                "guide_lead": guide_lead,
                "sampling_rate": sampling_rate,
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
# 5. CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptbxl_root", required=True,
                         help="Cartella radice del database PTB-XL locale "
                              "(quella che contiene ptbxl_database.csv)")
    parser.add_argument("--output_dir", required=True,
                         help="Cartella dove salvare segments/ e beats_index.csv")
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500],
                         help="100 Hz o 500 Hz (come da file PTB-XL originali)")
    parser.add_argument("--guide_lead", type=str, default="II",
                         choices=LEAD_NAMES,
                         help="Derivazione usata per il rilevamento dei picchi R")
    parser.add_argument("--pre_ms", type=float, default=300.0)
    parser.add_argument("--post_ms", type=float, default=600.0)
    parser.add_argument("--min_beats_per_record", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                         help="Solo per debug: processa solo le prime N registrazioni")
    args = parser.parse_args()

    build_dataset(
        ptbxl_root=args.ptbxl_root,
        output_dir=args.output_dir,
        sampling_rate=args.sampling_rate,
        guide_lead=args.guide_lead,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        min_beats_per_record=args.min_beats_per_record,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
