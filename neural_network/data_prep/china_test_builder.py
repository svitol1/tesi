"""
Script per trasformare l'INTERO database https://physionet.org/content/ecg-arrhythmia/1.0.0/
in un unico split di TEST per la valutazione dei modelli per la rilevazione del
malposizionamento degli elettrodi.

Gestisce la struttura ricorsiva dei file RECORDS (es. WFDBRecords/44/440/JS43498).
I segnali vengono preprocessati (filtri + notch) riutilizzando i moduli condivisi,
ed estrae TUTTI i battiti validi per ogni registrazione.

Output generati nella cartella di destinazione:
    <output_dir>/test/segments/*.npy    Segmenti di segnale (12, window_len)
    <output_dir>/test/beats_index.csv   Indice dei battiti estratti (con label 'normal')
    <output_dir>/test/skipped_records.csv Record scartati e relativo motivo

Uso:
----
python build_physionet_test_dataset.py \
    --physionet_root /path/to/ecg-arrhythmia/1.0.0 \
    --output_dir /path/to/output_dataset \
    --notch_freq 50.0
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

try:
    from .common import (
        SAMPLING_RATE,
        detect_r_peaks_v1_v5_average,
        segment_beats,
    )
    from .preprocessing import preprocess_ecg
except ImportError:
    from common import (
        SAMPLING_RATE,
        detect_r_peaks_v1_v5_average,
        segment_beats,
    )
    from preprocessing import preprocess_ecg

LEAD_NAMES_PHYSIONET = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

FILENAME_DIGITS = 7


def discover_physionet_records(physionet_root: str):
    """
    Risolve ricorsivamente tutti i file RECORDS presenti nelle sottocartelle
    del database PhysioNet per trovare tutte le registrazioni effettive (.mat/.hea).

    Restituisce una lista di tuple: (record_name, full_path_without_extension).
    """
    root_path = Path(physionet_root)
    records = []
    seen = set()

    # Cerca tutti i file denominati "RECORDS" in qualsiasi punto del dataset
    records_files = sorted(root_path.rglob("RECORDS"))

    if records_files:
        for rec_file in records_files:
            parent_dir = rec_file.parent
            with rec_file.open(encoding="utf-8") as f:
                for line in f:
                    entry = line.strip()
                    if not entry or entry.startswith("#"):
                        continue

                    # Costruisce il percorso completo a partire dalla cartella del file RECORDS corrente
                    full_path = parent_dir / entry

                    # Se punta a un file .hea reale (o alla base senza estensione del file .mat/.hea)
                    hea_candidate = full_path.with_suffix(".hea")
                    if hea_candidate.exists() or full_path.exists():
                        # Elimina l'estensione se presente per passare la base a wfdb.rdsamp
                        stem_path = full_path.with_suffix("") if full_path.suffix in [".hea", ".mat"] else full_path
                        str_path = str(stem_path)
                        if str_path not in seen and stem_path.with_suffix(".hea").exists():
                            seen.add(str_path)
                            records.append((stem_path.name, str_path))

    # Fallback: Se la ricerca tramite RECORDS fallisce, cerca direttamente i file .hea
    if not records:
        print("--> Avviso: Nessun puntamento valido nei file RECORDS. Cerca direttamente i file .hea...")
        hea_files = sorted(root_path.rglob("*.hea"))
        for hea_path in hea_files:
            stem_path = hea_path.with_suffix("")
            str_path = str(stem_path)
            if str_path not in seen:
                seen.add(str_path)
                records.append((stem_path.name, str_path))

    if not records:
        raise FileNotFoundError(f"Nessuna registrazione trovata in {physionet_root}")

    return records


def load_and_preprocess_physionet(record_path: str, notch_freq: float = 50.0):
    """
    Carica una registrazione WFDB, verifica la frequenza di campionamento e
    applica il preprocessing (filtri + notch) su ciascun canale.
    """
    signal, meta = wfdb.rdsamp(record_path)

    # Verifica e eventuale resample della frequenza di campionamento
    fs_in = int(meta["fs"])
    if fs_in != SAMPLING_RATE:
        signal, _ = wfdb.processing.resample_sig(signal, fs_in, SAMPLING_RATE)

    # Normalizzazione dei nomi delle derivazioni per il confronto
    sig_names = [s.upper() for s in meta["sig_name"]]
    expected_names = [s.upper() for s in LEAD_NAMES_PHYSIONET]

    if sig_names != expected_names:
        if sorted(sig_names) == sorted(expected_names):
            order = [sig_names.index(name) for name in expected_names]
            signal = signal[:, order]
        else:
            raise ValueError(f"Derivazioni inattese: trovate {meta['sig_name']}")

    # Filtraggio canale per canale
    signal = signal.astype(np.float32)
    for lead_idx in range(signal.shape[1]):
        signal[:, lead_idx] = preprocess_ecg(
            signal[:, lead_idx],
            fs=SAMPLING_RATE,
            notch_freq=notch_freq
        )

    return signal


def build_physionet_test_split(physionet_root: str, output_dir: str, notch_freq: float = 50.0):
    """
    Costruisce lo split di TEST convertendo l'intero dataset PhysioNet.
    """
    out_root = os.path.join(output_dir, "test")
    segments_dir = os.path.join(out_root, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    print(f"--> Ricerca registrazioni nella struttura PhysioNet: {physionet_root}")
    records = discover_physionet_records(physionet_root)
    print(f"--> Trovate {len(records)} registrazioni valide da elaborare.")

    rows = []
    skipped = []
    beat_counter = 0

    for rec_name, record_path in tqdm(records, desc="[TEST PhysioNet] Elaborazione"):
        record_key = f"physionet_{rec_name}"

        # 1. Caricamento e Preprocessing del segnale completo
        try:
            signal = load_and_preprocess_physionet(record_path, notch_freq=notch_freq)
            if np.isnan(signal).any():
                raise ValueError("segnale_contiene_NaN")
        except Exception as e:
            skipped.append((record_key, f"errore_caricamento: {str(e)}"))
            continue

        # 2. Rilevamento Picchi R (tramite media V1 e V5)
        r_peaks = detect_r_peaks_v1_v5_average(signal, LEAD_NAMES_PHYSIONET, fs=SAMPLING_RATE)
        if len(r_peaks) == 0:
            skipped.append((record_key, "nessun_picco_r_rilevato"))
            continue

        # 3. Estrazione di TUTTI i battiti validi
        beats, used_peaks = segment_beats(signal, r_peaks, fs=SAMPLING_RATE)
        if beats.shape[0] == 0:
            skipped.append((record_key, "nessun_battito_segmentabile_ai_bordi"))
            continue

        # 4. Salvataggio battiti .npy (leads-first: 12, window_len) e tracciamento
        for local_idx, r_sample in enumerate(used_peaks):
            filename = f"seg{beat_counter:0{FILENAME_DIGITS}d}.npy"

            # Formato salvato (12, window_len)
            np.save(os.path.join(segments_dir, filename), beats[local_idx].T)

            rows.append({
                "filename": filename,
                "ecg_id": record_key,
                "patient_id": f"physionet_{rec_name}",
                "source": "physionet",
                "beat_idx_in_record": local_idx,
                "r_peak_sample": int(r_sample),
                "label": "normal",
            })
            beat_counter += 1

    # Salvataggio dell'indice CSV dei battiti e dei file scartati
    index_df = pd.DataFrame(rows)
    index_csv_path = os.path.join(out_root, "beats_index.csv")
    index_df.to_csv(index_csv_path, index=False)

    skipped_csv_path = os.path.join(out_root, "skipped_records.csv")
    pd.DataFrame(skipped, columns=["record_key", "motivo"]).to_csv(skipped_csv_path, index=False)

    print("\n--> Completato!")
    print(f"  Battiti salvati        : {beat_counter}")
    print(f"  Registrazioni scartate : {len(skipped)}")
    print(f"  Registrazioni estratte : {index_df['patient_id'].nunique() if not index_df.empty else 0}")
    print(f"  Directory dei segmenti : {segments_dir}")
    print(f"  File d'indice          : {index_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--physionet_root",
        required=True,
        help="Percorso radice locale del dataset PhysioNet (cartella contenente WFDBRecords/ o il file RECORDS principale)."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory dove verrà creato lo split di test (es. /path/to/dataset_output)."
    )
    parser.add_argument(
        "--notch_freq",
        type=float,
        default=50.0,
        help="Frequenza del filtro notch in Hz (default: 50.0 Hz)."
    )

    args = parser.parse_args()

    build_physionet_test_split(
        physionet_root=args.physionet_root,
        output_dir=args.output_dir,
        notch_freq=args.notch_freq
    )


if __name__ == "__main__":
    main()