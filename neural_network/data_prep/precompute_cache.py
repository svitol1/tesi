"""
FASE 1 - LENTA - eseguirla una sola volta (o solo quando cambiano i dati
grezzi, il preprocessing o l'algoritmo di rilevamento picchi).

Legge le registrazioni grezze PTB-XL + Georgia, applica il preprocessing
(filtri + notch) e rileva i picchi R (media V1/V5), poi salva su disco:

    <cache_dir>/signals/<record_key>.npy   segnale preprocessato (N, 12) float32
    <cache_dir>/peaks/<record_key>.npy     picchi R rilevati (array di int)
    <cache_dir>/metadata.csv               record_key, source, patient_id,
                                            n_samples, n_r_peaks
    <cache_dir>/skipped_records.csv        registrazioni scartate (NaN /
                                            errori di lettura) e motivo

Questa fase NON applica split train/val/test, esclusioni, ne' sa nulla di
classi/trasformazioni. Cache di TUTTE le registrazioni valide di entrambi i
dataset. Tutto il resto (split, esclusioni confirmed=True, class_weights,
trasformazioni di malposizionamento) e' responsabilita' di
dataset_builder.py, che legge da questa cache e puo' essere rieseguito
quante volte serve in poco tempo.

Uso
---
python precompute_cache.py \
    --ptbxl_root /path/to/ptbxl \
    --georgia_root /path/to/georgia \
    --cache_dir /path/to/cache
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .common import (
        LEAD_NAMES_PTBXL, LEAD_NAMES_GEORGIA,
        load_ptbxl_metadata, discover_records_georgia,
        load_and_preprocess_ptbxl, load_and_preprocess_georgia,
        detect_r_peaks_v1_v5_average,
    )
except ImportError:
    from common import (
        LEAD_NAMES_PTBXL, LEAD_NAMES_GEORGIA,
        load_ptbxl_metadata, discover_records_georgia,
        load_and_preprocess_ptbxl, load_and_preprocess_georgia,
        detect_r_peaks_v1_v5_average,
    )


def build_cache(ptbxl_root, georgia_root, cache_dir):
    signals_dir = os.path.join(cache_dir, "signals")
    peaks_dir = os.path.join(cache_dir, "peaks")
    os.makedirs(signals_dir, exist_ok=True)
    os.makedirs(peaks_dir, exist_ok=True)

    rows = []
    skipped = []

    print("--> Caricamento metadati PTB-XL...")
    ptbxl_meta = load_ptbxl_metadata(ptbxl_root)

    for ecg_id, row in tqdm(ptbxl_meta.iterrows(), total=len(ptbxl_meta), desc="PTB-XL"):
        record_key = f"ptbxl_{ecg_id}"
        try:
            signal = load_and_preprocess_ptbxl(ptbxl_root, row, notch_freq=50.0)
            if np.isnan(signal).any():
                skipped.append((record_key, "segnale_contiene_NaN"))
                continue
            r_peaks = detect_r_peaks_v1_v5_average(signal, LEAD_NAMES_PTBXL)
        except Exception as e:
            skipped.append((record_key, f"errore: {e}"))
            continue

        np.save(os.path.join(signals_dir, f"{record_key}.npy"), signal)
        np.save(os.path.join(peaks_dir, f"{record_key}.npy"), r_peaks)
        rows.append({
            "record_key": record_key,
            "source": "ptbxl",
            "patient_id": row["patient_id"],
            "n_samples": signal.shape[0],
            "n_r_peaks": len(r_peaks),
        })

    print("--> Scoperta registrazioni Georgia...")
    georgia_records = discover_records_georgia(georgia_root)

    for name, record_path in tqdm(georgia_records, desc="Georgia"):
        record_key = f"georgia_{name}"
        try:
            signal = load_and_preprocess_georgia(record_path, notch_freq=60.0)
            if np.isnan(signal).any():
                skipped.append((record_key, "segnale_contiene_NaN"))
                continue
            r_peaks = detect_r_peaks_v1_v5_average(signal, LEAD_NAMES_GEORGIA)
        except Exception as e:
            skipped.append((record_key, f"errore: {e}"))
            continue

        np.save(os.path.join(signals_dir, f"{record_key}.npy"), signal)
        np.save(os.path.join(peaks_dir, f"{record_key}.npy"), r_peaks)
        rows.append({
            "record_key": record_key,
            "source": "georgia",
            "patient_id": name,  # per Georgia patient_id == ecg_id
            "n_samples": signal.shape[0],
            "n_r_peaks": len(r_peaks),
        })

    metadata_df = pd.DataFrame(rows)
    metadata_df.to_csv(os.path.join(cache_dir, "metadata.csv"), index=False)
    pd.DataFrame(skipped, columns=["record_key", "motivo"]).to_csv(
        os.path.join(cache_dir, "skipped_records.csv"), index=False)

    print("\nCache completata.")
    print(f"  Registrazioni cachate : {len(rows)}")
    print(f"  Registrazioni scartate: {len(skipped)}")
    print(f"  Cartella cache        : {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ptbxl_root", required=True)
    parser.add_argument("--georgia_root", required=True)
    parser.add_argument("--cache_dir", required=True)
    args = parser.parse_args()
    build_cache(args.ptbxl_root, args.georgia_root, args.cache_dir)


if __name__ == "__main__":
    main()
