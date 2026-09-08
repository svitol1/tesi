"""
FASE 1/2 - LENTA - eseguirla una sola volta (o solo quando cambiano i dati
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
classi/trasformazioni: cache TUTTE le registrazioni valide di entrambi i
dataset. Tutto il resto (split, esclusioni confirmed=True, class_weights,
trasformazioni di malposizionamento) e' responsabilita' di
dataset_builder.py, che legge da questa cache e puo' essere rieseguito
quante volte serve in pochi secondi/minuti invece di ore.

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
    print("--> Caricamento metadati PTB-XL...")
    ptbxl_meta = load_ptbxl_metadata(ptbxl_root)

    for ecg_id, row in tqdm(ptbxl_meta.iterrows(), total=len(ptbxl_meta), desc="PTB-XL"):
        record_key = f"ptbxl_{ecg_id}"
        # IMPORTANTE: patient_id e' noto SUBITO dai metadati grezzi, prima
        # di qualunque tentativo di lettura/preprocessing. Registriamo una
        # riga per QUESTA registrazione anche se il preprocessing fallisce,
        # cosi' dataset_builder.py puo' calcolare lo split sulla stessa
        # popolazione di pazienti completa usata dallo script originale
        # (split prima, scarti dopo) invece che su un sottoinsieme gia'
        # ridotto dalle registrazioni fallite qui in cache.
        try:
            signal = load_and_preprocess_ptbxl(ptbxl_root, row, notch_freq=50.0)
            if np.isnan(signal).any():
                raise ValueError("segnale_contiene_NaN")
            r_peaks = detect_r_peaks_v1_v5_average(signal, LEAD_NAMES_PTBXL)
        except Exception as e:
            rows.append({
                "record_key": record_key, "source": "ptbxl",
                "patient_id": row["patient_id"], "status": "failed",
                "motivo": str(e), "n_samples": None, "n_r_peaks": None,
            })
            continue

        np.save(os.path.join(signals_dir, f"{record_key}.npy"), signal)
        np.save(os.path.join(peaks_dir, f"{record_key}.npy"), r_peaks)
        rows.append({
            "record_key": record_key, "source": "ptbxl",
            "patient_id": row["patient_id"], "status": "ok",
            "motivo": "", "n_samples": signal.shape[0], "n_r_peaks": len(r_peaks),
        })

    print("--> Scoperta registrazioni Georgia...")
    georgia_records = discover_records_georgia(georgia_root)

    for name, record_path in tqdm(georgia_records, desc="Georgia"):
        record_key = f"georgia_{name}"
        try:
            signal = load_and_preprocess_georgia(record_path, notch_freq=60.0)
            if np.isnan(signal).any():
                raise ValueError("segnale_contiene_NaN")
            r_peaks = detect_r_peaks_v1_v5_average(signal, LEAD_NAMES_GEORGIA)
        except Exception as e:
            rows.append({
                "record_key": record_key, "source": "georgia",
                "patient_id": name, "status": "failed",  # per Georgia patient_id == ecg_id
                "motivo": str(e), "n_samples": None, "n_r_peaks": None,
            })
            continue

        np.save(os.path.join(signals_dir, f"{record_key}.npy"), signal)
        np.save(os.path.join(peaks_dir, f"{record_key}.npy"), r_peaks)
        rows.append({
            "record_key": record_key, "source": "georgia",
            "patient_id": name, "status": "ok",
            "motivo": "", "n_samples": signal.shape[0], "n_r_peaks": len(r_peaks),
        })

    metadata_df = pd.DataFrame(rows)
    metadata_df.to_csv(os.path.join(cache_dir, "metadata.csv"), index=False)

    n_ok = (metadata_df["status"] == "ok").sum()
    n_failed = (metadata_df["status"] == "failed").sum()
    print("\nCache completata.")
    print(f"  Registrazioni cachate con successo: {n_ok}")
    print(f"  Registrazioni fallite (in metadata ma senza signal/peaks): {n_failed}")
    print(f"  Cartella cache: {cache_dir}")
    print("  Nota: metadata.csv contiene una riga per OGNI registrazione scoperta, "
          "comprese quelle fallite (status='failed'), cosi' che dataset_builder.py "
          "possa calcolare lo split sulla popolazione completa. I file .npy in "
          "signals/ e peaks/ esistono solo per status='ok'.")


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
