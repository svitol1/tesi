"""
================================================================================
Visualizzazione con ecg_plot di una registrazione (PTB-XL o Georgia),
a confronto tra il tracciato originale (cosi' come nel database) e il
tracciato dopo aver applicato una trasformazione di malposizionamento
(tipicamente quella predetta dal modello per una registrazione sospetta,
per un controllo visivo del test di conferma).

Richiede: pip install ecg_plot

Uso
---
    python visualize_recording.py \
        --ecg_id ptbxl_12001 \
        --ptbxl_root /path/to/ptbxl \
        --georgia_root /path/to/georgia \
        --transform V1_V2
================================================================================
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import wfdb
import ecg_plot

# This file may be executed directly from cnn_test/find_mislabel/ with a plain
# "python check_recording.py" command. In that case, Python does not have the
# repo root on sys.path and relative imports like "...data_prep" are invalid.
# Add the project root explicitly so we can import the sibling package modules.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_prep.lead_misplacement import apply_transform
from data_prep.dataset_builder import (
    load_ptbxl_metadata,
    discover_records_georgia,
    load_and_preprocess_ptbxl,
    load_and_preprocess_georgia,
)

SAMPLING_RATE = 500


def load_recording(ecg_id: str, ptbxl_root: str, georgia_root: str, preprocess: bool = True) -> np.ndarray:
    """
    Carica il segnale (n_samples, 12) di una registrazione a partire dal suo
    ecg_id (es. 'ptbxl_12001' o 'georgia_JS00001'), nell'ordine derivazioni
    standard I,II,III,aVR,aVL,aVF,V1..V6 (coerente con lead_misplacement.py
    e con dataset_builder.py).

    preprocess=True  -> stesso preprocessing (filtri + notch) visto dal modello
    preprocess=False -> segnale grezzo WFDB, per una lettura "clinica" più pulita
    """
    if ecg_id.startswith("ptbxl_"):
        raw_id = int(ecg_id[len("ptbxl_"):])
        ptbxl_meta = load_ptbxl_metadata(ptbxl_root)
        if raw_id not in ptbxl_meta.index:
            raise ValueError(f"ecg_id PTB-XL {raw_id} non trovato in ptbxl_database.csv")
        row = ptbxl_meta.loc[raw_id]

        if preprocess:
            signal = load_and_preprocess_ptbxl(ptbxl_root, row, notch_freq=50.0)
        else:
            record_path = str(Path(ptbxl_root) / row["filename_hr"])
            signal, _ = wfdb.rdsamp(record_path)

    elif ecg_id.startswith("georgia_"):
        name = ecg_id[len("georgia_"):]
        georgia_records = dict(discover_records_georgia(georgia_root))
        if name not in georgia_records:
            raise ValueError(f"Registrazione Georgia '{name}' non trovata sotto {georgia_root}")
        record_path = georgia_records[name]

        if preprocess:
            signal = load_and_preprocess_georgia(record_path, notch_freq=60.0)
        else:
            signal, _ = wfdb.rdsamp(record_path)

    else:
        raise ValueError(f"ecg_id non riconosciuto (prefisso atteso 'ptbxl_' o 'georgia_'): {ecg_id}")

    return signal.astype(np.float32)


def plot_recording(signal: np.ndarray, sample_rate: int, title: str):
    """signal: (n_samples, 12) -> ecg_plot vuole (12, n_samples)."""
    ecg_plot.plot(signal.T, sample_rate=sample_rate, title=title, columns=2)
    ecg_plot.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ecg_id", required=True, help="es. ptbxl_12001, georgia_JS00001")
    parser.add_argument("--ptbxl_root", required=True)
    parser.add_argument("--georgia_root", required=True)
    parser.add_argument("--transform", default=None,
                        help="Se specificato (es. V1_V2, RA_LA...), mostra anche il tracciato "
                             "dopo aver applicato questa trasformazione, per confronto")
    parser.add_argument("--raw", action="store_true",
                        help="Mostra il segnale grezzo WFDB invece di quello preprocessato dal modello")
    args = parser.parse_args()

    signal = load_recording(args.ecg_id, args.ptbxl_root, args.georgia_root, preprocess=not args.raw)

    plot_recording(signal, SAMPLING_RATE, title=f"{args.ecg_id} - originale (database)")

    if args.transform:
        transformed = apply_transform(signal, args.transform)
        plot_recording(transformed, SAMPLING_RATE, title=f"{args.ecg_id} - dopo {args.transform}")


if __name__ == "__main__":
    main()