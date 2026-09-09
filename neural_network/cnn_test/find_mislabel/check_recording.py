"""
================================================================================
Visualizzazione con ecg_plot di una registrazione (PTB-XL, Georgia o China),
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

    # China/PhysioNet gia' convertito da china_test_builder.py:
    python check_recording.py \
        --ecg_id physionet_JS00001 \
        --china_root /path/to/china_test \
        --transform V1_V2
================================================================================
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
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
from data_prep.common import (
    load_ptbxl_metadata,
    discover_records_georgia,
    load_and_preprocess_ptbxl,
    load_and_preprocess_georgia,
)

SAMPLING_RATE = 500


def load_china_recording(ecg_id: str, china_root: str) -> np.ndarray:
    """
    Carica tutti i beat salvati per una registrazione China.

    Il China builder salva ogni beat come array (12, window_len), mentre il
    resto dello script usa il formato (window_len, 12). Il risultato ha forma
    (n_beats, window_len, 12), perche' il dataset non conserva il segnale
    continuo completo della registrazione.
    """
    root = Path(china_root)
    data_dir = root / "test" if (root / "test").is_dir() else root
    index_path = data_dir / "beats_index.csv"
    segments_dir = data_dir / "segments"

    if not index_path.exists():
        raise FileNotFoundError(f"Index China non trovato in: {index_path}")
    if not segments_dir.is_dir():
        raise FileNotFoundError(f"Directory segmenti China non trovata in: {segments_dir}")

    index_df = pd.read_csv(index_path)
    required_cols = {"filename", "ecg_id"}
    missing = required_cols - set(index_df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in {index_path}: {sorted(missing)}")

    recording_df = index_df[index_df["ecg_id"].astype(str) == ecg_id]
    if recording_df.empty:
        raise ValueError(f"Registrazione China '{ecg_id}' non trovata in {index_path}")

    beats = []
    for filename in recording_df["filename"].astype(str):
        segment_path = segments_dir / filename
        if not segment_path.exists():
            raise FileNotFoundError(f"Segmento China non trovato: {segment_path}")

        segment = np.load(segment_path).astype(np.float32)
        if segment.ndim != 2:
            raise ValueError(f"Shape inattesa per {segment_path}: {segment.shape}")
        if segment.shape[0] == 12:
            segment = segment.T
        elif segment.shape[1] != 12:
            raise ValueError(f"Shape inattesa per {segment_path}: {segment.shape}")
        beats.append(segment)

    return np.stack(beats)


def load_recording(
    ecg_id: str,
    ptbxl_root: str | None = None,
    georgia_root: str | None = None,
    china_root: str | None = None,
    preprocess: bool = True,
) -> np.ndarray:
    """
    Carica il segnale (n_samples, 12) di una registrazione a partire dal suo
    ecg_id (es. 'ptbxl_12001' o 'georgia_JS00001'), nell'ordine derivazioni
    standard I,II,III,aVR,aVL,aVF,V1..V6 (coerente con lead_misplacement.py
    e con dataset_builder.py). Per China ritorna invece tutti i beat con forma
    (n_beats, window_len, 12).

    preprocess=True  -> stesso preprocessing (filtri + notch) visto dal modello
    preprocess=False -> segnale grezzo WFDB, per una lettura "clinica" più pulita
    """
    if ecg_id.startswith("physionet_"):
        if china_root is None:
            raise ValueError("Per un ecg_id China e' necessario specificare --china_root")
        return load_china_recording(ecg_id, china_root)

    if ecg_id.startswith("ptbxl_"):
        if ptbxl_root is None:
            raise ValueError("Per un ecg_id PTB-XL e' necessario specificare --ptbxl_root")
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
        if georgia_root is None:
            raise ValueError("Per un ecg_id Georgia e' necessario specificare --georgia_root")
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
        raise ValueError(
            f"ecg_id non riconosciuto (prefisso atteso 'ptbxl_', 'georgia_' o "
            f"'physionet_'): {ecg_id}"
        )

    return signal.astype(np.float32)


def plot_recording(signal: np.ndarray, sample_rate: int, title: str):
    """Plot di un'intera registrazione nel layout standard di ecg_plot."""
    if signal.ndim == 3:
        # China salva i battiti separatamente: li ricongiungiamo per
        # visualizzare l'intera registrazione in un'unica figura.
        signal = np.concatenate(signal, axis=0)
    elif signal.ndim != 2:
        raise ValueError(f"Shape inattesa per il segnale da visualizzare: {signal.shape}")

    ecg_plot.plot(signal.T, sample_rate=sample_rate, title=title, columns=2)
    ecg_plot.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ecg_id", required=True,
                        help="es. ptbxl_12001, georgia_JS00001, physionet_JS00001")
    parser.add_argument("--ptbxl_root", default=None)
    parser.add_argument("--georgia_root", default=None)
    parser.add_argument("--china_root", default=None,
                        help="Directory del dataset China/PhysioNet generato da china_test_builder.py")
    parser.add_argument("--transform", default=None,
                        help="Se specificato (es. V1_V2, RA_LA...), mostra anche il tracciato "
                             "dopo aver applicato questa trasformazione, per confronto")
    parser.add_argument("--raw", action="store_true",
                        help="Mostra il segnale grezzo WFDB invece di quello preprocessato dal modello")
    args = parser.parse_args()

    is_china = args.ecg_id.startswith("physionet_")
    if is_china != (args.china_root is not None):
        raise ValueError("Per China usare --china_root con un ecg_id 'physionet_'; "
                         "per PTB-XL/Georgia non usare --china_root")

    signal = load_recording(
        args.ecg_id,
        args.ptbxl_root,
        args.georgia_root,
        args.china_root,
        preprocess=not args.raw,
    )

    plot_recording(signal, SAMPLING_RATE, title=f"{args.ecg_id} - originale (database)")

    if args.transform:
        transformed = apply_transform(signal, args.transform)
        plot_recording(transformed, SAMPLING_RATE, title=f"{args.ecg_id} - dopo {args.transform}")


if __name__ == "__main__":
    main()