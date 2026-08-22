"""
Stima le statistiche per derivazione del training set (PTB-XL) e
rimappa linearmente il test set (Georgia) in modo che la sua
distribuzione di ampiezza per ciascuna derivazione coincida con
quella del training set.

Trasformazione applicata per ciascun lead l:
    x'_l = a_l * x_l + b_l

con:
    centro  = mediana
    scala   = IQR (75° - 25° percentile)
    a_l = scale_train_l / scale_test_l
    b_l = center_train_l - a_l * center_test_l

Assunzione sul formato dei file
--------------------------------
Ogni file .npy in segments_dir ha shape (12, n_samples), coerente con
beats[local_idx].T salvato dallo script di preprocessing Georgia.

Uso
---
python map_test_to_train_distribution.py \
    --train_segments_dir /path/to/ptbxl/segments \
    --test_segments_dir  /path/to/georgia/segments \
    --output_dir         /path/to/georgia_mapped \
    --n_samples 5000
"""

import argparse
import glob
import os
import shutil

import numpy as np

LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def list_npy_files(segments_dir: str):
    files = sorted(glob.glob(os.path.join(segments_dir, "*.npy")))
    if len(files) == 0:
        raise FileNotFoundError(f"Nessun file .npy trovato in {segments_dir}")
    return files


def load_sampled_signals(segments_dir: str, n_samples: int):
    """Carica un sottoinsieme casuale di beat per stimare le statistiche
    senza dover caricare l'intero dataset in RAM.
    Ritorna array (n_beats_campionati, n_leads, n_samples_per_beat)."""
    files = list_npy_files(segments_dir)
    rng = np.random.default_rng(0)

    if len(files) > n_samples:
        idx = rng.choice(len(files), size=n_samples, replace=False)
        files = [files[i] for i in idx]

    arrs = []
    ref_shape = None
    for f in files:
        arr = np.load(f)
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(
                f"Shape incoerente in {f}: atteso {ref_shape}, trovato {arr.shape}."
            )
        arrs.append(arr)

    return np.stack(arrs, axis=0)


def compute_lead_stats(data: np.ndarray):
    """data: (n_beats, n_leads, n_samples) -> (center, scale), shape (n_leads,)."""
    n_leads = data.shape[1]
    flat = data.transpose(1, 0, 2).reshape(n_leads, -1)

    center = np.median(flat, axis=1)
    # IQR è una misura robusta agli outlier (valori molto grandi/piccoli)
    # viene utilizzata per stimare la scala dei segnali.
    q1 = np.percentile(flat, 25, axis=1)
    q3 = np.percentile(flat, 75, axis=1)
    scale = q3 - q1
    scale = np.where(scale < 1e-8, 1e-8, scale)  # evita divisioni per zero

    return center, scale


def apply_mapping_to_dir(segments_dir: str, output_segments_dir: str,
                          a: np.ndarray, b: np.ndarray):
    os.makedirs(output_segments_dir, exist_ok=True)
    files = list_npy_files(segments_dir)

    for f in files:
        arr = np.load(f)  # (n_leads, n_samples)
        mapped = (arr * a[:, None] + b[:, None]).astype(arr.dtype)
        np.save(os.path.join(output_segments_dir, os.path.basename(f)), mapped)

    return len(files)


def print_diagnostics(train_center, test_center, a, b):
    print("\n{:<6} {:>12} {:>12} {:>10} {:>10}".format(
        "Lead", "train_med", "test_med", "a (scale)", "b (shift)"))
    for i, name in enumerate(LEAD_NAMES):
        print("{:<6} {:>12.4f} {:>12.4f} {:>10.4f} {:>10.4f}".format(
            name, train_center[i], test_center[i], a[i], b[i]))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_segments_dir", required=True)
    parser.add_argument("--test_segments_dir", required=True)
    parser.add_argument("--test_index_csv", default=None,
                         help="(opzionale) beats_index.csv del test set, copiato nell'output")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=5000,
                         help="numero di beat campionati per stimare le statistiche")
    args = parser.parse_args()

    print(f"Campionamento training set da: {args.train_segments_dir}")
    train_data = load_sampled_signals(args.train_segments_dir, args.n_samples)
    # to make the train shape as the test shape
    train_data = train_data.transpose(0, 2, 1)  # (n_beats, n_samples, n_leads)
    train_center, train_scale = compute_lead_stats(train_data)

    print(f"Campionamento test set da: {args.test_segments_dir}")
    test_data = load_sampled_signals(args.test_segments_dir, args.n_samples)
    test_center, test_scale = compute_lead_stats(test_data)

    a = train_scale / test_scale
    b = train_center - a * test_center

    print_diagnostics(train_center, test_center, a, b)

    os.makedirs(args.output_dir, exist_ok=True)
    out_segments_dir = os.path.join(args.output_dir, "segments")
    n_mapped = apply_mapping_to_dir(args.test_segments_dir, out_segments_dir, a, b)
    print(f"Rimappati {n_mapped} beat in: {out_segments_dir}")

    if args.test_index_csv and os.path.exists(args.test_index_csv):
        dst = os.path.join(args.output_dir, os.path.basename(args.test_index_csv))
        shutil.copy(args.test_index_csv, dst)
        print(f"Indice copiato in: {dst}")


if __name__ == "__main__":
    main()
