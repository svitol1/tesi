"""
Costruisce un China dataset trasformato da ``china_test/test``.

Ogni registrazione riceve una classe, quindi tutti i battiti
appartenenti ad essa saranno trasformati di conseguenza.

Esempio d'uso:

    python china_test_cleaner.py \
        --input_dir data_prep/china_test/test \
        --output_dir data_prep/china_test/transformed \
        --excluded_csv_dir results \
        --classes normal RA_LA LA_LL RA_LL V1_V2 \
        --class_percentages 0.5 0.125 0.125 0.125 0.125 \
        --seed 42

    Le percentuali possono sommarsi a 1 o 100.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ..common import load_excluded_records
    from ..lead_misplacement import MISPLACEMENT_TRANSFORMS, apply_transform
except ImportError:
    from common import load_excluded_records
    from lead_misplacement import MISPLACEMENT_TRANSFORMS, apply_transform


def normalize_percentages(percentages: list[float]) -> np.ndarray:
    # Accetta sia proporzioni (0.5) sia percentuali (50), ma lavora
    # internamente sempre con valori normalizzati che sommano a 1.
    values = np.asarray(percentages, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("--class_percentages must contain finite non-negative numbers")
    total = values.sum()
    if total <= 0 or not np.isclose(total, 1.0) and not np.isclose(total, 100.0):
        raise ValueError("--class_percentages must sum to 1 or 100")
    return values / total


def recording_class_counts(n_recordings: int, percentages: np.ndarray) -> np.ndarray:
    """ Le quantità attese sono spesso frazionarie, quindi assegniamo prima la parte
        intera e distribuiamo i recording rimasti alle classi con il resto maggiore.
    """
    expected = n_recordings * percentages
    counts = np.floor(expected).astype(int)
    remainder = n_recordings - counts.sum()
    if remainder:
        order = np.argsort(-(expected - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def assign_recording_classes(recording_ids: list[str], classes: list[str], percentages: list[float], seed: int) -> dict:
    """ assegna ad ogni classe la propria etichetta, in questo modo tutti
        i battiti della stessa registrazione ricevono la stessa classe.
    """
    normalized = normalize_percentages(percentages)
    counts = recording_class_counts(len(recording_ids), normalized)
    shuffled = list(recording_ids)
    np.random.default_rng(seed).shuffle(shuffled)

    assignment = {}
    start = 0
    for class_name, count in zip(classes, counts):
        for recording_id in shuffled[start:start + count]:
            assignment[recording_id] = class_name
        start += count
    return assignment


def build_dataset(input_dir: str, output_dir: str, classes: list[str], percentages: list[float],
                  seed: int, excluded_csv_dir: str | None = None,
                  excluded_csv_filename: str = "mislabels.csv") -> None:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    index_path = input_root / "beats_index.csv"
    input_segments = input_root / "segments"
    output_segments = output_root / "segments"

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    if not input_segments.is_dir():
        raise FileNotFoundError(f"Segments directory not found: {input_segments}")
    if len(classes) != len(percentages):
        raise ValueError("--classes and --class_percentages must have the same length")
    unknown = set(classes) - set(MISPLACEMENT_TRANSFORMS)
    if unknown:
        raise ValueError(f"Unknown classes: {sorted(unknown)}")
    normalize_percentages(percentages)

    index_df = pd.read_csv(index_path)
    required = {"filename", "ecg_id"}
    missing = required - set(index_df.columns)
    if missing:
        raise ValueError(f"{index_path} is missing required columns: {sorted(missing)}")

    excluded = (
        load_excluded_records(excluded_csv_dir, excluded_csv_filename)
        if excluded_csv_dir else set()
    )
    index_df["ecg_id"] = index_df["ecg_id"].astype(str)
    usable_df = index_df[~index_df["ecg_id"].isin(excluded)].copy()
    recording_ids = sorted(usable_df["ecg_id"].unique().tolist())
    # L'assegnazione viene fatta una sola volta per ecg_id e poi propagata a
    # tutti i segmenti appartenenti a quel recording.
    assignment = assign_recording_classes(recording_ids, classes, percentages, seed)
    usable_df["label"] = usable_df["ecg_id"].map(assignment)

    output_segments.mkdir(parents=True, exist_ok=True)
    for row in tqdm(usable_df.itertuples(index=False), total=len(usable_df), desc="Transforming segments"):
        source_path = input_segments / row.filename
        target_path = output_segments / row.filename
        if not source_path.exists():
            raise FileNotFoundError(f"Segment listed in index does not exist: {source_path}")
        segment = np.load(source_path)
        if segment.ndim != 2 or segment.shape[0] != 12:
            raise ValueError(f"Expected {source_path} to have shape (12, window_len), got {segment.shape}")
        # I file sono salvati leads-first (12, T), mentre apply_transform
        # considera le derivazioni nell'ultima dimensione.
        transformed = apply_transform(segment.T, row.label).T
        np.save(target_path, transformed)

    output_index = output_root / "beats_index.csv"
    usable_df.to_csv(output_index, index=False)
    skipped = sorted(excluded)
    pd.DataFrame(skipped, columns=["ecg_id"]).to_csv(output_root / "excluded_records.csv", index=False)

    counts = usable_df[["ecg_id", "label"]].drop_duplicates()["label"].value_counts()
    print(f"Recordings read: {len(recording_ids)}")
    print(f"Recordings excluded: {len(excluded & set(index_df['ecg_id']))}")
    print(f"Segments saved: {len(usable_df)}")
    print(f"Output index: {output_index}")
    for class_name in classes:
        print(f"  {class_name}: {int(counts.get(class_name, 0))} recordings")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--excluded_csv_dir", default=None,
                        help="directory dove cercare ricorsivamente")
    parser.add_argument("--excluded_csv_filename", default="mislabels.csv",
                        help="nome del CSV che cerca dentro --excluded_csv_dir")
    parser.add_argument("--classes", nargs="+", required=True, help="Classi da assegnare alle registrazioni")
    parser.add_argument("--class_percentages", nargs="+", type=float, required=True,
                        help="Proporzioni percentuali per classe. Somma a 1 o 100")
    parser.add_argument("--seed", type=int, default=42, help="seed utilizzato per mischaire le registrazioni")
    return parser.parse_args()


def main():
    args = parse_args()
    build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        classes=args.classes,
        percentages=args.class_percentages,
        seed=args.seed,
        excluded_csv_dir=args.excluded_csv_dir,
        excluded_csv_filename=args.excluded_csv_filename,
    )


if __name__ == "__main__":
    main()