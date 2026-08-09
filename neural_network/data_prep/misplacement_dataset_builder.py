"""
Prendiamo in input l'indice dei battiti già segmentati (output di
build_ptbxl_beat_dataset.py: beats_index.csv + cartella segments/) 
e generiamo il dataset finale per l'addestramento del modello di 
rilevamento del malposizionamento degli elettrodi.

REGOLE APPLICATE
----------------------------------
1. La classe (normale / tipo di scambio) viene decisa UNA VOLTA per
   ogni registrazione (ecg_id), non per singolo battito. Di conseguenza
   tutti i battiti di una stessa registrazione condividono la stessa
   etichetta: o sono tutti normali, o sono tutti trasformati con lo
   stesso tipo di scambio.
2. Per ogni battito viene salvata SOLO la versione corrispondente alla
   classe assegnata alla registrazione: mai sia la versione originale
   sia quella trasformata dello stesso segmento.
3. L'assegnazione delle classi alle registrazioni è randomizzata con
   seed fisso e bilanciata tra le classi.

USO
---
python generate_misplacement_dataset.py \
    --beats_index /path/to/output/beats_index.csv \
    --segments_dir /path/to/output/segments \
    --output_dir /path/to/final_dataset \
    --classes normal RA_LA LA_LL RA_LL V1_V2 \
    --class_weights 0.5 0.125 0.125 0.125 0.125
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from lead_misplacement import apply_transform, MISPLACEMENT_TRANSFORMS

FILENAME_DIGITS = 7


def assign_classes_per_recording(ecg_ids: np.ndarray, classes: list,
                                  weights: list, seed: int = 42) -> dict:
    """
    Assegna una classe a ciascuna registrazione (ecg_id), UNA volta sola.
    Ritorna un dizionario {ecg_id: classe}.

    L'assegnazione è randomizzata ma riproducibile (seed fisso) e segue
    le proporzioni indicate in `weights` (normalizzati automaticamente).
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    unique_ids = np.unique(ecg_ids)
    assigned = rng.choice(classes, size=len(unique_ids), p=weights)
    return dict(zip(unique_ids, assigned))


def build_final_dataset(beats_index_csv: str, segments_dir: str, output_dir: str,
                         classes: list, class_weights: list, seed: int = 42):
    unknown = set(classes) - set(MISPLACEMENT_TRANSFORMS.keys())
    if unknown:
        raise ValueError(f"Classi sconosciute: {unknown}. "
                          f"Disponibili: {list(MISPLACEMENT_TRANSFORMS.keys())}")

    index_df = pd.read_csv(beats_index_csv)

    out_segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(out_segments_dir, exist_ok=True)

    # assegna una classe a ogni registrazione (non a ogni battito)
    ecg_id_to_class = assign_classes_per_recording(
        index_df["ecg_id"].values, classes, class_weights, seed=seed
    )
    index_df["label"] = index_df["ecg_id"].map(ecg_id_to_class)

    # controllo di coerenza: ogni ecg_id deve avere UNA sola classe
    n_labels_per_record = index_df.groupby("ecg_id")["label"].nunique()
    assert (n_labels_per_record == 1).all(), (
        "Trovate registrazioni con più di un'etichetta: violazione della regola "
        "'tutti i battiti di una registrazione devono avere la stessa classe'."
    )

    # applica la trasformazione e salva UNA SOLA versione per battito
    final_rows = []
    for global_idx, row in tqdm(index_df.iterrows(), total=len(index_df),
                                 desc="Generazione dataset finale"):
        beat = np.load(os.path.join(segments_dir, row["filename"]))
        transformed = apply_transform(beat, row["label"])

        out_filename = f"seg{global_idx:0{FILENAME_DIGITS}d}.npy"
        np.save(os.path.join(out_segments_dir, out_filename), transformed)

        final_rows.append({
            "filename": out_filename,
            "ecg_id": row["ecg_id"],
            "patient_id": row["patient_id"],
            "beat_idx_in_record": row["beat_idx_in_record"],
            "strat_fold": row["strat_fold"],
            "label": row["label"],
        })

    final_df = pd.DataFrame(final_rows)
    final_csv_path = os.path.join(output_dir, "final_dataset_index.csv")
    final_df.to_csv(final_csv_path, index=False)

    print("\nCompletato.")
    print(f"Registrazioni totali : {len(ecg_id_to_class)}")
    print(f"Battiti totali : {len(final_df)}")
    print("Distribuzione classi (per registrazione):")
    class_counts = pd.Series(list(ecg_id_to_class.values())).value_counts()
    for cls, count in class_counts.items():
        print(f"{cls:10s}: {count} registrazioni")
    print(f"Cartella segmenti : {out_segments_dir}")
    print(f"Indice finale (CSV) : {final_csv_path}")

    return final_csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beats_index", required=True,
                         help="CSV prodotto da beat_dataset_builder.py")
    parser.add_argument("--segments_dir", required=True,
                         help="Cartella segments/ prodotta da beat_dataset_builder.py")
    parser.add_argument("--output_dir", required=True,
                         help="Cartella dove salvare il dataset finale")
    parser.add_argument("--classes", nargs="+", required=True,
                         help=f"Elenco classi da generare. Disponibili: "
                              f"{list(MISPLACEMENT_TRANSFORMS.keys())}")
    parser.add_argument("--class_weights", nargs="+", type=float, default=None,
                         help="Pesi relativi delle classi (stessa cardinalità di --classes). "
                              "Default: distribuzione uniforme.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights = args.class_weights if args.class_weights else [1.0] * len(args.classes)
    if len(weights) != len(args.classes):
        raise ValueError("--class_weights deve avere la stessa lunghezza di --classes")

    build_final_dataset(
        beats_index_csv=args.beats_index,
        segments_dir=args.segments_dir,
        output_dir=args.output_dir,
        classes=args.classes,
        class_weights=weights,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
