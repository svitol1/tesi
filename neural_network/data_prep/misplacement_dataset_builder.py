"""
Prendiamo in input l'indice dei battiti già segmentati (output di
build_ptbxl_beat_dataset.py: beats_index.csv + cartella segments/)
e generiamo il dataset finale per l'addestramento del modello di
rilevamento del malposizionamento degli elettrodi.

REGOLE APPLICATE
----------------------------------
1. La classe (normale / tipo di scambio) viene decisa per SINGOLO
    segmento, ma per ogni registrazione c'è un vincolo: i due segmenti
    di ogni ecg_id devono ricevere due classi diverse.
2. Per ogni battito viene salvata SOLO la versione corrispondente alla
    classe assegnata al segmento: mai sia la versione originale
   sia quella trasformata dello stesso segmento.
3. L'assegnazione delle classi ai segmenti è randomizzata con
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

# numero di cifre usate nei nomi dei file di output (es. seg0000001.npy)
FILENAME_DIGITS = 7


def assign_classes_per_segment(index_df: pd.DataFrame, classes: list,
                               weights: list, seed: int = 42) -> pd.Series:
    """
    Assegna una classe a ciascun segmento, imponendo classi diverse
    all'interno della stessa registrazione.

    L'assegnazione è randomizzata ma riproducibile (seed fisso) e segue
    le proporzioni indicate in `weights` (normalizzati automaticamente).

    Vincoli applicati:
      - ogni ecg_id deve avere esattamente 2 segmenti;
      - i 2 segmenti dello stesso ecg_id ricevono 2 classi diverse.
    """
    # crea un generatore random riproducibile con il seed fornito
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    # normalizza i pesi in modo che sommino a 1
    weights = weights / weights.sum()

    labels = pd.Series(index=index_df.index, dtype=object)

    for ecg_id, group_df in index_df.groupby("ecg_id"):
        n_segments = len(group_df)
        if n_segments != 2:
            raise ValueError(
                f"ecg_id={ecg_id} ha {n_segments} segmenti, ma ne sono attesi 2"
            )
        if n_segments > len(classes):
            raise ValueError(
                "Numero classi insufficiente per assegnare etichette distinte "
                f"(ecg_id={ecg_id}, segmenti={n_segments}, classi={len(classes)})"
            )

        chosen = rng.choice(classes, size=n_segments, replace=False, p=weights)
        ordered_idx = group_df.sort_values("beat_idx_in_record").index.to_numpy()
        labels.loc[ordered_idx] = chosen

    return labels


def build_final_dataset(beats_index_csv: str, segments_dir: str, output_dir: str,
                         classes: list, class_weights: list, seed: int = 42):
    # verifica che le classi richieste esistano tra le trasformazioni disponibili
    unknown = set(classes) - set(MISPLACEMENT_TRANSFORMS.keys())
    if unknown:
        raise ValueError(f"Classi sconosciute: {unknown}. "
                          f"Disponibili: {list(MISPLACEMENT_TRANSFORMS.keys())}")

    index_df = pd.read_csv(beats_index_csv)

    out_segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(out_segments_dir, exist_ok=True)

    # assegna una classe a ogni segmento, forzando classi diverse per ecg_id
    index_df["label"] = assign_classes_per_segment(
        index_df=index_df,
        classes=classes,
        weights=class_weights,
        seed=seed,
    )

    # controllo di coerenza: ogni ecg_id deve avere esattamente DUE classi distinte
    n_labels_per_record = index_df.groupby("ecg_id")["label"].nunique()
    assert (n_labels_per_record == 2).all(), (
        "Trovate registrazioni senza due etichette distinte. "
        "Ogni ecg_id deve avere 2 segmenti con 2 classi diverse."
    )

    final_rows = []
    # itera su tutte le righe del DataFrame
    for global_idx, row in tqdm(index_df.iterrows(), total=len(index_df),
                                 desc="Generazione dataset finale"):
        beat = np.load(os.path.join(segments_dir, row["filename"]))
        # applica la trasformazione corrispondente all'etichetta assegnata
        transformed = apply_transform(beat, row["label"])

        out_filename = f"seg{global_idx:0{FILENAME_DIGITS}d}.npy"
        np.save(os.path.join(out_segments_dir, out_filename), transformed)

        # aggiunge una riga corrispondente al file appena salvato
        final_rows.append({
            "filename": out_filename,
            "ecg_id": row["ecg_id"],
            "patient_id": row["patient_id"],
            "beat_idx_in_record": row["beat_idx_in_record"],
            "strat_fold": row["strat_fold"],
            "label": row["label"],
        })

    # costruisce un DataFrame finale dalle righe raccolte
    final_df = pd.DataFrame(final_rows)
    final_csv_path = os.path.join(output_dir, "final_dataset_index.csv")
    final_df.to_csv(final_csv_path, index=False)

    print("\nCompletato.")
    print(f"Registrazioni totali : {index_df['ecg_id'].nunique()}")
    print(f"Battiti totali : {len(final_df)}")
    print("Distribuzione classi (per segmento):")
    # conta le classi assegnate per segmento
    class_counts = final_df["label"].value_counts()
    # stampa il conteggio per ogni classe
    for cls, count in class_counts.items():
        print(f"{cls:10s}: {count} segmenti")

    print("Combinazioni classi per registrazione:")
    pair_counts = (
        final_df.groupby("ecg_id")["label"]
        .apply(lambda s: " + ".join(sorted(s.tolist())))
        .value_counts()
    )
    for pair, count in pair_counts.items():
        print(f"{pair:20s}: {count} registrazioni")

    print(f"Cartella segmenti : {out_segments_dir}")
    print(f"Indice finale (CSV) : {final_csv_path}")

    # ritorna il percorso del CSV generato
    return final_csv_path


def main():
    # crea il parser degli argomenti usando il docstring del modulo come descrizione
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

    # se non vengono forniti i pesi, usa distribuzione uniforme
    weights = args.class_weights if args.class_weights else [1.0] * len(args.classes)
    # verifica che il numero di pesi combaci con il numero di classi
    if len(weights) != len(args.classes):
        raise ValueError("--class_weights deve avere la stessa lunghezza di --classes")

    # chiama la funzione principale che costruisce il dataset finale
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
