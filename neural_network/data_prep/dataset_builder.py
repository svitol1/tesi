"""
FASE 2 - VELOCE - da rieseguire quante volte serve per ogni combinazione
di --classes / --class_weights / split / esclusioni.

Costruisce il dataset combinato PTB-XL + Georgia per l'addestramento e la
valutazione del modello di rilevamento malposizionamento elettrodi, A
PARTIRE DALLA CACHE prodotta da precompute_cache.py (segnali preprocessati
+ picchi R gia' rilevati). Non tocca piu' i dati grezzi: nessun I/O wfdb,
nessun filtraggio, nessun Pan-Tompkins, quindi e' molto piu' rapida della
fase 1.

Split (per registrazione/paziente, seed fisso), applicato SEPARATAMENTE a
PTB-XL e Georgia e poi unito:
    - train : 80% delle registrazioni di ciascun dataset (default)
    - val   : 10% delle registrazioni di ciascun dataset (default)
    - test  : il resto

Per PTB-XL lo split e' per PATIENT_ID (un paziente puo' avere piu'
registrazioni: split per ecg_id causerebbe data leakage). Per Georgia
patient_id == ecg_id, quindi lo split per registrazione e' equivalente.

TRAIN / VAL
-----------
Per ogni registrazione si usano ESATTAMENTE 2 battiti (3° e 5° picco R,
gia' identificati in cache), e si assegna una classe diversa a ciascuno dei
2 battiti (bilanciata secondo --class_weights, seed fisso), tra "normal" e
i tipi di malposizionamento in lead_misplacement.py. Viene salvata solo la
versione trasformata corrispondente alla classe assegnata.

TEST
----
Si usano TUTTI i battiti validi di ogni registrazione test, etichettati
"normal" senza alcuna trasformazione (la valutazione avviene per intera
registrazione con majority voting, vedi test/test.py).

ESCLUSIONE REGISTRAZIONI CONFERMATE ERRATE
-------------------------------------------
--excluded_csv_dir esclude dalla costruzione (in tutti gli split) le
registrazioni segnalate come "candidate" e poi CONFERMATE (confirmed=True)
da un giro di verifica col modello. Vive in questa fase (non nella cache)
apposta: la lista di esclusioni cambia man mano che si confermano nuovi
casi, e non deve richiedere di rifare il caching.

Output
------
<output_dir>/train/segments/*.npy + <output_dir>/train/final_dataset_index.csv
<output_dir>/val/segments/*.npy   + <output_dir>/val/final_dataset_index.csv
<output_dir>/test/segments/*.npy  + <output_dir>/test/beats_index.csv

Ogni file .npy ha shape (12, window_len), leads-first, pronto per
ECGGraphDataset senza bisogno di trasposizioni successive.

Uso
---
# una tantum, o quando cambiano dati grezzi/preprocessing:
python precompute_cache.py --ptbxl_root ... --georgia_root ... --cache_dir /path/to/cache

# quante volte serve, per ogni combinazione di classi/pesi/esclusioni:
python dataset_builder.py \
    --cache_dir /path/to/cache \
    --output_dir /path/to/combined_dataset \
    --classes normal RA_LA LA_LL RA_LL V1_V2 \
    --class_weights 0.5 0.125 0.125 0.125 0.125 \
    --excluded_csv_dir /path/to/csv_confirmati
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .common import (
        segment_beats, split_ids, load_excluded_records,
        select_train_val_peaks, assign_classes_per_recording,
    )
    from .lead_misplacement import apply_transform, MISPLACEMENT_TRANSFORMS
except ImportError:
    from common import (
        segment_beats, split_ids, load_excluded_records,
        select_train_val_peaks, assign_classes_per_recording,
    )
    from lead_misplacement import apply_transform, MISPLACEMENT_TRANSFORMS

FILENAME_DIGITS = 7


def load_cached_record(cache_dir, record_key):
    signal = np.load(os.path.join(cache_dir, "signals", f"{record_key}.npy"))
    r_peaks = np.load(os.path.join(cache_dir, "peaks", f"{record_key}.npy"))
    return signal, r_peaks


def process_split(split_name, cache_dir, meta_subset, output_dir,
                   classes=None, class_weights=None, seed=42, excluded_keys=None):
    """Processa un intero split (train / val / test) leggendo dalla cache
    e salva segmenti + indice CSV.

    meta_subset : sotto-DataFrame di metadata.csv gia' filtrato alle sole
        registrazioni di questo split (colonne: record_key, source,
        patient_id, n_samples, n_r_peaks).
    excluded_keys : set opzionale di record_key (es. 'ptbxl_14144',
        'georgia_E00238') da saltare completamente."""
    out_root = os.path.join(output_dir, split_name)
    segments_dir = os.path.join(out_root, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    is_test = split_name == "test"
    rows = []
    beat_counter = 0
    skipped = []

    # Le righe status='failed' non entrano nello split: il vecchio
    # precompute_cache.py non le scriveva in metadata.csv. Restano comunque
    # disponibili nel metadata completo per diagnosi e conteggi.
    n_slot = len(meta_subset)
    failed_mask = meta_subset["status"] != "ok"
    skipped.extend(
        (rk, f"fallita_in_cache: {motivo}")
        for rk, motivo in zip(meta_subset.loc[failed_mask, "record_key"],
                               meta_subset.loc[failed_mask, "motivo"])
    )
    meta_subset = meta_subset[~failed_mask]
    n_failed_in_cache = int(failed_mask.sum())

    if excluded_keys:
        n_before = len(meta_subset)
        meta_subset = meta_subset[~meta_subset["record_key"].isin(excluded_keys)]
        print(f"[{split_name}] Registrazioni escluse (confirmed=True): "
              f"{n_before - len(meta_subset)}")

    print(f"[{split_name}] Slot assegnati dallo split: {n_slot} | "
          f"fallite in cache: {n_failed_in_cache} | utilizzabili: {len(meta_subset)}")

    if not is_test:
        # ---- Prima fase: raccogliamo 2 battiti grezzi per registrazione
        # (PTB-XL + Georgia insieme), poi assegnamo le classi in modo
        # bilanciato su TUTTE le registrazioni di questo split.
        raw_beats = {}  # record_key -> (beats, used_peaks, source, patient_id)

        for _, meta_row in tqdm(meta_subset.iterrows(), total=len(meta_subset),
                                 desc=f"[{split_name}] Estrazione battiti"):
            record_key = meta_row["record_key"]
            signal, r_peaks = load_cached_record(cache_dir, record_key)
            selected = select_train_val_peaks(r_peaks)
            if selected is None:
                skipped.append((record_key, "picchi_insufficienti"))
                continue
            beats, used_peaks = segment_beats(signal, selected)
            if beats.shape[0] != 2:
                skipped.append((record_key, "picchi_fuori_dai_bordi_segnale"))
                continue
            raw_beats[record_key] = (beats, used_peaks, meta_row["source"], meta_row["patient_id"])

        assignment = assign_classes_per_recording(
            list(raw_beats.keys()), classes, class_weights, seed=seed
        )

        for record_key, (beats, used_peaks, source, patient_id) in tqdm(
            sorted(raw_beats.items()), desc=f"[{split_name}] Applicazione trasformazioni"
        ):
            labels = assignment[record_key]
            for local_idx in range(2):
                label = labels[local_idx]
                transformed = apply_transform(beats[local_idx], label)  # (T, 12)
                filename = f"seg{beat_counter:0{FILENAME_DIGITS}d}.npy"
                np.save(os.path.join(segments_dir, filename), transformed.T)  # -> (12, T)
                rows.append({
                    "filename": filename,
                    "ecg_id": record_key,
                    "patient_id": f"{source}_{patient_id}",
                    "source": source,
                    "beat_idx_in_record": local_idx,
                    "r_peak_sample": int(used_peaks[local_idx]),
                    "label": label,
                })
                beat_counter += 1

        index_df = pd.DataFrame(rows)
        index_csv_path = os.path.join(out_root, "final_dataset_index.csv")
        index_df.to_csv(index_csv_path, index=False)

    else:
        # ---- Test: tutti i battiti validi, etichetta "normal", nessuna
        # trasformazione (registrazioni reali, presumibilmente corrette).
        for _, meta_row in tqdm(meta_subset.iterrows(), total=len(meta_subset),
                                 desc=f"[{split_name}] Estrazione battiti"):
            record_key = meta_row["record_key"]
            source = meta_row["source"]
            patient_id = meta_row["patient_id"]
            signal, r_peaks = load_cached_record(cache_dir, record_key)
            if len(r_peaks) == 0:
                skipped.append((record_key, "nessun_battito_valido"))
                continue
            beats, used_peaks = segment_beats(signal, r_peaks)
            if beats.shape[0] == 0:
                skipped.append((record_key, "nessun_battito_valido"))
                continue
            for local_idx, r_sample in enumerate(used_peaks):
                filename = f"seg{beat_counter:0{FILENAME_DIGITS}d}.npy"
                np.save(os.path.join(segments_dir, filename), beats[local_idx].T)
                rows.append({
                    "filename": filename,
                    "ecg_id": record_key,
                    "patient_id": f"{source}_{patient_id}",
                    "source": source,
                    "beat_idx_in_record": local_idx,
                    "r_peak_sample": int(r_sample),
                    "label": "normal",
                })
                beat_counter += 1

        index_df = pd.DataFrame(rows)
        index_csv_path = os.path.join(out_root, "beats_index.csv")
        index_df.to_csv(index_csv_path, index=False)

    skipped_csv_path = os.path.join(out_root, "skipped_records.csv")
    pd.DataFrame(skipped, columns=["record_key", "motivo"]).to_csv(skipped_csv_path, index=False)

    print(f"\n[{split_name}] Completato.")
    print(f"  Battiti salvati       : {beat_counter}")
    print(f"  Registrazioni saltate : {len(skipped)}")
    print(f"  Cartella segmenti     : {segments_dir}")
    print(f"  Indice (CSV)          : {index_csv_path}")
    if not is_test:
        print("  Distribuzione classi:")
        for cls, count in index_df["label"].value_counts().items():
            print(f"    {cls:10s}: {count}")
    else:
        print(f"  Registrazioni totali (per majority voting): {index_df['patient_id'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", required=True,
                         help="Cartella prodotta da precompute_cache.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--classes", nargs="+",
                         default=["normal", "RA_LA", "LA_LL", "RA_LL", "V1_V2"])
    parser.add_argument("--class_weights", nargs="+", type=float, default=None)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--excluded_csv_dir", default=None,
                         help="Directory radice sotto cui cercare RICORSIVAMENTE i file "
                              "'mislabels.csv' (colonne 'ecg_id' e 'confirmed') prodotti dal "
                              "giro di verifica del modello. Se specificata, le registrazioni "
                              "con confirmed=True vengono escluse da tutti gli split. Se "
                              "omessa, nessuna esclusione viene applicata.")
    parser.add_argument("--excluded_csv_filename", default="mislabels.csv",
                         help="Nome del file da cercare ricorsivamente dentro "
                              "--excluded_csv_dir (default: 'mislabels.csv').")
    args = parser.parse_args()

    weights = args.class_weights if args.class_weights else [1.0] * len(args.classes)
    if len(weights) != len(args.classes):
        raise ValueError("--class_weights deve avere la stessa lunghezza di --classes")
    unknown = set(args.classes) - set(MISPLACEMENT_TRANSFORMS.keys())
    if unknown:
        raise ValueError(f"Classi sconosciute: {unknown}")

    excluded_keys = None
    if args.excluded_csv_dir:
        print("--> Caricamento registrazioni da escludere (confirmed=True)...")
        excluded_keys = load_excluded_records(args.excluded_csv_dir, args.excluded_csv_filename)
        print(f"  Registrazioni totali da escludere: {len(excluded_keys)}")

    print("--> Caricamento metadata dalla cache...")
    # metadata.csv contiene anche le registrazioni fallite, ma la popolazione
    # usata per lo split deve restare quella delle sole registrazioni
    # cachate con successo, come nella pipeline precedente.
    meta = pd.read_csv(os.path.join(args.cache_dir, "metadata.csv"))
    meta_for_split = meta[meta["status"] == "ok"]
    meta_ptbxl = meta_for_split[meta_for_split["source"] == "ptbxl"]
    meta_georgia = meta_for_split[meta_for_split["source"] == "georgia"]

    print("--> Split train/val/test per PAZIENTE (PTB-XL) e per registrazione (Georgia)...")
    patients_train, patients_val, patients_test = split_ids(
        meta_ptbxl["patient_id"].unique().tolist(), args.train_frac, args.val_frac, seed=args.seed
    )
    georgia_train, georgia_val, georgia_test = split_ids(
        meta_georgia["patient_id"].unique().tolist(), args.train_frac, args.val_frac, seed=args.seed
    )

    print(f"  PTB-XL  : train={len(patients_train)} val={len(patients_val)} test={len(patients_test)} "
          f"(split per paziente)")
    print(f"  Georgia : train={len(georgia_train)} val={len(georgia_val)} test={len(georgia_test)}")

    splits = [
        ("train", patients_train, georgia_train),
        ("val", patients_val, georgia_val),
        ("test", patients_test, georgia_test),
    ]

    for split_name, ptbxl_patient_ids, georgia_patient_ids in splits:
        meta_subset = pd.concat([
            meta_ptbxl[meta_ptbxl["patient_id"].isin(ptbxl_patient_ids)],
            meta_georgia[meta_georgia["patient_id"].isin(georgia_patient_ids)],
        ], ignore_index=True)

        process_split(
            split_name=split_name,
            cache_dir=args.cache_dir,
            meta_subset=meta_subset,
            output_dir=args.output_dir,
            classes=args.classes,
            class_weights=weights,
            seed=args.seed,
            excluded_keys=excluded_keys,
        )


if __name__ == "__main__":
    main()
