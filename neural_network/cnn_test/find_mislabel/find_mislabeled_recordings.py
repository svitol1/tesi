"""
================================================================================
Analisi del training set (a livello di INTERA REGISTRAZIONE, non di singolo
beat) per individuare registrazioni probabilmente mal-etichettate nel
database sorgente (PTB-XL / Georgia).

Motivazione
-----------
Il train set, come costruito da dataset_builder.py, contiene solo 2 beat per
registrazione. Valutare la correttezza di una registrazione sulla base di 1-2
beat e' pero' fragile: un singolo beat puo' essere misclassificato per rumore,
morfologia atipica o errore di allineamento del picco R, senza che questo
abbia nulla a che fare con un malposizionamento reale degli elettrodi.

Questo script riesegue quindi, per le registrazioni del TRAIN set, esattamente
cio' che test.py fa gia' per il test set: estrae TUTTI i beat validi della
registrazione a partire dal segnale grezzo e aggrega le predizioni con
majority voting (>50%). In questo modo il risultato per ogni registrazione
e' definito in modo robusto dell'intera registrazione, non di un singolo beat.

Logica
------
Per ogni registrazione appartenente al train (o qualsiasi set vengano inserito)
set (identificata a partire da final_dataset_index.csv):

1. Si ricarica il segnale grezzo (NON trasformato) da PTB-XL/Georgia e si
    estraggono tutti i beat validi con rilevamento dei picchi + segment_beats.
2. Si esegue l'inferenza su tutti i beat e si aggrega con majority voting
   -> "predizione grezza" per l'intera registrazione.
3. Se P_raw != "normal", la registrazione e' un CANDIDATO: il modello,
   vedendo la registrazione così com'è nel database (presunta corretta),
   ritiene comunque che ci sia un malposizionamento di tipo P_raw.
4. Test di conferma: si applica la trasformazione P_raw (quella predetta
   dal modello) a TUTTI i beat della registrazione, si rifa' inferenza e si
   aggrega di nuovo con majority voting -> P_confirm.
   Se P_confirm torna a "normal", e' un forte indizio che la registrazione
   avesse GIA' fisicamente lo scambio P_raw: applicare sinteticamente lo
   stesso scambio lo ha "annullato", riportando il segnale a una geometria
   dei lead normale.

Output: un CSV con una riga per registrazione analizzata (predizione
grezza, se candidata, esito del test di conferma).

Uso
---
    # PTB-XL + Georgia (percorso originale):
    python find_mislabeled_recordings.py \
        --ptbxl_root /path/to/ptbxl \
        --georgia_root /path/to/georgia \
        --index_csv data_prep/combined_dataset/train/final_dataset_index.csv \
        --weights ecg_cnn_weights.pt \
        --classes normal RA_LA LA_LL RA_LL V1_V2 \
        --batch_size 64

    # China/PhysioNet gia' convertito da china_test_builder.py:
    python find_mislabeled_recordings.py \
        --china_root /path/to/china_test \
        --china_csv /path/to/china_recording_predictions.csv \
        --weights ecg_cnn_weights.pt \
        --classes normal RA_LA LA_LL RA_LL V1_V2 \
        --batch_size 64
================================================================================
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# This script may be executed directly from nested folders such as
# cnn_test/find_mislabel/ or from the repo root. In both cases, Python only
# knows about the script directory, not the repository root. Add both the repo
# root and the cnn_test directory so imports work regardless of the cwd.
script_path = Path(__file__).resolve()
project_root = script_path.parents[2]
cnn_test_dir = script_path.parents[1]
for extra_path in (project_root, cnn_test_dir):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

try:
    from cnn_dataset import ECGSegmentDataset
    from cnn_model import ECG_CNN
except ImportError:
    from cnn_dataset import ECGSegmentDataset
    from cnn_model import ECG_CNN

try:
    from data_prep.common import (
        LEAD_NAMES_PTBXL,
        LEAD_NAMES_GEORGIA,
        load_ptbxl_metadata,
        discover_records_georgia,
        load_and_preprocess_ptbxl,
        load_and_preprocess_georgia,
        detect_r_peaks_v1_v5_average,
        segment_beats,
    )
    from data_prep.lead_misplacement import apply_transform
except ImportError:
    from data_prep.common import (
        LEAD_NAMES_PTBXL,
        LEAD_NAMES_GEORGIA,
        load_ptbxl_metadata,
        discover_records_georgia,
        load_and_preprocess_ptbxl,
        load_and_preprocess_georgia,
        detect_r_peaks_v1_v5_average,
        segment_beats,
    )
    from data_prep.lead_misplacement import apply_transform

NORMAL_LABEL = "normal"
VOTE_THRESHOLD = 0.50
LEAD_NAMES_PHYSIONET = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Trova registrazioni del train set probabilmente mal-etichettate, "
                    "analizzando l'intera registrazione (majority voting) invece dei soli 2 beat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ptbxl_root", type=str,
                   help="Root del database PTB-XL (usato insieme a --georgia_root)")
    p.add_argument("--georgia_root", type=str,
                   help="Root del database Georgia (usato insieme a --ptbxl_root)")
    p.add_argument("--china_root", type=str,
                   help="Directory del dataset China/PhysioNet generato da china_test_builder.py")
    p.add_argument("--china_csv", type=str,
                   help="CSV con le registrazioni China da analizzare (deve contenere recording_id)")
    p.add_argument("--index_csv", type=str,
                    help="Path a final_dataset_index.csv dello split che si vuole analizzare"
                    "(output di dataset_builder.py)")
    p.add_argument("--weights", type=str, required=True, help="Pesi del modello (.pt)")
    p.add_argument("--classes", nargs="+", required=True,
                    help="Classi nell'ORDINE usato in training (stesso ordine di train.py / test.py)")
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--max_recordings", type=int, default=None,
                    help="Limite opzionale sul numero di registrazioni da analizzare (utile per debug rapido)")
    p.add_argument("--output_csv", type=str, default="suspicious_recordings_full.csv")
    return p.parse_args()


# --------------------------------------------------------------------------
# Risoluzione delle registrazioni del train set verso i file grezzi
# --------------------------------------------------------------------------

def get_train_recording_tasks(train_index_csv: str, ptbxl_root: str, georgia_root: str):
    """
    Legge final_dataset_index.csv del train set ed estrae l'insieme unico di
    registrazioni (ecg_id), risolvendole verso i rispettivi file grezzi
    PTB-XL / Georgia. Ritorna una lista di task, ognuno un dict con le info
    necessarie per ricaricare il segnale grezzo di quella registrazione.
    """
    train_df = pd.read_csv(train_index_csv)
    required_cols = {"ecg_id"}
    missing = required_cols - set(train_df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in {train_index_csv}: {sorted(missing)}")

    unique_ecg_ids = sorted(train_df["ecg_id"].astype(str).unique().tolist())

    ptbxl_raw_ids, georgia_names = [], []
    unrecognized = []
    for key in unique_ecg_ids:
        if key.startswith("ptbxl_"):
            ptbxl_raw_ids.append(int(key[len("ptbxl_"):]))
        elif key.startswith("georgia_"):
            georgia_names.append(key[len("georgia_"):])
        else:
            unrecognized.append(key)
    if unrecognized:
        print(f"Attenzione: {len(unrecognized)} ecg_id con prefisso non riconosciuto, ignorati "
              f"(es. {unrecognized[:3]})")

    tasks = []

    # ---- PTB-XL ----
    ptbxl_meta = load_ptbxl_metadata(ptbxl_root)
    missing_ptbxl = 0
    for raw_id in ptbxl_raw_ids:
        if raw_id not in ptbxl_meta.index:
            missing_ptbxl += 1
            continue
        row = ptbxl_meta.loc[raw_id]
        tasks.append({
            "ecg_id": f"ptbxl_{raw_id}",
            "source": "ptbxl",
            "patient_id": f"ptbxl_{row['patient_id']}",
            "lead_names": LEAD_NAMES_PTBXL,
            "loader": lambda row=row: load_and_preprocess_ptbxl(ptbxl_root, row, notch_freq=50.0),
        })
    if missing_ptbxl:
        print(f"Attenzione: {missing_ptbxl} ecg_id PTB-XL del train set non trovati in ptbxl_database.csv")

    # ---- Georgia ----
    georgia_records = discover_records_georgia(georgia_root)  # [(name, path), ...]
    georgia_path_by_name = {name: path for name, path in georgia_records}
    missing_georgia = 0
    for name in georgia_names:
        record_path = georgia_path_by_name.get(name)
        if record_path is None:
            missing_georgia += 1
            continue
        tasks.append({
            "ecg_id": f"georgia_{name}",
            "source": "georgia",
            "patient_id": f"georgia_{name}",
            "lead_names": LEAD_NAMES_GEORGIA,
            "loader": lambda record_path=record_path: load_and_preprocess_georgia(record_path, notch_freq=60.0),
        })
    if missing_georgia:
        print(f"Attenzione: {missing_georgia} ecg_id Georgia del train set non trovati sotto {georgia_root}")

    return tasks


def get_china_recording_tasks(china_root: str, china_csv: str):
    """
    Legge `recording_id` dal CSV dei risultati della fase di test e costruisce
    un task solo per quelle registrazioni. I battiti corrispondenti vengono
    poi recuperati da `beats_index.csv` e dai file in `segments/` creati da
    china_test_builder.py.

    Sono accettati sia la directory radice dell'output (contenente `test/`)
    sia direttamente la directory `test/`.
    """
    root = Path(china_root)
    data_dir = root / "test" if (root / "test").is_dir() else root
    index_path = data_dir / "beats_index.csv"
    segments_dir = data_dir / "segments"

    if not index_path.exists():
        raise FileNotFoundError(f"Index China non trovato in: {index_path}")
    if not segments_dir.is_dir():
        raise FileNotFoundError(f"Directory dei segmenti China non trovata in: {segments_dir}")

    selected_df = pd.read_csv(china_csv)
    if "recording_id" not in selected_df.columns:
        raise ValueError(f"Colonna 'recording_id' mancante in {china_csv}")
    selected_ids = set(selected_df["recording_id"].dropna().astype(str))
    if not selected_ids:
        raise ValueError(f"Nessun recording_id valido trovato in {china_csv}")

    china_df = pd.read_csv(index_path)
    required_cols = {"filename", "ecg_id"}
    missing = required_cols - set(china_df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in {index_path}: {sorted(missing)}")

    # Filtra prima del groupby: i segmenti delle altre registrazioni non
    # vengono mai caricati e non entrano nel ciclo di inferenza.
    china_df["ecg_id"] = china_df["ecg_id"].astype(str)
    available_ids = set(china_df["ecg_id"])
    missing_ids = sorted(selected_ids - available_ids)
    if missing_ids:
        print(
            f"Attenzione: {len(missing_ids)} recording_id del CSV non trovati "
            f"in {index_path} (es. {missing_ids[:3]})"
        )
    china_df = china_df[china_df["ecg_id"].isin(selected_ids)]

    tasks = []
    for ecg_id, recording_df in china_df.groupby("ecg_id", sort=True):
        filenames = recording_df["filename"].astype(str).tolist()
        patient_id = str(recording_df["patient_id"].iloc[0]) if "patient_id" in recording_df else str(ecg_id)

        def load_beats(filenames=filenames):
            loaded_beats = []
            for filename in filenames:
                segment = np.load(segments_dir / filename).astype(np.float32)
                if segment.ndim != 2:
                    raise ValueError(f"Segmento China non bidimensionale: {filename}")
                if segment.shape[0] == 12:
                    segment = segment.T
                elif segment.shape[1] != 12:
                    raise ValueError(f"Shape inattesa per segmento China {filename}: {segment.shape}")
                loaded_beats.append(segment)
            return np.stack(loaded_beats)

        tasks.append({
            "ecg_id": str(ecg_id),
            "source": "china",
            "patient_id": patient_id,
            "lead_names": LEAD_NAMES_PHYSIONET,
            "loader": load_beats,
            "presegmented": True,
        })

    return tasks


# --------------------------------------------------------------------------
# Inferenza + majority voting a livello di registrazione
# --------------------------------------------------------------------------

def predict_beats(model, beats_leads_last: np.ndarray, device, batch_size: int) -> np.ndarray:
    """
    Esegue l'inferenza su tutti i beat di UNA registrazione.
    beats_leads_last: (n_beats, T, 12), come ritornato da segment_beats.
    Ritorna l'array degli indici di classe predetti (n_beats,).
    """
    segments = np.transpose(beats_leads_last, (0, 2, 1)).astype(np.float32)  # -> (n_beats, 12, T)
    dummy_labels = np.zeros(len(segments), dtype=np.int64)  # non usate, richieste solo dall'interfaccia del Dataset

    dataset = ECGSegmentDataset(
        segments=segments,
        labels=dummy_labels,
        normalize=True,
        expected_length=segments.shape[2],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            out = model(x_batch)
            preds.extend(out.argmax(dim=1).cpu().numpy())
    return np.array(preds)


def majority_vote(pred_indices: np.ndarray, idx_to_label: dict, threshold: float = VOTE_THRESHOLD):
    """
    Aggrega le predizioni dei singoli beat di una registrazione con la
    stessa regola di predict_recordings() in test.py (maggioranza, soglia
    >50%, fallback alla maggioranza relativa se nessuna classe supera la
    soglia).
    """
    counts = Counter(pred_indices)
    most_common_idx, most_common_count = counts.most_common(1)[0]
    ratio = most_common_count / len(pred_indices)
    label = idx_to_label[most_common_idx]  # fallback = maggioranza relativa, come in test.py
    return label, ratio


def apply_transform_to_all_beats(beats_leads_last: np.ndarray, label: str) -> np.ndarray:
    """
    Applica apply_transform a tutti i beat di una registrazione in un colpo
    solo: opera solo sull'ultimo asse (le 12 derivazioni), quindi e' gia'
    vettorizzata su qualunque dimensione precedente (n_beats, T, 12).
    """
    return apply_transform(beats_leads_last, label)


def load_model(weights_path: str, num_classes: int, device):
    model = ECG_CNN(
        hidden_channels=(64, 128, 256),
        dropout=0.17937371217945058,
        num_classes=num_classes,
        use_mlp_classifier=False,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()

    # China e' un dataset gia' segmentato: quando vengono indicati root e CSV,
    # e' la sola sorgente usata e non servono ne' l'indice train ne' i database raw.
    if args.china_root and args.china_csv:
        tasks_source = "china"
    elif args.ptbxl_root and args.georgia_root and args.index_csv:
        tasks_source = "ptbxl/georgia"
    else:
        raise ValueError(
            "Specificare insieme --china_root e --china_csv oppure specificare insieme "
            "--ptbxl_root, --georgia_root e --index_csv"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Device: {device}")

    LABEL_TO_IDX = {name: i for i, name in enumerate(args.classes)}
    IDX_TO_LABEL = {i: name for name, i in LABEL_TO_IDX.items()}
    num_classes = len(args.classes)

    print(f"--> Caricamento modello da: {args.weights}")
    model = load_model(args.weights, num_classes, device)

    if tasks_source == "china":
        print(f"--> Lettura registrazioni China selezionate da: {args.china_csv}")
        tasks = get_china_recording_tasks(args.china_root, args.china_csv)
    else:
        print(f"--> Risoluzione registrazioni train set da: {args.index_csv}")
        tasks = get_train_recording_tasks(args.index_csv, args.ptbxl_root, args.georgia_root)
    print(f"--> {len(tasks)} registrazioni da analizzare.")

    if args.max_recordings is not None:
        tasks = tasks[: args.max_recordings]
        print(f"--> Limitate a {len(tasks)} registrazioni (--max_recordings).")

    results = []
    skipped = []

    for task in tqdm(tasks, desc="Analisi registrazioni train"):
        ecg_id = task["ecg_id"]
        try:
            loaded_data = task["loader"]()
            if np.isnan(loaded_data).any():
                skipped.append((ecg_id, "segnale_contiene_NaN"))
                continue
        except Exception as e:
            skipped.append((ecg_id, f"errore: {e}"))
            continue

        # I dati China sono gia' beat (n_beats, T, 12)
        if task.get("presegmented", False):
            beats = loaded_data
        else:
            r_peaks = detect_r_peaks_v1_v5_average(loaded_data, task["lead_names"])
            beats, _ = segment_beats(loaded_data, r_peaks)
        if beats.shape[0] == 0:
            skipped.append((ecg_id, "nessun_battito_valido"))
            continue
        n_beats = beats.shape[0]

        # ---- Passo 1: predizione grezza (registrazione presunta corretta) ----
        raw_pred_idx = predict_beats(model, beats, device, args.batch_size)
        raw_label, raw_ratio = majority_vote(raw_pred_idx, IDX_TO_LABEL)
        is_candidate = raw_label != NORMAL_LABEL

        row = {
            "ecg_id": ecg_id,
            "patient_id": task["patient_id"],
            "source": task["source"],
            "n_beats": n_beats,
            "raw_pred_label": raw_label,
            "raw_pred_ratio": raw_ratio,
            "is_candidate": is_candidate,
            "confirm_pred_label": None,
            "confirm_pred_ratio": None,
            "confirmed": False,
        }

        # ---- Passo 2: test di conferma (solo per i candidati) ----
        if is_candidate:
            transformed_beats = apply_transform_to_all_beats(beats, raw_label)
            confirm_pred_idx = predict_beats(model, transformed_beats, device, args.batch_size)
            confirm_label, confirm_ratio = majority_vote(confirm_pred_idx, IDX_TO_LABEL)

            row["confirm_pred_label"] = confirm_label
            row["confirm_pred_ratio"] = confirm_ratio
            row["confirmed"] = confirm_label == NORMAL_LABEL

        results.append(row)

    if skipped:
        print(f"\nAttenzione: {len(skipped)} registrazioni saltate (segnale illeggibile o senza battiti validi).")

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)

    n_total = len(results_df)
    n_candidates = int(results_df["is_candidate"].sum()) if n_total else 0
    n_confirmed = int(results_df["confirmed"].sum()) if n_total else 0

    print("\n" + "=" * 60)
    print(" RISULTATI (livello di intera registrazione, majority voting)")
    print("=" * 60)
    print(f"Registrazioni analizzate               : {n_total}")
    print(f"Candidate (P_raw != normal)             : {n_candidates}")
    print(f"Confermate dal test di riapplicazione   : {n_confirmed}")
    print(f"\nReport di dettaglio salvato in: {args.output_csv}")


if __name__ == "__main__":
    main()
