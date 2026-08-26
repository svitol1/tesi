"""
================================================================================
Script di testing e valutazione
================================================================================

Esegue l'inferenza sui segmenti ECG e applica un'aggregazione a livello
di registrazione (con soglia >50%).

Compatibile con il test set prodotto da build_combined_dataset.py
(cartella test/, con beats_index.csv contenente TUTTI i battiti validi
di ogni registrazione, etichetta "normal" per tutte).

Uso da riga di comando:
    python test.py --data_dir data_prep/combined_dataset/test --weights ecg_gcn_weights.pt \
        --classes normal RA_LA LA_LL RA_LL V1_V2
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader

try:
    from ..dataset import ECGGraphDataset, build_signed_ecg_graph_topology
    from ..model import ECG_GCN
except ImportError:
    # Allow running this file directly: python test/test.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from dataset import ECGGraphDataset, build_signed_ecg_graph_topology
    from model import ECG_GCN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing ECG_GCN per rilevazione malposizionamento elettrodi"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path alla cartella contenente beats_index.csv e la sottocartella segments/",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="ecg_gcn_weights.pt",
        help="Path al file dei pesi salvati (.pt)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Dimensione del batch per l'inferenza"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["normal", "RA_LA", "LA_LL", "RA_LL", "V1_V2"],
        help="Elenco classi nell'ORDINE usato in training (stesso ordine con cui "
             "train.py le ha enumerate da sorted(df['label'].unique()) sul train set). "
             "Verifica sempre contro il log di training prima di interpretare i risultati.",
    )
    return parser.parse_args()


def load_test_data(data_root: str, label_to_idx: dict, expected_len: int = 500):
    """
    Carica i segmenti di test e mantiene la traccia delle registrazioni
    (patient_id/recording_id) per l'aggregazione a majority voting.
    """
    root = Path(data_root)
    index_path = root / "beats_index.csv"
    segments_dir = root / "segments"

    if not index_path.exists():
        raise FileNotFoundError(f"Index non trovato in: {index_path}")
    if not segments_dir.exists():
        raise FileNotFoundError(f"Cartella segmenti non trovata in: {segments_dir}")

    df = pd.read_csv(index_path)
    required_cols = {"filename", "label", "patient_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")

    unknown_labels = set(df["label"].astype(str).unique()) - set(label_to_idx.keys())
    if unknown_labels:
        raise ValueError(f"Etichette nel CSV non presenti in label_to_idx: {sorted(unknown_labels)}")

    idx_to_label = {i: name for name, i in label_to_idx.items()}

    segments = []
    labels = []
    recording_ids = []
    skipped_missing = 0
    skipped_shape = 0
    skipped_nan = 0

    for row in df.itertuples(index=False):
        seg_path = segments_dir / row.filename
        if not seg_path.exists():
            skipped_missing += 1
            continue

        seg = np.load(seg_path).astype(np.float32)

        if seg.ndim != 2:
            skipped_shape += 1
            continue

        # Compatibilita' con segmenti salvati come (T, 12): li convertiamo a (12, T).
        if seg.shape[1] == 12 and seg.shape[0] == expected_len:
            seg = seg.T

        if seg.shape[0] != 12 or seg.shape[1] != expected_len:
            skipped_shape += 1
            continue

        if np.isnan(seg).any():
            skipped_nan += 1
            continue

        segments.append(seg)
        labels.append(label_to_idx[str(row.label)])
        recording_ids.append(str(row.patient_id))

    if skipped_missing:
        print(f"Attenzione: {skipped_missing} file segmenti mancanti su disco, saltati.")
    if skipped_shape:
        print(f"Attenzione: {skipped_shape} segmenti con shape inattesa, saltati.")
    if skipped_nan:
        print(f"Attenzione: {skipped_nan} segmenti contenenti NaN, saltati.")

    if not segments:
        raise RuntimeError(f"Nessun segmento valido caricato da {data_root}")

    segments = np.stack(segments, axis=0)
    labels = np.asarray(labels, dtype=np.int64)

    return segments, labels, recording_ids, label_to_idx, idx_to_label


def predict_recordings(
    recording_predictions: dict, recording_ground_truth: dict, threshold: float = 0.50
):
    """
    Aggrega le predizioni dei singoli segmenti a livello di intera registrazione.

    Regola: Se un tipo di inversione viene individuato in oltre il 50% dei
    segmenti di una registrazione, quella registrazione viene classificata con tale inversione.
    """
    rec_true = []
    rec_pred = []
    recording_summary = []

    for rec_id, seg_preds in recording_predictions.items():
        total_segments = len(seg_preds)
        counts = Counter(seg_preds)

        most_common_class, most_common_count = counts.most_common(1)[0]
        ratio = most_common_count / total_segments

        if ratio > threshold:
            final_pred = most_common_class
        else:
            # Fallback in caso di parita' perfetta o nessuna classe > 50%:
            # assegna comunque la classe di maggioranza relativa
            final_pred = most_common_class

        gt_label = recording_ground_truth[rec_id]

        rec_true.append(gt_label)
        rec_pred.append(final_pred)

        recording_summary.append(
            {
                "recording_id": rec_id,
                "total_segments": total_segments,
                "ground_truth": gt_label,
                "predicted": final_pred,
                "confidence_ratio": ratio,
                "correct": gt_label == final_pred,
            }
        )

    return np.array(rec_true), np.array(rec_pred), pd.DataFrame(recording_summary)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Utilizzo device: {device}")

    # Deve corrispondere ESATTAMENTE all'ordine/mapping usato in training.
    # train.py costruisce label_to_idx da sorted(df["label"].unique()) sul
    # train set: verifica il log di training se hai dubbi sull'ordine.
    LABEL_TO_IDX = {name: i for i, name in enumerate(args.classes)}

    print(f"--> Caricamento dataset di test da: {args.data_dir}")
    SEGMENT_LEN = 500
    segments, labels, recording_ids, label_to_idx, idx_to_label = load_test_data(
        args.data_dir, LABEL_TO_IDX, expected_len=SEGMENT_LEN
    )

    num_classes = len(LABEL_TO_IDX)
    print(f"--> Caricati {len(segments)} segmenti totali appartenenti a {len(set(recording_ids))} registrazioni.")
    print(f"--> Classi individuate ({num_classes}): {label_to_idx}")

    # 2. Topologia del Grafo Vettoriale e Dataset PyG
    edge_index_pos, edge_weight_pos, edge_index_neg, edge_weight_neg = build_signed_ecg_graph_topology()
    test_dataset = ECGGraphDataset(
        segments=segments,
        labels=labels,
        edge_index_pos=edge_index_pos,
        edge_weight_pos=edge_weight_pos,
        edge_index_neg=edge_index_neg,
        edge_weight_neg=edge_weight_neg,
        normalize=True,
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Caricamento Modello e Pesi
    print(f"--> Caricamento pesi modello da: {args.weights}")
    model = ECG_GCN(
            node_feat_dim=64,
            hidden_dim=128,
            num_classes=num_classes,
            num_gnn_layers=2,
            use_attention=False,
            dropout=0.23116470198461048,
            use_mlp_classifier=False
        ).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # 4. Inferenza sui singoli segmenti
    print("--> Esecuzione inferenza sui segmenti...")
    seg_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            preds = out.argmax(dim=1).cpu().numpy()
            seg_predictions.extend(preds)

    # 5. Raggruppamento predizioni per Registrazione
    rec_preds_dict = {}
    rec_gt_dict = {}

    for rec_id, seg_pred, seg_gt in zip(recording_ids, seg_predictions, labels):
        if rec_id not in rec_preds_dict:
            rec_preds_dict[rec_id] = []
            rec_gt_dict[rec_id] = seg_gt
        rec_preds_dict[rec_id].append(seg_pred)

    # 6. Aggregazione >50% a livello di Registrazione
    rec_true, rec_pred, summary_df = predict_recordings(
        rec_preds_dict, rec_gt_dict, threshold=0.50
    )

    # 7. Calcolo Metriche e Report
    target_names = [idx_to_label[i] for i in range(num_classes)]
    all_label_indices = list(range(num_classes))

    print("\n" + "=" * 60)
    print(" RISULTATI A LIVELLO DI SINGOLA REGISTRAZIONE (>50% Majority Voting)")
    print("=" * 60)

    rec_acc = (rec_true == rec_pred).mean() * 100
    print(f"\nAccuratezza totale sulle Registrazioni: {rec_acc:.2f}%\n")

    print("--- Classification Report ---")
    print(classification_report(
        rec_true, rec_pred, labels=all_label_indices,
        target_names=target_names, digits=4, zero_division=0
    ))

    print("\n--- Matrice di Confusione ---")
    cm = confusion_matrix(rec_true, rec_pred, labels=all_label_indices)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    # Distribuzione delle etichette vere nel test set: utile per capire se
    # il test set contiene solo "normal" (come per build_combined_dataset.py)
    # o anche altre classi.
    true_dist = pd.Series(rec_true).map(idx_to_label).value_counts()
    print("\n--- Distribuzione ground truth per registrazione ---")
    print(true_dist)

    # Salvataggio del report CSV di dettaglio per le registrazioni
    output_csv = "test_recordings_summary.csv"
    summary_df["ground_truth_label"] = summary_df["ground_truth"].map(idx_to_label)
    summary_df["predicted_label"] = summary_df["predicted"].map(idx_to_label)
    summary_df.to_csv(output_csv, index=False)
    print(f"\n--> Report di dettaglio salvato con successo in '{output_csv}'")


if __name__ == "__main__":
    main()