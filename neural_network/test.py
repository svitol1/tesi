"""
================================================================================
Script di testing e valutazione
================================================================================

Esegue l'inferenza sui segmenti ECG e applica un'aggregazione a livello
di registrazione (con soglia >50%).

Uso da riga di comando:
    python test.py --data_dir data_prep/misplacement_dataset --weights ecg_gcn_weights.pt
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader

from dataset import ECGGraphDataset, build_ecg_graph_topology
from model import ECG_GCN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing ECG_GCN per rilevazione malposizionamento elettrodi"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path alla cartella contenente final_dataset_index.csv e la sottocartella segments/",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best_weights.pt",
        help="Path al file dei pesi salvati (.pt)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Dimensione del batch per l'inferenza"
    )
    return parser.parse_args()


def load_test_data(data_root: str, expected_len: int = 500):
    """
    Carica i segmenti di test e mantieni la traccia delle registrazioni (patient_id/recording_id).
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

    # Mappatura etichette
    label_names = sorted(df["label"].astype(str).unique().tolist())
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}

    segments = []
    labels = []
    recording_ids = []

    for row in df.itertuples(index=False):
        seg_path = segments_dir / row.filename
        if not seg_path.exists():
            continue

        seg = np.load(seg_path).astype(np.float32)
        if seg.shape[1] != expected_len:
            continue

        segments.append(seg)
        labels.append(label_to_idx[str(row.label)])
        recording_ids.append(str(row.patient_id))  # Identificativo registrazione

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

        # Trova la classe più frequente nei segmenti della registrazione
        most_common_class, most_common_count = counts.most_common(1)[0]
        ratio = most_common_count / total_segments

        # Applicazione della regola del >50%
        if ratio > threshold:
            final_pred = most_common_class
        else:
            # Fallback in caso di parità perfetta o nessuna classe > 50%:
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

    # 1. Caricamento dati di test
    print(f"--> Caricamento dataset di test da: {args.data_dir}")
    SEGMENT_LEN = 500
    segments, labels, recording_ids, label_to_idx, idx_to_label = load_test_data(
        args.data_dir, expected_len=SEGMENT_LEN
    )

    num_classes = len(label_to_idx)
    print(f"--> Caricati {len(segments)} segmenti totali appartenenti a {len(set(recording_ids))} registrazioni.")
    print(f"--> Classi individuate ({num_classes}): {label_to_idx}")

    # 2. Topologia del Grafo Vettoriale e Dataset PyG
    edge_index, edge_weight = build_ecg_graph_topology(mode="vector_geometric", threshold=0.0)
    test_dataset = ECGGraphDataset(
        segments=segments,
        labels=labels,
        edge_index=edge_index,
        edge_weight=edge_weight,
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
        dropout=0.3,
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

    print("\n" + "=" * 60)
    print(" RISULTATI A LIVELLO DI SINGOLA REGISTRAZIONE (>50% Majority Voting)")
    print("=" * 60)

    rec_acc = (rec_true == rec_pred).mean() * 100
    print(f"\nAccuratezza totale sulle Registrazioni: {rec_acc:.2f}%\n")

    print("--- Classification Report ---")
    print(
        classification_report(
            rec_true, rec_pred, target_names=target_names, digits=4, zero_division=0
        )
    )

    print("\n--- Matrice di Confusione ---")
    cm = confusion_matrix(rec_true, rec_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    # Salvataggio del report CSV di dettaglio per le registrazioni
    output_csv = "test_recordings_summary.csv"
    summary_df["ground_truth_label"] = summary_df["ground_truth"].map(idx_to_label)
    summary_df["predicted_label"] = summary_df["predicted"].map(idx_to_label)
    summary_df.to_csv(output_csv, index=False)
    print(f"\n--> Report di dettaglio salvato con successo in '{output_csv}'")


if __name__ == "__main__":
    main()