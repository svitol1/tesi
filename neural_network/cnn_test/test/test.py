"""
================================================================================
Script di testing e valutazione
================================================================================

Esegue l'inferenza sui segmenti ECG caricandoli da disco batch per batch
ed applica un'aggregazione a livello di registrazione (soglia >50%).

Risolve il problema di Out-Of-Memory / Kill del processo su dataset di grandi dimensioni.

Uso da riga di comando:
    python test.py --data_dir data_prep/combined_dataset/test \
        --weights ecg_gcn_weights.pt \
        --classes normal RA_LA LA_LL RA_LL V1_V2 \
        --batch_size 64
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from cnn_model import ECG_CNN
except ImportError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from cnn_model import ECG_CNN


class ECGFileDataset(Dataset):
    """
    Dataset PyTorch con caricamento 'Lazy' da disco per evitare
    l'esaurimento della RAM su dataset enormi.
    """
    def __init__(self, df: pd.DataFrame, segments_dir: Path, label_to_idx: dict, expected_len: int = 500):
        self.df = df
        self.segments_dir = segments_dir
        self.label_to_idx = label_to_idx
        self.expected_len = expected_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seg_path = self.segments_dir / row["filename"]

        if not seg_path.exists():
            # Ritorna array vuoto/zero in caso di file mancante
            return torch.zeros((12, self.expected_len), dtype=torch.float32), -1, str(row["ecg_id"]), False

        try:
            seg = np.load(seg_path).astype(np.float32)

            # Normalizzazione shape a (12, expected_len)
            if seg.ndim == 2 and seg.shape[1] == 12 and seg.shape[0] == self.expected_len:
                seg = seg.T

            if seg.shape != (12, self.expected_len) or np.isnan(seg).any():
                return torch.zeros((12, self.expected_len), dtype=torch.float32), -1, str(row["ecg_id"]), False

            # Normalizzazione z-score per canale
            mean = seg.mean(axis=1, keepdims=True)
            std = seg.std(axis=1, keepdims=True)
            std[std == 0] = 1e-8
            seg = (seg - mean) / std

            label_idx = self.label_to_idx[str(row["label"])]
            return torch.from_numpy(seg), label_idx, str(row["ecg_id"]), True

        except Exception:
            return torch.zeros((12, self.expected_len), dtype=torch.float32), -1, str(row["ecg_id"]), False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing ECG_CNN per rilevazione malposizionamento elettrodi"
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
        "--batch_size", type=int, default=64, help="Dimensione del batch per l'inferenza"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Numero di thread per la lettura parallela da disco"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Limite opzionale sul numero massimo di segmenti da testare"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Elenco classi nell'ORDINE usato in training.",
    )
    return parser.parse_args()


def predict_recordings(
    recording_predictions: dict, recording_ground_truth: dict, threshold: float = 0.50
):
    """
    Aggrega le predizioni dei singoli segmenti a livello di intera registrazione.
    """
    rec_true = []
    rec_pred = []
    recording_summary = []

    for rec_id, seg_preds in recording_predictions.items():
        total_segments = len(seg_preds)
        counts = Counter(seg_preds)

        most_common_class, most_common_count = counts.most_common(1)[0]
        ratio = most_common_count / total_segments

        final_pred = most_common_class if ratio > threshold else most_common_class
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

    LABEL_TO_IDX = {name: i for i, name in enumerate(args.classes)}
    IDX_TO_LABEL = {i: name for name, i in LABEL_TO_IDX.items()}

    root = Path(args.data_dir)
    index_path = root / "beats_index.csv"
    segments_dir = root / "segments"

    if not index_path.exists():
        raise FileNotFoundError(f"Index non trovato in: {index_path}")

    print(f"--> Lettura indice da: {index_path}")
    df = pd.read_csv(index_path)

    if args.max_samples is not None and args.max_samples < len(df):
        print(f"--> Limite applicato: test sui primi {args.max_samples} segmenti su {len(df)}")
        df = df.iloc[:args.max_samples]

    test_dataset = ECGFileDataset(
        df=df,
        segments_dir=segments_dir,
        label_to_idx=LABEL_TO_IDX,
        expected_len=500
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    # Caricamento del modello
    num_classes = len(LABEL_TO_IDX)
    print(f"--> Caricamento pesi modello da: {args.weights}")
    model = ECG_CNN(
        hidden_channels=(32, 64, 128),
        dropout=0.18281203311944025,
        num_classes=num_classes,
        use_mlp_classifier=False
    ).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Streaming inferenza e accumulo leggero
    rec_preds_dict = {}
    rec_gt_dict = {}
    total_valid_segments = 0

    print("--> Avvio inferenza batch per batch in streaming...")
    with torch.no_grad():
        for x_batch, label_batch, ecg_id_batch, valid_mask in tqdm(test_loader, desc="Esecuzione Test"):
            # Filtra solo i campioni validi caricati correttamente da disco
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue

            x_batch = x_batch[valid_indices].to(device)
            out = model(x_batch)
            preds = out.argmax(dim=1).cpu().numpy()

            valid_labels = label_batch[valid_indices].numpy()
            valid_ids = [ecg_id_batch[i] for i in valid_indices.numpy()]

            for rec_id, seg_pred, seg_gt in zip(valid_ids, preds, valid_labels):
                if rec_id not in rec_preds_dict:
                    rec_preds_dict[rec_id] = []
                    rec_gt_dict[rec_id] = seg_gt
                rec_preds_dict[rec_id].append(seg_pred)
                total_valid_segments += 1

    print(f"--> Completati {total_valid_segments} segmenti validi su {len(rec_preds_dict)} registrazioni uniche.")

    if not rec_preds_dict:
        raise RuntimeError("Nessun segmento valido elaborato.")

    # Aggregazione Majority Voting > 50%
    rec_true, rec_pred, summary_df = predict_recordings(
        rec_preds_dict, rec_gt_dict, threshold=0.50
    )

    # Stampa del Report
    target_names = [IDX_TO_LABEL[i] for i in range(num_classes)]
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

    # Salvataggio del Summary CSV
    output_csv = "test_recordings_summary.csv"
    summary_df["ground_truth_label"] = summary_df["ground_truth"].map(IDX_TO_LABEL)
    summary_df["predicted_label"] = summary_df["predicted"].map(IDX_TO_LABEL)
    summary_df.to_csv(output_csv, index=False)
    print(f"\n--> Report di dettaglio salvato con successo in '{output_csv}'")


if __name__ == "__main__":
    main()