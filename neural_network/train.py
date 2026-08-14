"""
================================================================================
Training loop per la GCN pura definita in model.py, sui dati costruiti
tramite dataset.py.
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path

from torch_geometric.loader import DataLoader

from model import ECG_GCN
from dataset import ECGGraphDataset, build_ecg_graph_topology


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        answer = input(prompt + suffix).strip().lower()
    except EOFError:
        return default

    if answer == "":
        return default
    return answer in {"y", "yes", "s", "si"}


def load_misplacement_data(
    dataset_root: str = "data_prep/misplacement_dataset",
    expected_len: int = 500,
):
    """Load ECG segments and labels from misplacement_dataset.

    Expects:
    - index CSV at <dataset_root>/final_dataset_index.csv
    - npy segments at <dataset_root>/segments/<filename>
    """
    root = Path(dataset_root)
    index_path = root / "final_dataset_index.csv"
    segments_dir = root / "segments"

    if not index_path.exists():
        raise FileNotFoundError(f"Index non trovato: {index_path}")
    if not segments_dir.exists():
        raise FileNotFoundError(f"Cartella segmenti non trovata: {segments_dir}")

    df = pd.read_csv(index_path)
    required_cols = {"filename", "label", "patient_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")

    label_names = sorted(df["label"].astype(str).unique().tolist())
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    segments = []
    labels = []
    patient_ids = []
    missing_files = 0

    for row in df.itertuples(index=False):
        seg_path = segments_dir / row.filename
        if not seg_path.exists():
            missing_files += 1
            continue

        seg = np.load(seg_path).astype(np.float32)
        if seg.ndim != 2:
            raise ValueError(
                f"Shape non valida per {seg_path.name}: {seg.shape}, atteso array 2D"
            )

        # Compatibilita con dataset salvati come (T, 12): li convertiamo a (12, T).
        if seg.shape[1] == 12 and seg.shape[0] != 12:
            seg = seg.T

        if seg.shape[0] != 12:
            raise ValueError(
                f"Shape non valida per {seg_path.name}: {seg.shape}, atteso (12, T) "
                "oppure (T, 12)"
            )
        if seg.shape[1] != expected_len:
            raise ValueError(
                f"Lunghezza non valida per {seg_path.name}: {seg.shape[1]}, "
                f"atteso {expected_len} (400 pre + 600 post a 500Hz = 500 campioni)"
            )

        segments.append(seg)
        labels.append(label_to_idx[str(row.label)])
        patient_ids.append(row.patient_id)

    if not segments:
        raise RuntimeError("Nessun segmento valido caricato dal dataset")

    if missing_files:
        print(f"Attenzione: {missing_files} file segmenti mancanti nel dataset")

    segments = np.stack(segments, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    patient_ids = np.asarray(patient_ids)

    print(f"Segmenti caricati: {segments.shape[0]} | shape singolo segmento: {segments.shape[1:]}")
    print(f"Classi ({len(label_names)}): {label_to_idx}")

    return segments, labels, patient_ids, label_to_idx


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    load_old_weights = ask_yes_no(
        "Vuoi caricare pesi precedenti da best_weights.pt?",
        default=False,
    )

    # --------------------------------------------------------------------
    # 1) Dati
    # --------------------------------------------------------------------
    SEGMENT_LEN = 500      # 400ms pre + 600ms post picco R = 1 secondo a 500 Hz
    segments, labels, patient_ids, label_to_idx = load_misplacement_data(
        dataset_root="data_prep/misplacement_dataset",
        expected_len=SEGMENT_LEN,
    )
    NUM_CLASSES = len(label_to_idx)

    # Generazione topologia basata sui vettori 3D delle derivazioni
    edge_index, edge_weight = build_ecg_graph_topology(
        mode="vector_geometric",
        threshold=0.0
    )

    dataset = ECGGraphDataset(
        segments=segments,
        labels=labels,
        edge_index=edge_index,
        edge_weight=edge_weight,
        normalize=True
    )

    # Split per Paziente (per evitare Data Leakage)
    unique_patients = np.unique(patient_ids)
    rng = np.random.default_rng(0)
    shuffled_patients = rng.permutation(unique_patients)
    n_train_patients = max(1, int(0.8 * len(shuffled_patients)))
    train_patients = set(shuffled_patients[:n_train_patients])

    train_idx = np.where(np.isin(patient_ids, list(train_patients)))[0]
    val_idx = np.where(~np.isin(patient_ids, list(train_patients)))[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Split train/val non valido: uno dei due insiemi è vuoto")

    train_dataset = dataset[train_idx.tolist()]
    val_dataset = dataset[val_idx.tolist()]

    print(f"Train segments: {len(train_idx)} | Val segments: {len(val_idx)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # --------------------------------------------------------------------
    # 2) Modello, ottimizzatore, loss
    # --------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")

    model = ECG_GCN(
        node_feat_dim=64,
        hidden_dim=128,
        num_classes=NUM_CLASSES,
        num_gnn_layers=2,
        use_attention=False,
        dropout=0.3,
    ).to(device)

    if load_old_weights:
        weights_path = Path("best_weights.pt")
        if weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict)
                print(f"Pesi caricati da {weights_path}")
            except Exception as exc:
                print(f"Impossibile caricare i pesi da {weights_path}: {exc}")
                print("Continuo con training da zero.")
        else:
            print(f"Checkpoint non trovato in {weights_path}, training da zero.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # --------------------------------------------------------------------
    # 3) Training loop
    # --------------------------------------------------------------------
    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for data in loader:
                data = data.to(device)
                out = model(data)                     # [batch_size, NUM_CLASSES]
                y = data.y.view(-1)                    # [batch_size]
                loss = criterion(out, y)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * y.size(0)
                correct += (out.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        return total_loss / total, correct / total

    EPOCHS = 15
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # --------------------------------------------------------------------
    # 4) Salvataggio pesi
    # --------------------------------------------------------------------
    torch.save(model.state_dict(), "ecg_gcn_weights.pt")
    print("Pesi salvati in ecg_gcn_weights.pt")


if __name__ == "__main__":
    main()
