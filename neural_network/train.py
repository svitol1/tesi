"""
================================================================================
Training loop per la GCN pura definita in model.py, sui dati costruiti
tramite build_combined_dataset.py (PTB-XL + Georgia combinati).
================================================================================

A differenza della versione precedente, lo split train/val NON viene
piu' fatto qui tramite strat_fold: build_combined_dataset.py produce gia'
cartelle separate train/ e val/, ciascuna con la propria
final_dataset_index.csv e segments/. Qui ci limitiamo a caricarle.

Il test finale (tutti i battiti per registrazione + majority voting) va
eseguito separatamente con test/test.py sulla cartella test/ prodotta
dallo stesso builder.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path

from torch_geometric.loader import DataLoader

from model import ECG_GCN
from dataset import ECGGraphDataset, build_signed_ecg_graph_topology

WEIGHTS_PATH = Path("ecg_gcn_weights.pt")

def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        answer = input(prompt + suffix).strip().lower()
    except EOFError:
        return default

    if answer == "":
        return default
    return answer in {"y", "yes", "s", "si"}


def load_split(
    split_root: str,
    expected_len: int = 500,
    label_to_idx: dict | None = None,
):
    """Carica un singolo split (train/ o val/) prodotto da
    dataset_builder.py.

    Expects:
    - index CSV a <split_root>/final_dataset_index.csv
    - segmenti npy a <split_root>/segments/<filename>

    Se `label_to_idx` e' None, viene costruito dalle etichette trovate
    in questo split (tipicamente usato per il train set). Se e' fornito,
    viene riusato cosi' com'e' (usato per il val set, per garantire che
    gli indici numerici delle classi combacino con quelli del train set)
    e viene sollevato un errore se il val set contiene etichette non
    presenti nel train set.
    """
    root = Path(split_root)
    index_path = root / "final_dataset_index.csv"
    segments_dir = root / "segments"

    if not index_path.exists():
        raise FileNotFoundError(f"Index non trovato: {index_path}")
    if not segments_dir.exists():
        raise FileNotFoundError(f"Cartella segmenti non trovata: {segments_dir}")

    df = pd.read_csv(index_path)
    required_cols = {"filename", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")

    found_labels = sorted(df["label"].astype(str).unique().tolist())

    if label_to_idx is None:
        label_to_idx = {name: i for i, name in enumerate(found_labels)}
    else:
        unknown = set(found_labels) - set(label_to_idx.keys())
        if unknown:
            raise ValueError(
                f"Etichette in {index_path} non presenti nel train set: {sorted(unknown)}"
            )

    segments = []
    labels = []
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

        # Compatibilita' con dataset salvati come (T, 12): li convertiamo a (12, T).
        if seg.shape[1] == 12 and seg.shape[0] == 500:
            seg = seg.T

        if seg.shape[0] != 12:
            raise ValueError(
                f"Shape non valida per {seg_path.name}: {seg.shape}, atteso (12, T) "
            )
        if seg.shape[1] != expected_len:
            raise ValueError(
                f"Lunghezza non valida per {seg_path.name}: {seg.shape[1]}, "
                f"atteso {expected_len} (400 pre + 600 post a 500Hz = 500 campioni)"
            )

        segments.append(seg)
        labels.append(label_to_idx[str(row.label)])

    if not segments:
        raise RuntimeError(f"Nessun segmento valido caricato da {split_root}")

    if missing_files:
        print(f"Attenzione: {missing_files} file segmenti mancanti in {split_root}")

    segments = np.stack(segments, axis=0)
    labels = np.asarray(labels, dtype=np.int64)

    print(f"[{root.name}] Segmenti caricati: {segments.shape[0]} | "
          f"shape singolo segmento: {segments.shape[1:]}")
    if "source" in df.columns:
        print(f"[{root.name}] Provenienza: "
              f"{df['source'].value_counts().to_dict()}")

    return segments, labels, label_to_idx


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    load_old_weights = ask_yes_no(
        f"Vuoi caricare pesi precedenti da {WEIGHTS_PATH}?",
        default=False,
    )

    # --------------------------------------------------------------------
    # 1) Dati
    # --------------------------------------------------------------------
    SEGMENT_LEN = 500      # 400ms pre + 600ms post picco R = 1 secondo a 500 Hz
    DATASET_ROOT = "data_prep/"

    train_segments, train_labels, label_to_idx = load_split(
        split_root=f"{DATASET_ROOT}/train",
        expected_len=SEGMENT_LEN,
    )
    val_segments, val_labels, _ = load_split(
        split_root=f"{DATASET_ROOT}/val",
        expected_len=SEGMENT_LEN,
        label_to_idx=label_to_idx,
    )

    NUM_CLASSES = len(label_to_idx)
    print(f"Classi ({NUM_CLASSES}): {label_to_idx}")
    print(f"Train segments: {len(train_labels)} | Val segments: {len(val_labels)}")

    # Generazione topologia basata sui vettori 3D delle derivazioni
    edge_index_pos, edge_weight_pos, edge_index_neg, edge_weight_neg = build_signed_ecg_graph_topology()

    train_dataset = ECGGraphDataset(
        segments=train_segments,
        labels=train_labels,
        edge_index_pos=edge_index_pos,
        edge_weight_pos=edge_weight_pos,
        edge_index_neg=edge_index_neg,
        edge_weight_neg=edge_weight_neg,
        normalize=True,
    )
    val_dataset = ECGGraphDataset(
        segments=val_segments,
        labels=val_labels,
        edge_index_pos=edge_index_pos,
        edge_weight_pos=edge_weight_pos,
        edge_index_neg=edge_index_neg,
        edge_weight_neg=edge_weight_neg,
        normalize=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

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
        dropout=0.23116470198461048,
        use_mlp_classifier=False
    ).to(device)

    if load_old_weights:
        weights_path = WEIGHTS_PATH
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0010532404166862267, weight_decay=1.053909018491971e-05)

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=3
    )

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

    # --------------------------------------------------------------------
    # 3) Training loop
    # --------------------------------------------------------------------
    run_training = ask_yes_no("Vuoi eseguire il training del modello?", default=True)
    if run_training:
        EPOCHS = 25
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = run_epoch(train_loader, train=True)
            val_loss, val_acc = run_epoch(val_loader, train=False)
            #scheduler.step(val_loss)
            print(f"Epoch {epoch:02d} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # --------------------------------------------------------------------
        # 4) Salvataggio pesi
        # --------------------------------------------------------------------
        torch.save(model.state_dict(), "ecg_gcn_weights.pt")
        print("Pesi salvati in ecg_gcn_weights.pt")
    else:
        print("Training saltato su richiesta utente.")

    print(
        "\nPer valutare il modello sul test set (tutti i battiti per "
        "registrazione + majority voting), esegui separatamente:\n"
        f"  python test/test.py --data_dir {DATASET_ROOT}/test --weights ecg_gcn_weights.pt"
    )


if __name__ == "__main__":
    main()