"""
Small training experiment to check whether ECG_GCN can memorize a tiny dataset.

This is a diagnostic, not a validation procedure: it intentionally trains and
evaluates on the same 16-32 segments. If the model cannot reach nearly 100%
training accuracy on this set, there is likely a problem in the data pipeline,
model, loss, or optimization setup.

Usage:
    python3 test_overfit.py
    python3 test_overfit.py --num_segments 16 --epochs 500
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

from dataset import ECGGraphDataset, build_signed_ecg_graph_topology
from model import ECG_GCN
from train import load_misplacement_data


SEGMENT_LENGTH = 500


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether ECG_GCN can overfit a tiny balanced subset"
    )
    parser.add_argument(
        "--data_root",
        default="data_prep/misplacement_dataset",
        help="Dataset directory containing final_dataset_index.csv and segments/",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=32,
        help="Number of segments to memorize (must be between 16 and 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of full-batch optimization steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for model initialization and subset selection",
    )
    return parser.parse_args()


def choose_balanced_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Choose a deterministic subset containing every available class."""
    if not 16 <= count <= 32:
        raise ValueError("--num_segments deve essere compreso tra 16 e 32")

    rng = np.random.default_rng(seed)
    class_indices = {
        label: rng.permutation(np.flatnonzero(labels == label))
        for label in np.unique(labels)
    }
    if count < len(class_indices):
        raise ValueError(
            f"Servono almeno {len(class_indices)} segmenti per includere tutte le classi"
        )

    selected = []
    positions = {label: 0 for label in class_indices}
    labels_in_order = sorted(class_indices)

    while len(selected) < count:
        added_in_round = False
        for label in labels_in_order:
            position = positions[label]
            available = class_indices[label]
            if position >= len(available):
                continue
            selected.append(int(available[position]))
            positions[label] += 1
            added_in_round = True
            if len(selected) == count:
                break
        if not added_in_round:
            raise ValueError("Il dataset non contiene abbastanza segmenti validi")

    return np.asarray(selected, dtype=np.int64)


def main():
    args = parse_args()
    if args.epochs <= 0 or args.lr <= 0:
        raise ValueError("--epochs e --lr devono essere positivi")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")

    segments, labels, _, label_to_idx = load_misplacement_data(
        dataset_root=args.data_root,
        expected_len=SEGMENT_LENGTH,
    )
    selected_indices = choose_balanced_indices(labels, args.num_segments, args.seed)
    selected_labels = labels[selected_indices]
    print(
        f"Subset: {len(selected_indices)} segmenti | "
        f"classi: {np.bincount(selected_labels).tolist()}"
    )
    print(f"Label mapping: {label_to_idx}")

    edge_index_pos, edge_weight_pos, edge_index_neg, edge_weight_neg = (
        build_signed_ecg_graph_topology()
    )
    dataset = ECGGraphDataset(
        segments=segments[selected_indices],
        labels=selected_labels,
        edge_index_pos=edge_index_pos,
        edge_weight_pos=edge_weight_pos,
        edge_index_neg=edge_index_neg,
        edge_weight_neg=edge_weight_neg,
        normalize=True,
    )
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    model = ECG_GCN(
        node_feat_dim=64,
        hidden_dim=64,
        num_classes=len(label_to_idx),
        num_gnn_layers=3,
        use_attention=False,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch = next(iter(loader)).to(device)
        optimizer.zero_grad()
        logits = model(batch)
        targets = batch.y.view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
            print(f"Epoch {epoch:03d} | loss={loss.item():.6f} | accuracy={accuracy:.4f}")

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader)).to(device)
        logits = model(batch)
        final_loss = criterion(logits, batch.y.view(-1)).item()
        final_predictions = logits.argmax(dim=1)
        final_accuracy = (final_predictions == batch.y.view(-1)).float().mean().item()

    print("\nOverfit check")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    if final_accuracy == 1.0 and final_loss < 0.01:
        print("PASS: il modello ha memorizzato il subset")
    else:
        print("FAIL: il modello non ha raggiunto loss < 0.01 e accuracy 100%")


if __name__ == "__main__":
    main()