import random
"""Test script: run batched inference on N random segments and print summary metrics.

Usage: python3 test.py
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cnn_pytorch import ecg_classifier


CLASS_NAMES = {
    0: "Normale / Blocco di Branca (N)",
    1: "Battito Ectopico Sopraventricolare (SVEB)",
    2: "Battito Ectopico Ventricolare (VEB)",
    3: "Battito di Fusione (F)",
}


def load_model(path: str, device: torch.device) -> torch.nn.Module:
    """Load a model checkpoint (supports raw state_dict or dict with 'model_state')."""
    model = ecg_classifier(num_classi=4, in_channels=2)
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint incompatibile con la rete a due derivazioni. "
            "Riesegui l'allenamento per generare un modello aggiornato."
        ) from exc
    return model.to(device)


def pad_or_trim(sig: np.ndarray, target: int) -> np.ndarray:
    """Center-pad or center-trim a 1D signal to `target` length.

    If the signal is shorter, pads with zeros equally on both sides.
    If longer, extracts a centered window of length `target`.
    """
    if sig.ndim == 2:
        return np.stack([pad_or_trim(channel, target) for channel in sig], axis=0)
    L = len(sig)
    if L == target:
        return sig
    if L < target:
        pad_left = (target - L) // 2
        pad_right = target - L - pad_left
        return np.pad(sig, (pad_left, pad_right), mode="constant")
    center = L // 2
    start = center - target // 2
    return sig[start : start + target]


def sample_inputs(csv_file: str, base_dir: str, n: int, target_length: int, seed=None) -> Tuple[np.ndarray, List[int], List[str]]:
    """Return (X, labels, paths) for n random rows in the metadata CSV.

    X is a numpy array shape (N, target_length).
    """
    df = pd.read_csv(csv_file)
    n = min(n, len(df))
    # If seed is None pandas will use a random seed, producing different samples each run.
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)

    inputs = []
    labels = []
    paths = []
    base = Path(base_dir)
    for _, row in sampled.iterrows():
        rel = row["file_path"]
        p = base / rel
        if not p.exists():
            continue
        sig = np.load(p)
        sig = pad_or_trim(sig, target_length)
        inputs.append(sig)
        labels.append(int(row["label"]))
        paths.append(rel)

    if len(inputs) == 0:
        return np.empty((0, target_length)), [], []

    return np.stack(inputs), labels, paths


def evaluate(model: torch.nn.Module, X: np.ndarray, batch_size: int = 32) -> Tuple[List[int], List[List[float]]]:
    """Run batched inference and return (preds, probs)."""
    device = next(model.parameters()).device
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    if X_t.ndim == 2:
        X_t = X_t.unsqueeze(1)
    elif X_t.ndim != 3:
        raise ValueError(f"Input non supportato: shape={tuple(X_t.shape)}")
    if X_t.size(1) != 2:
        raise ValueError(f"La rete si aspetta 2 derivazioni, ma l'input ne contiene {X_t.size(1)}")
    N = X_t.size(0)
    preds = []
    probs = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = X_t[i : i + batch_size].to(device)
            out = model(xb)
            p = F.softmax(out, dim=1)
            _, pr = torch.max(out, 1)
            preds.extend(pr.cpu().tolist())
            probs.extend(p.cpu().tolist())
    return preds, probs


def print_summary(labels: List[int], preds: List[int], probs: List[List[float]], paths: List[str]):
    """Print overall accuracy, per-class accuracy and confusion matrix."""
    N = len(labels)
    if N == 0:
        print("Nessun esempio da valutare.")
        return

    correct = sum(int(p == t) for p, t in zip(preds, labels))
    acc = correct / N * 100
    print("=" * 60)
    print(f"Test su {N} esempi - Accuracy: {acc:.2f}% ({correct}/{N})")
    print("-" * 60)

    num_classes = len(CLASS_NAMES)
    confusion = [[0] * num_classes for _ in range(num_classes)]
    per_total = [0] * num_classes
    per_correct = [0] * num_classes
    for t, p in zip(labels, preds):
        confusion[t][p] += 1
        per_total[t] += 1
        if t == p:
            per_correct[t] += 1

    for cls in range(num_classes):
        total = per_total[cls]
        correct_c = per_correct[cls]
        pct = (correct_c / total * 100) if total > 0 else 0.0
        print(f"Classe {cls} ({CLASS_NAMES[cls]}): {total} esempi - Acc: {pct:.2f}%")

    print("-" * 60)
    print("Matrice di confusione (righe=ground truth, colonne=predetti):")
    for row in confusion:
        print("\t".join(str(x) for x in row))

    # show some errors
    print("-" * 60)
    print("Esempi di errori (fino a 10):")
    shown = 0
    for rel, t, p, prob in zip(paths, labels, preds, probs):
        if t != p and shown < 10:
            conf = prob[p] * 100
            print(f"{rel}: target={CLASS_NAMES[t]} pred={CLASS_NAMES[p]} conf={conf:.2f}%")
            shown += 1


def test_random_segments(num_samples: int = 100, target_length: int = 360, batch_size: int = 32, seed=None):
    """Main: sample random rows, run inference and print metrics."""
    csv_file = "data_prep/dataset/metadata.csv"
    base_dir = "data_prep/dataset"
    model_path = "best_ecg_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(csv_file).exists() or not Path(base_dir).exists():
        print("Errore: dataset o metadata.csv non presenti. Rigenera il dataset prima di eseguire il test.")
        return

    X, labels, paths = sample_inputs(csv_file, base_dir, num_samples, target_length, seed=seed)
    if len(labels) == 0:
        print("Nessun segmento valido trovato tra le righe selezionate.")
        return

    try:
        model = load_model(model_path, device)
    except RuntimeError as exc:
        print(exc)
        return
    preds, probs = evaluate(model, X, batch_size=batch_size)
    print_summary(labels, preds, probs, paths)

def test_last_patient():
    """Test using all segments from the last patient in the metadata."""
    csv_file = "data_prep/dataset/metadata.csv"
    base_dir = "data_prep/dataset"
    model_path = "best_ecg_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(csv_file).exists() or not Path(base_dir).exists():
        print("Errore: dataset o metadata.csv non presenti. Rigenera il dataset prima di eseguire il test.")
        return

    df = pd.read_csv(csv_file)
    df["patient_id"] = df["file_path"].apply(lambda x: x.split("/")[0])
    last_patient = df["patient_id"].iloc[-1]
    patient_rows = df[df["patient_id"] == last_patient].reset_index(drop=True)

    inputs = []
    labels = []
    paths = []
    base = Path(base_dir)
    for _, row in patient_rows.iterrows():
        rel = row["file_path"]
        p = base / rel
        if not p.exists():
            continue
        sig = np.load(p)
        sig = pad_or_trim(sig, 360)
        inputs.append(sig)
        labels.append(int(row["label"]))
        paths.append(rel)

    if len(inputs) == 0:
        print("Nessun segmento valido trovato per l'ultimo paziente.")
        return

    X = np.stack(inputs)
    try:
        model = load_model(model_path, device)
    except RuntimeError as exc:
        print(exc)
        return
    preds, probs = evaluate(model, X, batch_size=32)
    print_summary(labels, preds, probs, paths)

if __name__ == "__main__":
    test_random_segments(100)
    test_last_patient()