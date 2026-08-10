"""
================================================================================
Training loop per la GCN pura definita in model.py, sui dati costruiti
tramite dataset.py.
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

from model import ECG_GCN
from dataset import ECGGraphDataset, build_ecg_edge_index


def main():
    torch.manual_seed(0)

    # --------------------------------------------------------------------
    # 1) Dati
    # --------------------------------------------------------------------
    # ---- dati fittizi: sostituisci con i tuoi array reali ----
    N = 256                 # numero di segmenti
    T = 900                 # 300 pre + 600 post picco R
    NUM_CLASSES = 13        # es: normale + 12 tipi di inversione (esempio)

    segments = np.random.randn(N, 12, T).astype(np.float32)
    labels = np.random.randint(0, NUM_CLASSES, size=N)

    edge_index = build_ecg_edge_index(mode="anatomical")

    dataset = ECGGraphDataset(segments, labels, edge_index, normalize=True)

    n_train = int(0.8 * N)
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # --------------------------------------------------------------------
    # 2) Modello, ottimizzatore, loss
    # --------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECG_GCN(
        node_feat_dim=64,
        hidden_dim=128,
        num_classes=NUM_CLASSES,
        num_gnn_layers=2,
        use_attention=False,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

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

    EPOCHS = 5
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
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
