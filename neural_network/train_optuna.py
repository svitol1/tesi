"""
================================================================================
Ottimizzazione degli iperparametri (Optuna) per la GCN pura definita in
model.py, sui dati costruiti tramite dataset.py.

Basato sulla struttura di train.py: stesso caricamento dati, stesso split
per paziente, stessa topologia del grafo. use_attention è fissato a False.
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

import optuna
from optuna.trial import TrialState
from torch_geometric.loader import DataLoader

from model import ECG_GCN
from dataset import ECGGraphDataset, build_signed_ecg_graph_topology
from train import load_misplacement_data  # riuso il loader già definito in train.py


# ----------------------------------------------------------------------------
# Configurazione globale della ricerca
# ----------------------------------------------------------------------------
SEGMENT_LEN = 500
N_TRIALS = 50
TIMEOUT_SECONDS = None          # None per disattivare il limite di tempo
SEARCH_EPOCHS = 10              # poche per velocizzare la ricerca
STUDY_NAME = "ecg_gcn_hparam_search"


def build_datasets():
    """Carica i dati e costruisce train/val dataset PyG, una sola volta."""
    segments, labels, patient_ids, label_to_idx = load_misplacement_data(
        dataset_root="data_prep/misplacement_dataset",
        expected_len=SEGMENT_LEN,
    )
    num_classes = len(label_to_idx)

    edge_index_pos, edge_weight_pos, edge_index_neg, edge_weight_neg = build_signed_ecg_graph_topology()

    dataset = ECGGraphDataset(
        segments=segments,
        labels=labels,
        edge_index_pos=edge_index_pos,
        edge_weight_pos=edge_weight_pos,
        edge_index_neg=edge_index_neg,
        edge_weight_neg=edge_weight_neg,
        normalize=True,
    )

    # Split per paziente (identico a train.py, seed fisso per coerenza tra trial)
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
    print(f"Classi ({num_classes}): {label_to_idx}")

    return train_dataset, val_dataset, num_classes


def make_objective(train_dataset, val_dataset, num_classes, device):
    """Crea la funzione objective, chiudendo sui dati già costruiti una volta."""

    def objective(trial: optuna.Trial) -> float:
        #seed fisso per trial: la variabilità viene solo dagli iperparametri
        torch.manual_seed(0)
        np.random.seed(0)

        # -------------------- spazio di ricerca --------------------
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128])
        num_gnn_layers = trial.suggest_int("num_gnn_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        use_mlp_classifier = trial.suggest_categorical("use_mlp_classifier", [False, True])
        # use_attention fissato a False: non fa parte della ricerca

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = ECG_GCN(
            node_feat_dim=64,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_gnn_layers=num_gnn_layers,
            use_attention=False,
            dropout=dropout,
            use_mlp_classifier=use_mlp_classifier
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(SEARCH_EPOCHS):
            # ---- train ----
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, data.y.view(-1))
                loss.backward()
                optimizer.step()

            # ---- validazione ----
            model.eval()
            correct, total = 0, 0
            val_loss_sum = 0.0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data)
                    y = data.y.view(-1)
                    loss = criterion(out, y)
                    val_loss_sum += loss.item() * y.size(0)
                    correct += (out.argmax(dim=1) == y).sum().item()
                    total += y.size(0)
            val_acc = correct / total
            val_loss = val_loss_sum / total

            # ---- pruning ----
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_acc

    return objective


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")

    train_dataset, val_dataset, num_classes = build_datasets()
    objective = make_objective(train_dataset, val_dataset, num_classes, device)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=None,
        load_if_exists=True,          # permette di riprendere una ricerca interrotta
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )

    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)

    # ------------------------------------------------------------------
    # Riepilogo risultati
    # ------------------------------------------------------------------
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n" + "=" * 60)
    print(f"Trial totali: {len(study.trials)}")
    print(f"  completati: {len(complete_trials)}")
    print(f"  interrotti (pruning): {len(pruned_trials)}")

    best = study.best_trial
    print(f"\nMiglior val_acc: {best.value:.4f}")
    print("Migliori iperparametri:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Salvataggio dei migliori iperparametri su file
    # ------------------------------------------------------------------
    out_path = Path("best_hparams.txt")
    with out_path.open("w") as f:
        f.write(f"best_val_acc: {best.value:.4f}\n")
        f.write("use_attention: False  (fissato, non incluso nella ricerca)\n")
        for k, v in best.params.items():
            f.write(f"{k}: {v}\n")
    print(f"\nParametri migliori salvati in {out_path}")

if __name__ == "__main__":
    main()
