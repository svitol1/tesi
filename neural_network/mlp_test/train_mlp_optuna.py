"""
================================================================================
Ottimizzazione degli iperparametri (Optuna) per l'MLP pura definita in
mlp_model.py, sui dati costruiti tramite dataset_builder.py.

Riusa da train_mlp.py:
    - load_split      : caricamento di una singola cartella (train/ o val/)
    - run_epoch        : un'epoca di train o eval, identica a quella
                          usata nel training "vero", cosi' la ricerca
                          ottimizza esattamente la stessa loss/metrica.

Lo spazio di ricerca è analogo a quello di train_cnn_optuna.py, adattato
ai soli iperparametri rilevanti per un MLP (hidden_dims al posto di
hidden_channels, use_batchnorm al posto di use_attention). seq_len non
fa parte della ricerca: è fissato da SEGMENT_LEN, coerentemente con il
fatto che l'MLP (a differenza della CNN) richiede una dimensione di
input fissa nota a priori (vedi mlp_model.py).
================================================================================
"""

from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.trial import TrialState
from torch_geometric.loader import DataLoader

from mlp_dataset import ECGSegmentDataset
from mlp_model import ECG_MLP
from train_mlp import load_split, run_epoch


# ----------------------------------------------------------------------------
# Configurazione globale della ricerca
# ----------------------------------------------------------------------------
SEGMENT_LEN = 500
DATASET_ROOT = "../data_prep/"
N_TRIALS = 50
TIMEOUT_SECONDS = None          # None per disattivare il limite di tempo
SEARCH_EPOCHS = 10              # poche per velocizzare la ricerca
STUDY_NAME = "ecg_mlp_hparam_search"


def build_datasets():
    """Carica train/ e val/ (gia' separati da dataset_builder.py)
    e costruisce i dataset corrispondenti, una sola volta."""
    train_segments, train_labels, label_to_idx = load_split(
        split_root=f"{DATASET_ROOT}/train",
        expected_len=SEGMENT_LEN,
    )
    val_segments, val_labels, _ = load_split(
        split_root=f"{DATASET_ROOT}/val",
        expected_len=SEGMENT_LEN,
        label_to_idx=label_to_idx,
    )
    num_classes = len(label_to_idx)

    train_dataset = ECGSegmentDataset(
        segments=train_segments,
        labels=train_labels,
        normalize=True,
        expected_length=SEGMENT_LEN
    )
    val_dataset = ECGSegmentDataset(
        segments=val_segments,
        labels=val_labels,
        normalize=True,
        expected_length=SEGMENT_LEN
    )

    print(f"Train segments: {len(train_labels)} | Val segments: {len(val_labels)}")
    print(f"Classi ({num_classes}): {label_to_idx}")

    return train_dataset, val_dataset, num_classes


def make_objective(train_dataset, val_dataset, num_classes, device):
    """Crea la funzione obiettivo, definendo lo spazio di ricerca."""

    def objective(trial: optuna.Trial) -> float:
        # seed fisso per trial: la variabilita' viene solo dagli iperparametri
        torch.manual_seed(0)
        np.random.seed(0)

        # -------------------- spazio di ricerca --------------------
        hidden_dims = trial.suggest_categorical(
            "hidden_dims",
            [(256, 128, 64), (512, 256, 128), (512, 256, 128, 64), (1024, 512, 128)]
        )
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        use_batchnorm = trial.suggest_categorical("use_batchnorm", [False, True])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = ECG_MLP(
            in_channels=12,
            seq_len=SEGMENT_LEN,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        val_acc = 0.0
        for epoch in range(SEARCH_EPOCHS):
            _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            _, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

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
    out_path = Path("best_hparams_mlp.txt")
    with out_path.open("w") as f:
        f.write(f"best_val_acc: {best.value:.4f}\n")
        f.write(f"seq_len: {SEGMENT_LEN}  (fissato, non incluso nella ricerca)\n")
        for k, v in best.params.items():
            f.write(f"{k}: {v}\n")
    print(f"\nParametri migliori salvati in {out_path}")


if __name__ == "__main__":
    main()
