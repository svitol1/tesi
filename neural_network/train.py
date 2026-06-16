import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# Import miei moduli
from cnn_pytorch import ecg_classifier
from dataset_pytorch import ECGDataset
from FocalLoss import FocalLoss


def build_weighted_sampler(train_dataset, num_classes: int = 4) -> WeightedRandomSampler:
    """Build a sampler that oversamples rare classes in the training subset.

    The sampler is computed from the metadata labels only, so it does not
    need to load ECG signals during setup.
    """
    if not isinstance(train_dataset, Subset):
        raise TypeError("train_dataset must be a torch.utils.data.Subset")

    metadata = train_dataset.dataset.metadata
    labels = np.array([int(metadata.iloc[idx]["label"]) for idx in train_dataset.indices])
    counts = np.bincount(labels, minlength=num_classes).astype(float)

    if np.any(counts == 0):
        missing = [str(i) for i, count in enumerate(counts) if count == 0]
        raise ValueError(f"Training split is missing classes: {', '.join(missing)}")

    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]

    print("WeightedRandomSampler configurato:")
    for i, count in enumerate(counts):
        print(f"  Classe {i}: {int(count):6d} campioni → peso = {class_weights[i]:.6f}")

    weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# =============================================================================
# Main
# =============================================================================
def main():
    # ---------------------------------------------------------
    # Configurazione dell'unità (CPU o GPU)
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Unità utilizzata per l'allenamento: {device}\n")

    csv_file = "data_prep/dataset/metadata.csv"
    base_dir = "data_prep/dataset"

    # ---------------------------------------------------------
    # Divisione del dataset in training e validation (per paziente)
    # ---------------------------------------------------------
    train_dataset, val_dataset = divide_dataset(csv_file, base_dir)

    # ---------------------------------------------------------
    # Creazione dei DataLoader
    # ---------------------------------------------------------
    BATCH_SIZE = 32

    train_sampler = build_weighted_sampler(train_dataset, num_classes=4)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------------------------------------------------
    # Inizializzazione modello, loss e ottimizzatore
    # ---------------------------------------------------------
    model = ecg_classifier(num_classi=4).to(device)

    # ---------------- Focal Loss -----------------------------
    # Il bilanciamento delle classi viene gestito dal sampler.
    # La Focal Loss resta utile per dare meno peso agli esempi facili.
    criterion = FocalLoss(alpha=None, gamma=2.0)
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ---------------------------------------------------------
    # Controllo se il modello è stato salvato in precedenza e carico i pesi
    # ---------------------------------------------------------
    if input("\nVuoi caricare il modello salvato e continuare l'allenamento? (y/n): ").lower() == "y":
        try:
            checkpoint = torch.load("best_ecg_model.pth", map_location=device)

            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
                if "optimizer_state" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state"])
                        print("Ottimizzatore ripristinato dal checkpoint.")
                    except Exception:
                        print("Non è stato possibile ripristinare lo stato dell'ottimizzatore.")
                print("Checkpoint caricato da 'best_ecg_model.pth'. Continuo l'allenamento...\n")
            else:
                model.load_state_dict(checkpoint)
                print("State dict caricato da 'best_ecg_model.pth'. Continuo l'allenamento...\n")
        except FileNotFoundError:
            print("Nessun modello salvato trovato. Inizio un nuovo allenamento...\n")
        except Exception as e:
            print(f"Errore nel caricamento del modello: {e}. Inizio un nuovo allenamento...\n")

    # ---------------------------------------------------------
    # Training e validation
    # ---------------------------------------------------------
    NUM_EPOCHS   = 15
    best_val_loss = float("inf")

    print("Inizio allenamento")
    print("-" * 50)

    for epoch in range(NUM_EPOCHS):

        # --- FASE DI TRAINING ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = (train_correct / train_total) * 100

        # --- FASE DI VALIDAZIONE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)   # ← FocalLoss

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (val_correct / val_total) * 100

        print(
            f"Epoca [{epoch+1:02d}/{NUM_EPOCHS}] -> "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint = {
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch":           epoch + 1,
                "best_val_loss":   best_val_loss,
                "focal_gamma":     criterion.gamma,
            }
            torch.save(checkpoint, "best_ecg_model.pth")
            print("  → Modello salvato (checkpoint)")

    print("-" * 50)
    print("Allenamento terminato!")


# =============================================================================
# Divisione dataset per paziente
# =============================================================================
def divide_dataset(csv_file, base_dir):
    df = pd.read_csv(csv_file)
    df["patient_id"] = df["file_path"].apply(lambda x: x.split("/")[0])

    unique_patients = df["patient_id"].unique().tolist()
    print(f"Pazienti totali trovati nel dataset: {len(unique_patients)}")

    random.seed(42)
    random.shuffle(unique_patients)

    num_train_patients = int(0.8 * len(unique_patients))
    train_patients     = unique_patients[:num_train_patients]
    val_patients       = unique_patients[num_train_patients:]

    print(f"Pazienti scelti per il Training ({len(train_patients)}): {train_patients}")
    print(f"Pazienti scelti per la Validazione ({len(val_patients)}): {val_patients}\n")

    train_indices = df[df["patient_id"].isin(train_patients)].index.tolist()
    val_indices   = df[df["patient_id"].isin(val_patients)].index.tolist()

    full_dataset  = ECGDataset(csv_file=csv_file, base_dir=base_dir)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    print(f"Segmenti totali di Train: {len(train_dataset)}")
    print(f"Segmenti totali di Validation: {len(val_dataset)}\n")

    return train_dataset, val_dataset


if __name__ == "__main__":
    main()