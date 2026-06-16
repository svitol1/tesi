import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class ECGDataset(Dataset):

    def __init__(self, csv_file, base_dir):
        self.metadata = pd.read_csv(csv_file)
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Ottiene la riga dal CSV
        row = self.metadata.iloc[idx]

        # Carica il segmento dal file .npy
        signal_path = self.base_dir / row["file_path"]
        signal = np.load(signal_path)

        if signal.ndim == 1:
            signal = np.expand_dims(signal, axis=0)
        elif signal.ndim != 2:
            raise ValueError(f"Formato segnale non supportato: shape={signal.shape}")

        label = int(row["label"])

        # Ritorna i tensori pronti per la rete
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)