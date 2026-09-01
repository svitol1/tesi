"""
================================================================================
Dataset PyTorch "standard" (senza grafo) per la CNN.
================================================================================

Controparte semplificata di dataset.py: qui non serve costruire nessuna
topologia. Ogni sample è semplicemente il segnale grezzo a 12 derivazioni
[12, T] con la sua etichetta, e le derivazioni vengono passate al modello
come canali (vedi model_cnn.py).

Essendo un Dataset "puro" PyTorch (non torch_geometric.data.Dataset), può
essere usato direttamente con torch.utils.data.DataLoader standard: niente
oggetti Data/Batch, niente __inc__, niente chiamate custom, perché ogni
sample ha già la stessa shape fissa [12, T] (il batching di default di
PyTorch fa già uno stack sulla prima dimensione).

La normalizzazione (z-score per derivazione) è identica a quella usata in
dataset.py, per garantire che la baseline CNN e il modello Signed-GCN vedano
esattamente lo stesso preprocessing dei dati e il confronto tra i due sia
equo.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

SEGMENT_LENGTH = 500

# Stesso ordine di derivazioni usato in dataset.py
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


class ECGSegmentDataset(Dataset):
    """
    Dataset PyTorch standard per segmenti ECG a 12 derivazioni, da usare
    con ECG_CNN (cnn_model.py).

    Parametri
    ---------
    segments : np.ndarray, shape (N, 12, T)
        Segnali ECG grezzi, stesso formato di ECGGraphDataset in dataset.py.
    labels : np.ndarray, shape (N,)
        Etichette intere, stesso encoding usato per il modello con grafo,
        in modo che i dataset siano intercambiabili e i risultati confrontabili.
    normalize : bool
        Se True, applica z-score per derivazione (stessa normalizzazione di
        ECGGraphDataset): ogni derivazione viene centrata e scalata usando
        la propria media/deviazione standard calcolate sulla singola finestra.
    expected_length : int | None
        Se specificato, controlla che T corrisponda (default 500). Passare
        None per disattivare il controllo (es. se si sperimenta con finestre
        di lunghezza diversa: sia la CNN in model_cnn.py che questo dataset
        non dipendono rigidamente da T).

    Ogni sample restituito è una tupla (x, y):
        x : torch.FloatTensor, shape [12, T]
        y : torch.LongTensor,  shape []  (scalare, indice di classe)
    Il DataLoader di default impacchetta poi automaticamente il batch in
    x: [batch_size, 12, T] e y: [batch_size], pronti per ECG_CNN.
    """

    def __init__(self, segments: np.ndarray, labels: np.ndarray,
                 normalize: bool = True,
                 expected_length: int | None = SEGMENT_LENGTH):

        assert segments.ndim == 3 and segments.shape[1] == 12, \
            f"segments deve avere shape (N, 12, T), trovato {segments.shape}"

        if expected_length is not None and segments.shape[2] != expected_length:
            raise ValueError(
                f"Lunghezza segmento inattesa: {segments.shape[2]}, atteso {expected_length}."
            )
        assert segments.shape[0] == labels.shape[0], \
            "segments e labels devono avere lo stesso numero di sample"

        self.segments = segments.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.segments[idx]  # shape (12, T)

        if self.normalize:
            # Z-score per derivazione: stessa logica di dataset.py, calcolata
            # riga per riga (ogni derivazione normalizzata sulla propria
            # media/deviazione standard, non su quella globale del batch).
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-8
            x = (x - mean) / std

        x = torch.from_numpy(x)                     # [12, T]
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # scalare

        return x, y