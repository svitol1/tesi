"""
================================================================================
Dataset PyTorch "standard" (senza grafo) per l'MLP.
================================================================================

Controparte di cnn_dataset.py per il modello ECG_MLP (mlp_model.py). Il
preprocessing è IDENTICO a quello usato per la CNN (e per la Signed-GCN in
dataset.py): ogni sample è il segnale grezzo a 12 derivazioni [12, T], con
z-score per derivazione. Questo è intenzionale: vogliamo che le tre
baseline (GCN, CNN, MLP) vedano esattamente lo stesso preprocessing dei
dati, così che le differenze di performance misurate in tesi siano
attribuibili all'architettura e non a differenze nel preprocessing.

La differenza rispetto a cnn_dataset.py non è nel preprocessing ma nel
vincolo sulla lunghezza del segmento: qui `expected_length` non può essere
disattivato (passando None) senza rompere il modello a valle. In
cnn_dataset.py il controllo su expected_length è opzionale perché la CNN,
grazie all'AdaptiveAvgPool1d/AdaptiveMaxPool1d finale, tollera lunghezze
diverse in training e inferenza. ECG_MLP invece fissa il numero di
input features del primo layer Linear a in_channels * seq_len nel
costruttore (vedi mlp_model.py): se qui si caricassero segmenti di
lunghezza diversa da quella usata per istanziare il modello, il forward
pass fallirebbe con una shape mismatch. Per questo expected_length ha qui
un default fisso (SEGMENT_LENGTH) e non è pensato per essere disattivato
in questo contesto, se non sperimentando ricreando anche il modello con lo
stesso seq_len.

Essendo un Dataset "puro" PyTorch (non torch_geometric.data.Dataset), può
essere usato direttamente con torch.utils.data.DataLoader standard: niente
oggetti Data/Batch, niente __inc__, niente chiamate custom, perché ogni
sample ha già la stessa shape fissa [12, T] (il batching di default di
PyTorch fa già uno stack sulla prima dimensione).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

SEGMENT_LENGTH = 500

# Stesso ordine di derivazioni usato in dataset.py e cnn_dataset.py
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


class ECGSegmentDataset(Dataset):
    """
    Dataset PyTorch standard per segmenti ECG a 12 derivazioni, da usare
    con ECG_MLP (mlp_model.py).

    Parametri
    ---------
    segments : np.ndarray, shape (N, 12, T)
        Segnali ECG grezzi, stesso formato di ECGGraphDataset in dataset.py
        e di ECGSegmentDataset in cnn_dataset.py.
    labels : np.ndarray, shape (N,)
        Etichette intere, stesso encoding usato per gli altri due modelli,
        in modo che i dataset siano intercambiabili e i risultati
        confrontabili.
    normalize : bool
        Se True, applica z-score per derivazione (stessa normalizzazione
        di ECGGraphDataset e della versione CNN): ogni derivazione viene
        centrata e scalata usando la propria media/deviazione standard
        calcolate sulla singola finestra.
    expected_length : int
        Lunghezza attesa T dei segmenti (default 500). A differenza della
        versione CNN, qui il controllo NON è opzionale nella pratica:
        ECG_MLP viene istanziato con un seq_len fisso, quindi tutti i
        segmenti devono avere esattamente questa lunghezza per essere
        compatibili con il modello a valle.

    Ogni sample restituito è una tupla (x, y):
        x : torch.FloatTensor, shape [12, T]
        y : torch.LongTensor,  shape []  (scalare, indice di classe)
    Il DataLoader di default impacchetta poi automaticamente il batch in
    x: [batch_size, 12, T] e y: [batch_size]; il flatten a [batch_size,
    12*T] avviene dentro ECG_MLP.forward, non qui, per mantenere il
    dataset agnostico rispetto al modello che lo consumerà.
    """

    def __init__(self, segments: np.ndarray, labels: np.ndarray,
                 normalize: bool = True,
                 expected_length: int = SEGMENT_LENGTH):

        assert segments.ndim == 3 and segments.shape[1] == 12, \
            f"segments deve avere shape (N, 12, T), trovato {segments.shape}"

        if segments.shape[2] != expected_length:
            raise ValueError(
                f"Lunghezza segmento inattesa: {segments.shape[2]}, atteso "
                f"{expected_length}. Per l'MLP la lunghezza deve combaciare "
                "esattamente con il seq_len usato per istanziare ECG_MLP."
            )
        assert segments.shape[0] == labels.shape[0], \
            "segments e labels devono avere lo stesso numero di sample"

        self.segments = segments.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.normalize = normalize
        self.expected_length = expected_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.segments[idx]  # shape (12, T)

        if self.normalize:
            # Z-score per derivazione: stessa logica di dataset.py e
            # cnn_dataset.py, calcolata riga per riga (ogni derivazione
            # normalizzata sulla propria media/deviazione standard, non su
            # quella globale del batch).
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-8
            x = (x - mean) / std

        x = torch.from_numpy(x)                     # [12, T]
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # scalare

        return x, y
