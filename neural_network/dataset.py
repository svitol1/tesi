"""
================================================================================
Costruzione del grafo delle 12 derivazioni ECG e Dataset PyG.
================================================================================

Ogni segmento ECG ha shape (12, T) con T = 900 (picco_R-300 : picco_R+600).
Le 12 derivazioni sono modellate come nodi di un grafo:
    - feature del nodo = segnale temporale grezzo della derivazione
      (verrà incorporato in un vettore dal LeadCNNEncoder dentro il modello,
      vedi model.py)
    - archi = relazioni anatomiche/elettriche tra derivazioni

Questo file non contiene nessuna parte di rete neurale: solo topologia del
grafo e wrapping dei dati in oggetti.
"""

import numpy as np
import torch

from torch_geometric.data import Data, Dataset


# Usare SEMPRE questo ordine quando si costruiscono i segmenti (N, 12, T),
# altrimenti gli archi non corrisponderanno più alle derivazioni giuste.
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_IDX = {name: i for i, name in enumerate(LEAD_ORDER)}


def build_ecg_edge_index(mode: str = "anatomical") -> torch.Tensor:
    """
    Costruisce edge_index (formato PyG: shape [2, num_edges], grafo NON direzionato
    quindi ogni arco è inserito in entrambe le direzioni).

    mode = "anatomical":
        - Le derivazioni degli arti (I, II, III, aVR, aVL, aVF) sono
          matematicamente dipendenti tra loro (leggi di Einthoven e Goldberger:
          II = I + III, aVR = -(I+II)/2, ecc.) -> le connettiamo a coppie
          complete (clique da 6 nodi).
        - Le derivazioni precordiali (V1..V6) rappresentano una progressione
          spaziale sul torace, quindi le connettiamo in "catena" (V1-V2-V3-V4-V5-V6),
          topologia naturale per rilevare inversioni tra derivazioni adiacenti
          (es. V1<->V2, V3<->V4).
        - Aggiungiamo alcuni archi "ponte" tra i due gruppi, basati sulla
          prossimità anatomica/elettrica reale (derivazioni laterali,
          inferiori, settali).

    mode = "fully_connected":
        - Grafo completo a 12 nodi. Più generico, lascia più libertà alla
          GNN, soprattutto se nel modello si usa GATv2Conv, che impara pesi di
          attenzione sugli archi.

    Ritorna: LongTensor di shape [2, num_edges]
    """
    edges = []

    def add_edge(a, b):
        i, j = LEAD_IDX[a], LEAD_IDX[b]
        edges.append((i, j))
        edges.append((j, i))

    if mode == "anatomical":
        limb_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
        precordial_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]

        # clique completa tra derivazioni degli arti
        for i in range(len(limb_leads)):
            for j in range(i + 1, len(limb_leads)):
                add_edge(limb_leads[i], limb_leads[j])

        # catena sequenziale tra derivazioni precordiali
        for i in range(len(precordial_leads) - 1):
            add_edge(precordial_leads[i], precordial_leads[i + 1])

        # archi ponte anatomici (relazioni spaziali reali sul torace)
        # DA CONTROLLARE
        bridge_edges = [
            ("aVR", "V1"),
            ("aVF", "V2"),
            ("aVF", "V3"),
            ("aVL", "V5"),
            ("aVL", "V6"),
            ("I", "V6"),
        ]
        for a, b in bridge_edges:
            add_edge(a, b)

    elif mode == "fully_connected":
        n = len(LEAD_ORDER)
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                edges.append((j, i))
    else:
        raise ValueError(f"mode '{mode}' non riconosciuto")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


class ECGGraphDataset(Dataset):
    """
    Dataset PyG per segmenti ECG a 12 derivazioni.

    Parametri
    ---------
    segments : np.ndarray, shape (N, 12, T)
        I segmenti [picco_R-300, picco_R+600] -> T = 900.
        L'ordine delle 12 derivazioni sull'asse 1 DEVE corrispondere a LEAD_ORDER.
    labels : np.ndarray, shape (N,)
        Etichetta di classe per ciascun segmento (0 = normale, 1..K = tipi di
        inversione: V1_V2, RA_LA, ecc.). Poiché tutti i segmenti di una stessa
        registrazione condividono la stessa etichetta, si fa lo split
        train/val/test PER REGISTRAZIONE e non per singolo segmento, altrimenti
        rischi data leakage (la rete apprende pattern specifici del soggetto e
        non generalizza).
    edge_index : torch.Tensor, shape [2, num_edges]
        Topologia del grafo, es. build_ecg_edge_index("anatomical").
    normalize : bool
        Se True, applica uno z-score per singola derivazione (per segmento).
        Lo z-score effettua una normalizzazione in modo da avere media 0 e deviazione
        standard 1, e viene calcolato separatamente per ciascun segmento e per ciascuna derivazione.
        Questo è utile perché le ampiezze dei segnali ECG possono variare molto tra
        soggetti diversi, e la rete neurale può concentrarsi sulle forme d'onda piuttosto
        che sulle ampiezze assolute.
    """

    def __init__(self, segments: np.ndarray, labels: np.ndarray,
                 edge_index: torch.Tensor, normalize: bool = True):
        super().__init__()
        assert segments.ndim == 3 and segments.shape[1] == 12, \
            "segments deve avere shape (N, 12, T)"
        self.segments = segments.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.edge_index = edge_index
        self.normalize = normalize

    def len(self):
        return len(self.labels)

    # Returns a single graph data object for the given index
    def get(self, idx):
        x = self.segments[idx]  # shape (12, T)

        if self.normalize:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-8
            x = (x - mean) / std

        x = torch.from_numpy(x)                       # [12, T]
        y = torch.tensor([self.labels[idx]], dtype=torch.long)  # [1]

        data = Data(x=x, edge_index=self.edge_index, y=y)
        return data
