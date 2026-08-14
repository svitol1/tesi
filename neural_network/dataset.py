"""
================================================================================
dataset.py
Costruzione del grafo delle 12 derivazioni ECG e Dataset PyG.
================================================================================

Ogni segmento ECG ha shape (12, T) con T = 500 campioni (finestra di
1 secondo [picco_R-400 ms, picco_R+600 ms] a 500 Hz). Le 12 derivazioni sono
modellate come nodi di un grafo:
    - feature del nodo = segnale temporale grezzo della derivazione
    - archi = relazioni geometriche/vettoriali basate sulla posizione 3D
              delle derivazioni nel corpo umano.

Questo file definisce la topologia del grafo e il wrapping dei dati
in oggetti torch_geometric.data.Data.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

SEGMENT_LENGTH = 500

# Ordine fisso dei nodi.
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_IDX = {name: i for i, name in enumerate(LEAD_ORDER)}

# Coordinate vettoriali unitarie (x, y, z) approssimate per le 12 derivazioni
# x = Asse Trasversale (Destra -> Sinistra)
# y = Asse Longitudinale/Verticale (Testa -> Piedi)
# z = Asse Anteroposteriore (Schiena -> Petto)
LEAD_VECTORS_3D = {
    # Derivazioni degli arti (Piano Frontale z=0, basate sul sistema esassiale)
    "I":   np.array([ 1.000,  0.000,  0.000]),  # 0°
    "II":  np.array([ 0.500,  0.866,  0.000]),  # +60°
    "III": np.array([-0.500,  0.866,  0.000]),  # +120°
    "aVR": np.array([-0.866, -0.500,  0.000]),  # -150°
    "aVL": np.array([ 0.866, -0.500,  0.000]),  # -30°
    "aVF": np.array([ 0.000,  1.000,  0.000]),  # +90°

    # Derivazioni Precordiali (Piano Orizzontale/Spaziale 3D)
    "V1":  np.array([-0.342,  0.174,  0.923]),  # Parasternale destra
    "V2":  np.array([-0.174,  0.174,  0.969]),  # Parasternale sinistra
    "V3":  np.array([ 0.174,  0.087,  0.981]),  # Tra V2 e V4
    "V4":  np.array([ 0.500,  0.000,  0.866]),  # Emiclaveare sinistra
    "V5":  np.array([ 0.866, -0.087,  0.492]),  # Ascellare anteriore
    "V6":  np.array([ 0.985, -0.174,  0.000])   # Ascellare media
}


def build_ecg_graph_topology(mode: str = "vector_geometric", threshold: float = 0.0):
    """
    Costruisce l'edge_index e l'edge_weight per PyTorch Geometric.

    Parameters
    ----------
    mode : str
        - "vector_geometric": Basato sulla similitudine del coseno tra i vettori
          3D delle 12 derivazioni cardiache.
        - "fully_connected": Grafo completo uniformemente pesato.
    threshold : float
        Soglia minima di similitudine per creare un arco (default 0.0, mantiene
        solo le affinità positive/parallele, scartando i vettori ortogonali o opposti).

    Returns
    -------
    edge_index : torch.LongTensor, shape [2, num_edges]
    edge_weight : torch.FloatTensor, shape [num_edges]
    """
    edges = []
    weights = []

    num_leads = len(LEAD_ORDER)

    if mode == "vector_geometric":
        # Normalizzazione dei vettori unitari
        vectors = np.array([LEAD_VECTORS_3D[lead] for lead in LEAD_ORDER])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_normalized = vectors / norms

        # Calcolo della matrice di similitudine del coseno (12x12)
        # S_ij = cos(theta) = (v_i . v_j) / (||v_i|| * ||v_j||)
        sim_matrix = np.dot(vectors_normalized, vectors_normalized.T)

        for i in range(num_leads):
            for j in range(num_leads):
                # Mantieni connessioni con similitudine superiore alla soglia
                # (nota: i == j dà peso 1.0, ovvero l'autoloop)
                w = sim_matrix[i, j]
                if w > threshold:
                    edges.append((i, j))
                    weights.append(w)

    elif mode == "fully_connected":
        for i in range(num_leads):
            for j in range(num_leads):
                edges.append((i, j))
                weights.append(1.0)
    else:
        raise ValueError(f"modalità '{mode}' non riconosciuta.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)

    return edge_index, edge_weight


class ECGGraphDataset(Dataset):
    """
    Dataset PyG per segmenti ECG a 12 derivazioni con supporto ai pesi degli archi.
    """

    def __init__(self, segments: np.ndarray, labels: np.ndarray,
                 edge_index: torch.Tensor, edge_weight: torch.Tensor = None,
                 normalize: bool = True, expected_length: int | None = SEGMENT_LENGTH):
        super().__init__()
        assert segments.ndim == 3 and segments.shape[1] == 12, \
            f"segments deve avere shape (N, 12, T), trovato {segments.shape}"
        if expected_length is not None and segments.shape[2] != expected_length:
            raise ValueError(
                f"Lunghezza segmento inattesa: {segments.shape[2]}, atteso {expected_length}."
            )
        self.segments = segments.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.normalize = normalize

    def len(self):
        return len(self.labels)

    def get(self, idx):
        x = self.segments[idx]  # shape (12, T)

        if self.normalize:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-8
            x = (x - mean) / std

        x = torch.from_numpy(x)                                  # [12, T]
        y = torch.tensor([self.labels[idx]], dtype=torch.long)   # [1]

        # Creazione del dato PyG includendo edge_weight
        data = Data(x=x, edge_index=self.edge_index, edge_weight=self.edge_weight, y=y)
        return data