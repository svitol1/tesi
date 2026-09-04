"""
================================================================================
dataset.py
Costruzione del grafo (firmato) delle 12 derivazioni ECG e Dataset PyG.
================================================================================

Ogni segmento ECG ha shape (12, T) con T = 500 campioni (finestra di
1 secondo [picco_R-400 ms, picco_R+600 ms] a 500 Hz). Le 12 derivazioni sono
modellate come nodi di un grafo:
    - feature del nodo = segnale temporale grezzo della derivazione
    - archi = relazioni geometriche/vettoriali basate sulla posizione 3D
              delle derivazioni nel corpo umano.

--------------------------------------------------------------------------------
Perchè un GRAFO SIGNED invece di uno unico con soglia
--------------------------------------------------------------------------------
La similarità del coseno tra i vettori di vista 3D delle derivazioni può
essere negativa (es. I e aVR sono quasi antiparallele ~ -0.87). Questa
anti-correlazione NON è proprio il tipo di relazione che si altera
in modo caratteristico quando due elettrodi vengono scambiati (es. RA-LA),
quindi è informazione rilevante per il task di rilevazione malposizionamento
e non va scartata.

Allo stesso tempo, GCNConv (Kipf & Welling) normalizza gli archi per il grado
dei nodi (D^-1/2 A D^-1/2), normalizzazione che presuppone pesi >= 0: pesi
negativi romperebbero questa formula (radice di un grado potenzialmente
negativo).

Soluzione adottata (ispirata a Derr et al., "Signed Graph Convolutional
Networks", ICDM 2018): invece di un solo grafo, costruiamo DUE sotto-grafi
sulle stesse 12 derivazioni:
    - grafo POSITIVO: coppie di derivazioni concordi (sim > 0), peso = sim
    - grafo NEGATIVO: coppie di derivazioni discordi (sim < 0), peso = |sim|
entrambi con pesi >= 0 per costruzione, quindi compatibili con GCNConv. Sarà
poi il modello (model.py) a combinare i due contributi, negando le feature
in ingresso nel ramo negativo per rappresentare esplicitamente l'anti-correlazione.

Questo file definisce la topologia del grafo e il wrapping dei
dati in oggetti torch_geometric.data.Data.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

SEGMENT_LENGTH = 500

# Tolleranza numerica sotto la quale una similarità è considerata "zero"
# (derivazioni considerate geometricamente ortogonali -> nessuna relazione
# lineare attesa -> nessun arco, né nel grafo positivo né in quello negativo).
# Nota: essendo la rete relativamente piccola, per evitare overfitting,
# è stato volutamente scelto un valore più alto in modo da creare un grafo
# più sparso (meno archi) e quindi più regolarizzato.
EPS = 0.2

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


def _compute_cosine_similarity_matrix() -> np.ndarray:
    """
    Calcola la matrice 12x12 di similarità del coseno tra i vettori di vista
    delle derivazioni. Valore in [-1, 1]:
        +1  -> stessa direzione (derivazioni "concordi")
         0  -> ortogonali (nessuna relazione lineare attesa)
        -1  -> direzioni opposte (derivazioni "discordi")
    """
    vectors = np.array([LEAD_VECTORS_3D[lead] for lead in LEAD_ORDER])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_normalized = vectors / norms
    return np.dot(vectors_normalized, vectors_normalized.T)


def build_signed_ecg_graph_topology():
    """
    Costruisce la topologia SIGNED del grafo delle 12 derivazioni: due
    sotto-grafi distinti (positivo/negativo), entrambi con pesi >= 0,
    pronti per essere passati a due GCNConv separate (vedi SignedGCNBlock
    in model.py).

    Regole di assegnazione per ciascuna coppia (i, j), incluso i == j:
        - sim(i, j) > EPS   -> arco nel grafo POSITIVO, peso = sim(i, j)
                                (include automaticamente i self-loop, dato
                                 che sim(i, i) = 1.0 per costruzione)
        - sim(i, j) < -EPS  -> arco nel grafo NEGATIVO, peso = |sim(i, j)|
        - altrimenti (|sim| <= EPS, derivazioni ortogonali) -> nessun arco

    Returns
    -------
    edge_index_pos : torch.LongTensor, shape [2, E_pos]
    edge_weight_pos : torch.FloatTensor, shape [E_pos]
    edge_index_neg : torch.LongTensor, shape [2, E_neg]
    edge_weight_neg : torch.FloatTensor, shape [E_neg]
    """
    sim_matrix = _compute_cosine_similarity_matrix()
    num_leads = len(LEAD_ORDER)

    edges_pos, weights_pos = [], []
    edges_neg, weights_neg = [], []

    for i in range(num_leads):
        for j in range(num_leads):
            w = sim_matrix[i, j]
            if w > EPS:
                edges_pos.append((i, j))
                weights_pos.append(w)
            elif w < -EPS:
                edges_neg.append((i, j))
                weights_neg.append(-w)  # salviamo il valore assoluto
            # |w| <= EPS -> derivazioni ortogonali, arco scartato in entrambi

    edge_index_pos = torch.tensor(edges_pos, dtype=torch.long).t().contiguous()
    edge_weight_pos = torch.tensor(weights_pos, dtype=torch.float)
    edge_index_neg = torch.tensor(edges_neg, dtype=torch.long).t().contiguous()
    edge_weight_neg = torch.tensor(weights_neg, dtype=torch.float)

    return edge_index_pos, edge_weight_pos, edge_index_neg, edge_weight_neg

class SignedECGData(Data):
    """
    Sottoclasse di Data che rende esplicito a PyG come incrementare gli
    indici dei nodi (edge_index_pos, edge_index_neg) quando più grafi
    (uno per segmento ECG) vengono impacchettati in un unico batch
    dal DataLoader.

    Nota: PyG di default incrementerebbe già correttamente qualunque
    attributo il cui nome contiene la sottostringa "index" (quindi
    funzionerebbe anche senza questa sottoclasse), ma la rendiamo esplicita
    per chiarezza e per non dipendere da un comportamento implicito legato
    al naming degli attributi.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("edge_index_pos", "edge_index_neg"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


class ECGGraphDataset(Dataset):
    """
    Dataset PyG per segmenti ECG a 12 derivazioni con grafo SIGNED
    (sotto-grafo positivo + sotto-grafo negativo), da usare con SignedGCNBlock.
    """

    def __init__(self, segments: np.ndarray, labels: np.ndarray,
                 edge_index_pos: torch.Tensor, edge_weight_pos: torch.Tensor,
                 edge_index_neg: torch.Tensor, edge_weight_neg: torch.Tensor,
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

        # Le due topologie (condivise da tutti i sample, dato che la
        # geometria delle 12 derivazioni non cambia da un ECG all'altro)
        self.edge_index_pos = edge_index_pos
        self.edge_weight_pos = edge_weight_pos
        self.edge_index_neg = edge_index_neg
        self.edge_weight_neg = edge_weight_neg

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

        # Creazione del dato PyG con i due sotto-grafi (stessa topologia
        # per ogni sample, ripetuta qui perché ogni Data è indipendente:
        # il costo è trascurabile con soli 12 nodi).
        data = SignedECGData(
            x=x,
            edge_index_pos=self.edge_index_pos,
            edge_weight_pos=self.edge_weight_pos,
            edge_index_neg=self.edge_index_neg,
            edge_weight_neg=self.edge_weight_neg,
            y=y,
            num_nodes=x.size(0),
        )
        return data