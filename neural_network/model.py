"""
================================================================================
Architettura della rete: GCN pura per la rilevazione del malposizionamento
degli elettrodi ECG.
================================================================================

Segmenti di input: (12, T) con T = 500 campioni (finestra di 1 secondo a
500 Hz). L'architettura qui sotto NON dipende rigidamente da T: il
LeadCNNEncoder termina con un AdaptiveAvgPool1d, quindi accetta in input
segnali di qualunque lunghezza senza bisogno di ricalcolare a mano le
dimensioni dei layer. Questo torna utile se in futuro la finestra dovesse
cambiare.

Quello che invece è stato scelto pensando esplicitamente a T=500 a 500 Hz
sono i kernel size della CNN (vedi commenti in LeadCNNEncoder): sono tarati
sulla durata reale in millisecondi delle onde ECG (P, QRS, T) a questa
frequenza di campionamento, non sulla lunghezza totale della finestra.

Pipeline concettuale:
    segnale grezzo per derivazione (12 nodi, T campioni ciascuno)
        -> LeadCNNEncoder (CNN 1D condivisa, estrae un vettore di feature
           per ciascuna derivazione)
        -> N x GCNConv / GATv2Conv (message passing tra derivazioni,
           sfruttando la topologia del grafo definita in dataset.py)
        -> global mean+max pooling (embedding dell'intero ECG a 12 derivazioni)
        -> MLP di classificazione
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_max_pool


class LeadCNNEncoder(nn.Module):
    """
    CNN 1D condivisa applicata a ciascuna derivazione singolarmente.

    Input:  [num_nodes_totali_nel_batch, 1, T]   (T = 500 nel nostro caso)
    Output: [num_nodes_totali_nel_batch, out_channels]

    Essendo condivisa tra tutte le derivazioni, i pesi imparano pattern
    morfologici generici (onda P, complesso QRS, onda T) indipendentemente
    dalla derivazione specifica; è la GCN a modellare poi come queste
    feature si relazionano tra le derivazioni.

    Dimensionamento dei kernel (a 500 Hz, 1 campione = 2 ms):
        - blocco 1, kernel=7  -> finestra ricettiva locale di 14 ms:
          abbastanza fine da seguire i bordi ripidi del complesso QRS
          (che dura tipicamente 80-120 ms, quindi 40-60 campioni)
        - blocco 2, kernel=5  -> 10 ms sulla rappresentazione già
          sotto-campionata (dopo il primo MaxPool)
        - blocco 3, kernel=3  -> raffinamento finale prima del pooling globale
    Tre blocchi con MaxPool(2) portano T=500 -> 250 -> 125 prima
    dell'AdaptiveAvgPool1d(1) finale, quindi anche all'ultimo strato
    convoluzionale la rete vede ancora un contesto temporale sufficiente
    (125 campioni = 250 ms), abbastanza per contenere l'intero complesso QRS.
    """

    def __init__(self, out_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                       # 500 -> 250

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                       # 250 -> 125

            nn.Conv1d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),               # 125 -> 1, qualunque T
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [N, 1, T]
        out = self.net(x)              # [N, out_channels, 1]
        out = out.squeeze(-1)          # [N, out_channels]
        return self.dropout(out)


class ECG_GCN(nn.Module):
    """
    GCN pura per la classificazione del malposizionamento elettrodi.

    Parametri
    ---------
    node_feat_dim : int
        Dimensione del vettore di feature prodotto da LeadCNNEncoder per
        ciascun nodo (derivazione). 64 è un buon compromesso per un
        segmento di 500 campioni: abbastanza capiente da rappresentare la
        morfologia del battito senza sovradimensionare la rete rispetto
        alla quantità di informazione realmente presente in 1 secondo di
        segnale.
    hidden_dim : int
        Dimensione nascosta degli strati di message passing.
    num_classes : int
        Numero di classi da predire (es. normale + tipi di inversione).
    num_gnn_layers : int
        Numero di strati di message passing (GCNConv o GATv2Conv) impilati.
        Con soli 12 nodi nel grafo, 2 strati sono già sufficienti a
        propagare informazione tra qualunque coppia di derivazioni connesse
        anche indirettamente; aumentarli oltre 3 rischia solo di
        introdurre over-smoothing (i nodi finiscono per assomigliarsi troppo).
    use_attention : bool
        Se True usa GATv2Conv al posto di GCNConv: impara pesi di attenzione
        sugli archi, invece di definire a priori i pesi degli archi in base alla
        affinità anatomica.
    dropout : float
        Dropout applicato nel classificatore finale.

    Forward
    -------
    Riceve un batch PyG (oggetto Data/Batch) con:
        data.x          -> [num_nodes_totali_batch, T]  (segnale grezzo per nodo)
        data.edge_index -> [2, num_edges_totali_batch]
        data.batch       -> [num_nodes_totali_batch]     (indice di grafo per nodo)
    Ritorna: logits di shape [batch_size, num_classes]
    """

    def __init__(self, node_feat_dim: int = 64, hidden_dim: int = 128,
                 num_classes: int = 13, num_gnn_layers: int = 2,
                 use_attention: bool = False, dropout: float = 0.3):
        super().__init__()
        self.use_attention = use_attention
        self.cnn_encoder = LeadCNNEncoder(out_channels=node_feat_dim)

        conv_layer = (lambda in_c, out_c: GATv2Conv(in_c, out_c, heads=4, concat=False)) \
            if use_attention else GCNConv

        self.convs = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(num_gnn_layers):
            self.convs.append(conv_layer(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),   # *2 per mean+max pooling
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Estrae i pesi degli archi dal batch (se presenti), altrimenti None
        edge_weight = getattr(data, 'edge_weight', None)

        # x: [num_nodes_totali_batch, T]  ->  aggiungo canale per la Conv1d
        x = x.unsqueeze(1)                    # [num_nodes, 1, T]
        x = self.cnn_encoder(x)               # [num_nodes, node_feat_dim]

        for conv in self.convs:
            if self.use_attention:
                # GATv2 impara dinamicamente l'attenzione tra nodi
                x = F.relu(conv(x, edge_index))
            else:
                # GCNConv applica la convoluzione pesata con edge_weight fisso
                x = F.relu(conv(x, edge_index, edge_weight=edge_weight))

        x_mean = global_mean_pool(x, batch)   # [batch_size, hidden_dim]
        x_max = global_max_pool(x, batch)     # [batch_size, hidden_dim]
        graph_embedding = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(graph_embedding)