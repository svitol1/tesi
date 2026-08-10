"""
================================================================================
Architettura della rete: GCN pura per la rilevazione del malposizionamento
degli elettrodi ECG.
================================================================================

Pipeline concettuale:
    segnale grezzo per derivazione (12 nodi, T campioni ciascuno)
        -> LeadCNNEncoder (CNN 1D condivisa, estrae un vettore di feature
           per ciascuna derivazione)
        -> N x GCNConv (message passing tra derivazioni,
           sfruttando la topologia del grafo definita in dataset.py)
        -> global mean+max pooling (embedding dell'intero ECG a 12 derivazioni)
        -> MLP di classificazione

Questo file contiene SOLO la definizione dei moduli nn.Module. La costruzione
del grafo e del Dataset è in dataset.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_max_pool


class LeadCNNEncoder(nn.Module):
    """
    CNN 1D condivisa applicata a ciascuna derivazione singolarmente.

    Input:  [num_nodes_totali_nel_batch, 1, T]
    Output: [num_nodes_totali_nel_batch, out_channels]

    Essendo condivisa tra tutte le derivazioni, i pesi imparano pattern
    morfologici generici del battito (onda P, complesso QRS, onda T)
    indipendentemente dalla derivazione specifica; è la GCN a modellare poi
    come queste feature si relazionano tra le derivazioni.
    """

    def __init__(self, out_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # -> [*, out_channels, 1]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [N, 1, T]
        out = self.net(x)              # [N, out_channels, 1]
        out = out.squeeze(-1)          # [N, out_channels]
        return self.dropout(out)


class ECG_GCN(nn.Module):
    """
    GCN pura per la classificazione del malposizionamento degli elettrodi.

    Parametri
    ---------
    node_feat_dim : int
        Dimensione del vettore di feature prodotto da LeadCNNEncoder per
        ciascun nodo (derivazione).
    hidden_dim : int
        Dimensione nascosta degli strati di message passing.
    num_classes : int
        Numero di classi da predire (es. normale + tipi di inversione).
    num_gnn_layers : int
        Numero di strati di message passing (GCNConv o GATv2Conv) impilati.
    use_attention : bool
        Se True usa GATv2Conv al posto di GCNConv: impara pesi di attenzione
        sugli archi (quanto una derivazione vicina è rilevante per l'altra),
        invece di pesare gli archi in modo fisso in base al grado dei nodi
        come fa la GCN classica.
    dropout : float
        Dropout applicato nel classificatore finale.

    Forward
    -------
    Riceve un batch PyG (oggetto Data/Batch) con:
        data.x          -> [num_nodes_totali_batch, T]  (segnale grezzo per nodo)
        data.edge_index -> [2, num_edges_totali_batch]
        data.batch      -> [num_nodes_totali_batch]     (indice di grafo per nodo)
    Ritorna: logits di shape [batch_size, num_classes]
    """

    def __init__(self, node_feat_dim: int = 64, hidden_dim: int = 128,
                 num_classes: int = 13, num_gnn_layers: int = 2,
                 use_attention: bool = False, dropout: float = 0.3):
        super().__init__()
        # definisco il CNN encoder condiviso per le derivazioni
        self.cnn_encoder = LeadCNNEncoder(out_channels=node_feat_dim)
        # scelgo il tipo di strato di message passing: GCNConv o GATv2Conv
        conv_layer = (lambda in_c, out_c: GATv2Conv(in_c, out_c, heads=4, concat=False)) \
            if use_attention else GCNConv

        # costruisco una lista di strati di message passing
        self.convs = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(num_gnn_layers):
            self.convs.append(conv_layer(in_dim, hidden_dim))
            in_dim = hidden_dim
        # costruisco il classificatore finale (MLP)s
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),   # *2 per mean+max pooling
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x: [num_nodes_totali_batch, T]  ->  aggiungo canale per la Conv1d
        x = x.unsqueeze(1)                    # [num_nodes, 1, T]
        x = self.cnn_encoder(x)               # [num_nodes, node_feat_dim]

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        x_mean = global_mean_pool(x, batch)   # [batch_size, hidden_dim]
        x_max = global_max_pool(x, batch)     # [batch_size, hidden_dim]
        # concateno mean e max pooling per ottenere l'embedding finale del grafo
        graph_embedding = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(graph_embedding)
