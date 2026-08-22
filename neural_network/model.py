"""
================================================================================
Architettura della rete: Signed-GCN per la rilevazione del malposizionamento
degli elettrodi ECG.
================================================================================

Segmenti di input: (12, T) con T = 500 campioni (finestra di 1 secondo a
500 Hz). L'architettura qui sotto NON dipende rigidamente da T: il
LeadCNNEncoder termina con un AdaptiveAvgPool1d, quindi accetta in input
segnali di qualunque lunghezza senza bisogno di ricalcolare a mano le
dimensioni dei layer.

Quello che invece è stato scelto pensando esplicitamente a T=500 a 500 Hz
sono i kernel size della CNN (vedi commenti in LeadCNNEncoder): sono tarati
sulla durata reale in millisecondi delle onde ECG (P, QRS, T) a questa
frequenza di campionamento, non sulla lunghezza totale della finestra.

Pipeline concettuale:
    segnale grezzo per derivazione (12 nodi, T campioni ciascuno)
        -> LeadCNNEncoder (CNN 1D condivisa, estrae un vettore di feature
           per ciascuna derivazione)
        -> N x SignedGCNBlock / GATv2Conv (message passing tra derivazioni,
           sfruttando la topologia SIGNED definita in dataset.py)
        -> global mean+max pooling (embedding dell'intero ECG a 12 derivazioni)
        -> MLP di classificazione

Il ramo con attenzione (use_attention=True, GATv2Conv) non soffre dello
stesso vincolo, perché non normalizza per il grado: può quindi lavorare
direttamente sul grafo COMPLETO con il peso signed passato come edge
feature (edge_attr), lasciando che l'attenzione impari da sé come pesare
relazioni concordi/discordi.
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


class SignedGCNBlock(nn.Module):
    """
    Singolo blocco di message passing "signed".

    Esegue la convoluzione grafica separatamente su:
        - il sotto-grafo delle relazioni CONCORDI (edge_index_pos/weight_pos),
          usando le feature dei nodi così come sono;
        - il sotto-grafo delle relazioni DISCORDI (edge_index_neg/weight_neg),
          usando le feature dei nodi NEGATE (-x), per rappresentare
          esplicitamente l'inversione di segno attesa fisiologicamente;
    e combina i due contributi con una somma.

    add_self_loops=False in entrambe le GCNConv: i self-loop sono già
    presenti esplicitamente nel sotto-grafo positivo (sim(i,i) = 1.0 per
    costruzione, vedi build_signed_ecg_graph_topology in dataset.py), quindi
    lasciare add_self_loops=True qui duplicherebbe il contributo del nodo
    su se stesso.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_pos = GCNConv(in_dim, out_dim, add_self_loops=False)
        self.conv_neg = GCNConv(in_dim, out_dim, add_self_loops=False)
        # proiezione lineare oer combinare i due rami
        self.lin = nn.Linear(out_dim * 2, out_dim)
        # LayerNorm gestita automaticamente da PyTorch
        # stabilizza e ri-scala i nodi con grado 0 nel ramo negativo
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index_pos, edge_weight_pos,
                edge_index_neg, edge_weight_neg):
        # Ramo concorde: feature originali, peso = similarità positiva.
        h_pos = self.conv_pos(x, edge_index_pos, edge_weight=edge_weight_pos)

        # Ramo discorde: feature NEGATE, peso = |similarità negativa|.
        h_neg = self.conv_neg(-x, edge_index_neg, edge_weight=edge_weight_neg)

        # concateno i due contributi per evitare che grandi valori nel ramo
        # positivo possano annullare quelli negativi (se somma algebrica).
        h = torch.cat([h_pos, h_neg], dim=1)
        # Proiettiamo alla dimensione di output
        out = self.lin(h)

        # La LayerNorm riallinea automaticamente la distribuzione dei nodi isolati
        return self.norm(out)


class ECG_GCN(nn.Module):
    """
    Signed-GCN per la classificazione del malposizionamento elettrodi.

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
        Numero di strati di message passing (SignedGCNBlock o GATv2Conv)
        impilati. Con soli 12 nodi nel grafo, 2 strati sono già sufficienti
        a propagare informazione tra qualunque coppia di derivazioni
        connesse anche indirettamente; aumentarli oltre 3 rischia solo di
        introdurre over-smoothing (i nodi finiscono per assomigliarsi troppo).
    use_attention : bool
        Se True usa GATv2Conv (sul grafo completo, con il peso firmato come
        edge_attr) al posto di SignedGCNBlock: impara pesi di attenzione
        sugli archi invece di usare pesi fissati a priori dall'affinità
        anatomica.
    dropout : float
        Dropout applicato nel classificatore finale.

    Forward
    -------
    Riceve un batch PyG (oggetto Data/Batch, vedi SignedECGData in
    dataset.py) con:
        data.x               -> [num_nodes_totali_batch, T]  (segnale grezzo)
        data.edge_index_pos  -> [2, num_edges_pos_totali_batch]
        data.edge_weight_pos -> [num_edges_pos_totali_batch]
        data.edge_index_neg  -> [2, num_edges_neg_totali_batch]
        data.edge_weight_neg -> [num_edges_neg_totali_batch]
        data.batch            -> [num_nodes_totali_batch]  (indice di grafo per nodo)
    Ritorna: logits di shape [batch_size, num_classes]
    """

    def __init__(self, node_feat_dim: int = 64, hidden_dim: int = 128,
                 num_classes: int = 13, num_gnn_layers: int = 2,
                 use_attention: bool = False, dropout: float = 0.3, use_mlp_classifier: bool = False):
        super().__init__()
        self.use_attention = use_attention
        self.cnn_encoder = LeadCNNEncoder(out_channels=node_feat_dim)

        if use_attention:
            # edge_dim=1: un'unica edge feature scalare, il peso signed
            # (similarità con segno) tra le due derivazioni.
            conv_layer = lambda in_c, out_c: GATv2Conv(
                in_c, out_c, heads=4, concat=False, edge_dim=1, add_self_loops=False
            )
        else:
            conv_layer = lambda in_c, out_c: SignedGCNBlock(in_c, out_c)

        self.convs = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(num_gnn_layers):
            self.convs.append(conv_layer(in_dim, hidden_dim))
            in_dim = hidden_dim

        if use_mlp_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),   # *2 per mean+max pooling
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index_pos, edge_weight_pos = data.edge_index_pos, data.edge_weight_pos
        edge_index_neg, edge_weight_neg = data.edge_index_neg, data.edge_weight_neg

        # x: [num_nodes_totali_batch, T]  ->  aggiungo canale per la Conv1d
        x = x.unsqueeze(1)                    # [num_nodes, 1, T]
        x = self.cnn_encoder(x)               # [num_nodes, node_feat_dim]

        if self.use_attention:
            # GATv2Conv non normalizza per grado: può quindi lavorare sul
            # grafo COMPLETO (positivo + negativo unificati) usando il peso
            # signed originale come edge feature, invece di dividere in
            # due sotto-grafi. Ricostruiamo qui il grafo completo:
            edge_index_full = torch.cat([edge_index_pos, edge_index_neg], dim=1)

            # Nel ramo negativo il peso è salvato come valore assoluto
            # (vincolo di non-negatività richiesto da GCNConv, non da
            # GATv2Conv): qui ripristiniamo il segno originale, così
            # l'attenzione vede la vera similarità signed in [-1, 1].
            edge_weight_signed = torch.cat([edge_weight_pos, -edge_weight_neg], dim=0)
            edge_attr = edge_weight_signed.unsqueeze(-1)  # [num_edges, 1]

            for conv in self.convs:
                x = F.relu(conv(x, edge_index_full, edge_attr=edge_attr))
        else:
            # Message passing signed a due rami (vedi SignedGCNBlock).
            # Relu è stata sostituita da LeakyReLU per evitare che i contributi
            # negativi vengano azzerati completamente.
            for conv in self.convs:
                x = F.leaky_relu(conv(x, edge_index_pos, edge_weight_pos,
                                      edge_index_neg, edge_weight_neg), negative_slope=0.1)
        # Effettuaiamo sia la media sia max pooling per ottenere
        # un valore più robusto dell'intero grafo
        x_mean = global_mean_pool(x, batch)   # [batch_size, hidden_dim]
        x_max = global_max_pool(x, batch)     # [batch_size, hidden_dim]
        graph_embedding = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(graph_embedding)