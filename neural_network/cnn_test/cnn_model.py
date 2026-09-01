"""
================================================================================
Architettura baseline: CNN 1D "stand-alone" per la rilevazione
del malposizionamento degli elettrodi ECG.
================================================================================

Questo file è la controparte "senza grafo" di model.py (Signed-GCN).
Serve come baseline per capire quanto valore aggiunge esplicitamente la
topologia signed tra derivazioni rispetto a una CNN che vede semplicemente
le 12 derivazioni come 12 canali di input di un unico segnale multivariato.

--------------------------------------------------------------------------------
Differenza concettuale rispetto a model.py
--------------------------------------------------------------------------------
Nel modello Signed-GCN il flusso era:
    1) LeadCNNEncoder: una CNN 1D condivisa viene applicata a ogni
       derivazione singolarmente (in_channels=1) -> ogni derivazione produce
       un proprio vettore di feature, in modo indipendente dalle altre.
    2) SignedGCNBlock / GATv2Conv: la mescolanza di informazione TRA
       derivazioni avviene solo dopo, ed è guidata esplicitamente dalla
       topologia (positiva/negativa) definita a priori in dataset.py.

Qui invece non abbiamo un grafo che dica alla rete quali derivazioni sono
"concordi" o "discordi": diamo in input tutte le 12 derivazioni insieme,
come 12 canali di un unico tensore [N, 12, T].
Di conseguenza:
    - la CNN deve imparare da sola, dai dati, quali derivazioni sono
      correlate/anti-correlate tra loro;
    - non c'è più una separazione del tipo "prima estraggo feature per
      derivazione, poi le combino": qui estrazione di feature e
      combinazione tra derivazioni avvengono insieme, in ogni layer.

Per rendere il confronto il più possibile equo rispetto al modello con
grafo, manteniamo:
    - gli stessi kernel size (7 -> 5 -> 3), scelti con lo stesso ragionamento
      fisiologico usato in LeadCNNEncoder (durata reale in ms delle onde
      ECG a 500 Hz), vedi commenti sotto;
    - un pooling finale "mean+max" concettualmente analogo a quello usato
      dopo la GCN, con la differenza che qui il pooling è fatto sulla
      dimensione TEMPORALE (non esiste più una dimensione "nodo/derivazione"
      separata, essendo le derivazioni state fuse nei canali);
    - una capacità (numero di parametri) simile, per non favorire un
      modello solo perché più grande.

Input:  [N, 12, T]  con T = 500 campioni (finestra di 1s a 500 Hz), ma anche
        qui, come in LeadCNNEncoder, il AdaptiveAvgPool1d/AdaptiveMaxPool1d
        finale rende la rete indipendente da T.
Output: [N, num_classes]  logits
"""

import torch
import torch.nn as nn


class ECG_CNN(nn.Module):
    """
    CNN 1D multi-canale che tratta le 12 derivazioni come 12 canali di
    input di un unico segnale multivariato.

    Parametri
    ---------
    in_channels : int
        Numero di derivazioni in ingresso. 12 nel nostro caso (le 12
        derivazioni ECG standard, stesso ordine di LEAD_ORDER in
        cnn_dataaset.py / dataset.py).
    hidden_channels : tuple[int, int, int]
        Numero di canali prodotti dai 3 blocchi convoluzionali. Scelti per
        avere una capacità simile a LeadCNNEncoder + SignedGCNBlock insieme
        (64 feature per nodo -> 128 hidden nella GCN).
    num_classes : int
        Numero di classi da predire (come in model.py: normale
        + tipi di inversione).
    dropout : float
        Dropout applicato nel classificatore finale e dopo l'ultimo blocco
        convoluzionale.

    Dimensionamento dei kernel (stesso ragionamento di LeadCNNEncoder,
    a 500 Hz, 1 campione = 2 ms):
        - blocco 1, kernel=7 -> finestra ricettiva locale di 14 ms: segue i
          bordi ripidi del complesso QRS (80-120 ms, 40-60 campioni)
        - blocco 2, kernel=5 -> 10 ms sul segnale già sotto-campionato
          (dopo il primo MaxPool)
        - blocco 3, kernel=3 -> raffinamento finale prima del pooling globale
    Tre blocchi con MaxPool(2): T=500 -> 250 -> 125 prima del pooling
    adattivo finale (125 campioni = 250 ms, sufficiente a contenere l'intero
    complesso QRS anche all'ultimo layer convoluzionale).
    """

    def __init__(self, in_channels: int = 12, hidden_channels=(16, 32, 64),
                 num_classes: int = 13, dropout: float = 0.3, use_mlp_classifier: bool = False):
        super().__init__()
        c1, c2, c3 = hidden_channels

        self.features = nn.Sequential(
            # Blocco 1: mescola per la prima volta le 12 derivazioni tra
            # loro (in_channels=12 -> c1). E' qui che la rete comincia a
            # imparare da sola le relazioni tra derivazioni che nel modello
            # con grafo erano invece imposte a priori dalla topologia signed.
            nn.Conv1d(in_channels, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                       # 500 -> 250

            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                       # 250 -> 125

            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)

        # Pooling globale sulla dimensione temporale, analogo concettuale
        # del "global mean+max pool sui nodi del grafo" in model.py: lì si
        # riassumeva l'informazione sulle 12 derivazioni, qui si riassume
        # l'informazione sull'intera finestra temporale (le derivazioni
        # sono già state fuse nei canali dai layer convoluzionali sopra).
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        if use_mlp_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(c3 * 2, c3),   # *2 per la concatenazione mean+max
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(c3, num_classes),
            )
        else:
            self.classifier = nn.Linear(c3 * 2, num_classes)

    def forward(self, x):
        # x: [N, 12, T]
        x = self.features(x)              # [N, c3, T']
        x = self.dropout(x)

        x_mean = self.global_avg_pool(x).squeeze(-1)   # [N, c3]
        x_max = self.global_max_pool(x).squeeze(-1)     # [N, c3]
        x = torch.cat([x_mean, x_max], dim=1)            # [N, c3 * 2]

        return self.classifier(x)                        # [N, num_classes]