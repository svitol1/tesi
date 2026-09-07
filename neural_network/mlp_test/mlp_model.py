"""
================================================================================
Architettura: MLP per la rilevazione
del malposizionamento degli elettrodi ECG.
================================================================================

Questo file è la terza variante della catena di reti neurali.

Qui rimuoviamo anche quest'ultimo bias: l'MLP vede l'intero segmento come
un unico vettore piatto di numeri, senza alcuna nozione built-in di "quali
campioni sono temporalmente vicini" né di "quali derivazioni sono vicine
tra loro". Ogni connessione tra un campione di input e un neurone nascosto
è un parametro libero, indipendente dagli altri: non c'è parameter sharing
di alcun tipo (né lungo il tempo, come nella CNN, né guidato dalla
topologia, come nella GCN).

--------------------------------------------------------------------------------
Perché questa baseline è utile
--------------------------------------------------------------------------------

Se l'MLP funziona quasi quanto la CNN, vuol dire che il task è "facile"
anche senza l'utilizzo del bias temporale. Se invece l'MLP è nettamente
peggiore, la struttura locale del segnale ECG (che la CNN sfrutta) è
importante per il problema. (cosa molto probabile)

--------------------------------------------------------------------------------
Differenza pratica IMPORTANTE rispetto a ECG_CNN: la lunghezza fissa
--------------------------------------------------------------------------------
In cnn_model.py, il AdaptiveAvgPool1d/AdaptiveMaxPool1d finale rende la CNN
indipendente da T: si può allenare o fare inferenza con finestre di
lunghezza diversa senza toccare l'architettura.
Qui questo NON è possibile: un nn.Linear ha un numero fisso di
in_features, quindi la lunghezza della finestra (seq_len) e il numero di
derivazioni (in_channels) devono essere fissati a priori nel costruttore e
devono corrispondere ESATTAMENTE a quelli dei dati usati in training e in
inferenza. Con seq_len=500 e in_channels=12 (come nel resto della
pipeline), il primo layer riceve un vettore di 12*500 = 6000 feature.

--------------------------------------------------------------------------------
Nota sul numero di parametri
--------------------------------------------------------------------------------
A differenza della CNN (che riusa gli stessi pochi filtri su tutta la
finestra temporale) e della GCN (che riusa gli stessi pesi su tutti i
nodi/derivazioni), l'MLP non condivide alcun parametro: già il solo primo
layer (6000 -> hidden_dims[0]) ha 6000 * hidden_dims[0] pesi. Per questo
motivo, a parità di hidden_dims "ragionevoli" per il task, l'MLP avrà
inevitabilmente MOLTI più parametri di CNN e GCN: non è possibile ottenere
una parità di capacità stretta come tra CNN e GCN senza rendere l'MLP
artificialmente troppo piccolo (e quindi svantaggiarlo) o troppo grande
(e quindi favorirlo per pura sovraparametrizzazione, non per l'architettura
in sé).

Input:  [N, 12, T]  con T = seq_len (default 500 campioni, 1s a 500 Hz).
        A differenza della CNN, T deve essere fisso e noto in anticipo.
Output: [N, num_classes]  logits
"""

import torch
import torch.nn as nn


class ECG_MLP(nn.Module):
    """
    MLP: appiattisce le 12 derivazioni x T campioni in un
    unico vettore di input e lo passa attraverso una sequenza di layer
    fully-connected (Linear -> BatchNorm1d -> ReLU -> Dropout).

    Parametri
    ---------
    in_channels : int
        Numero di derivazioni in ingresso. 12 nel nostro caso, stesso
        ordine di LEAD_ORDER usato in mlp_dataset.py / cnn_dataset.py /
        dataset.py.
    seq_len : int
        Numero di campioni temporali per derivazione (default 500, cioè
        1s a 500 Hz). Deve corrispondere ESATTAMENTE alla lunghezza dei
        segmenti usati in training e in inferenza, perché determina il
        numero fisso di input features (in_channels * seq_len) del primo
        layer Linear.
    hidden_dims : tuple[int, ...]
        Dimensioni dei layer nascosti fully-connected, in ordine. Il
        default (512, 256, 128) è stato scelto per dare all'MLP una
        capacità rappresentativa comparabile (stesso ordine di grandezza
        di feature finali, 128 come i 128 canali finali della CNN) pur
        sapendo che il numero totale di parametri resterà più alto per via
        della mancanza di parameter sharing (vedi docstring del modulo).
    num_classes : int
        Numero di classi da predire (come in model.py e cnn_model.py:
        normale + tipi di inversione).
    dropout : float
        Dropout applicato dopo ogni layer nascosto e prima del
        classificatore finale. Con un MLP "flatten totale" il rischio di
        overfitting è maggiore rispetto a CNN/GCN (niente parameter
        sharing => molti più parametri liberi a parità di hidden units),
        quindi qui il dropout ha un ruolo regolarizzante ancora più
        importante.
    use_batchnorm : bool
        Se True (default), applica BatchNorm1d dopo ogni layer Linear
        nascosto. Utile per stabilizzare il training di un MLP profondo su
        input con scale diverse (anche se i dati sono già normalizzati per
        derivazione in mlp_dataset.py).
    """

    def __init__(self, in_channels: int = 12, seq_len: int = 500,
                 hidden_dims=(512, 256, 128), num_classes: int = 13,
                 dropout: float = 0.3, use_batchnorm: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.input_dim = in_channels * seq_len  # 12 * 500 = 6000 di default

        layers = []
        prev_dim = self.input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        # x: [N, 12, T] -> [N, 12*T]
        n = x.size(0)

        if x.size(1) != self.in_channels or x.size(2) != self.seq_len:
            raise ValueError(
                f"ECG_MLP si aspetta input [N, {self.in_channels}, "
                f"{self.seq_len}], ricevuto [N, {x.size(1)}, {x.size(2)}]. "
                "A differenza della CNN, l'MLP non è indipendente da T: "
                "seq_len deve combaciare con quello passato al costruttore."
            )

        x = x.reshape(n, -1)          # [N, 12*T], flatten totale
        x = self.backbone(x)          # [N, hidden_dims[-1]]

        return self.classifier(x)     # [N, num_classes]
