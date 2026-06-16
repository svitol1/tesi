import torch.nn as nn


class ecg_classifier(nn.Module):
    """CNN pensata per finestre di singolo battito (input corti).

    Usa blocchi Conv-BN-ReLU-Pool e termina con AdaptiveAvgPool1d(1)
    per ottenere una rappresentazione indipendente dalla lunghezza esatta
    dell'input. Riduce la sensibilità ai cambi di dimensione e semplifica
    il calcolo del flattening per lo strato fully-connected.
    """

    def __init__(self, num_classi=4, in_channels=1):
        super(ecg_classifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(4),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classi),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
