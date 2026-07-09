"""Modello CNN 1D per la classificazione diretta sano/malato di eccDNA.

Architettura ispirata al CNN di DeepCircle (Chang et al., 2023) - due
livelli convoluzionali + dropout, che su un task molto simile (eccDNA da
sequenza, ~1000bp) raggiunge un'accuratezza intorno a 0.80 - con l'aggiunta
di un terzo livello convoluzionale e batch-normalization (idea di ECCNET,
Zeng et al., 2025) per una rappresentazione un po' piu' espressiva.
Dimensionata volutamente in modo leggero per poter allenare su CPU in tempi
ragionevoli (nessuna GPU disponibile su questa macchina).

L'input atteso e' una sequenza one-hot a 4 canali (A, C, G, T) con
augmentation circolare a finestra fissa, prodotta da
eccdna_utils.one_hot_encode_circular: tensore di forma (batch, 4, window_size).

L'uso di AdaptiveAvgPool1d rende il modello indipendente dalla window_size
esatta, cosi' la stessa architettura funziona qualunque sia la finestra
scelta in fase di training.
"""

import torch
import torch.nn as nn


class EccDNACNN(nn.Module):
    def __init__(self, dropout=0.2, dense_dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dense_dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, 4, window_size)
        x = self.features(x)
        return self.classifier(x).squeeze(-1)  # logit binario, (batch,)


if __name__ == "__main__":
    # Piccolo self-test: verifica che il forward pass funzioni con una
    # finestra qualsiasi e produca un logit per ogni elemento del batch.
    model = EccDNACNN()
    dummy = torch.zeros((8, 4, 1024))
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print("output shape:", out.shape, "| parametri totali:", n_params)
