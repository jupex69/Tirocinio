"""Rappresentazione VERAMENTE non tabellare nel multiclasse: CNN 1D sulla
sequenza grezza (one-hot), valutata cross-studio, confrontata col RandomForest
tabellare.

Finora il multiclasse usava 78 feature scalari (74 + 4 nuovi descrittori). Qui la
sequenza grezza entra direttamente in una CNN (finestra a lunghezza fissa: la
lunghezza non e' sfruttabile). Domanda: la rappresentazione non tabellare recupera
segnale che i riassunti perdono, cross-studio?

(A) within-study (split casuale) e (B) cross-study (train un DB, test l'altro),
sulle stesse classi/campioni degli altri esperimenti onesti.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from eccdna_utils import read_fasta_stream
from honest_all_siamese import load_all
from full_classifier import FASTA, RESULTS_DIR, SEED

L = 200
MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot(seq):
    s = len(seq); st = max(0, (s - L) // 2); sub = seq[st:st + L]
    x = np.zeros((4, L), np.float32)
    for i, ch in enumerate(sub):
        j = MAP.get(ch)
        if j is not None:
            x[j, i] = 1.0
    return x


class CNN(nn.Module):
    def __init__(self, n_classes, ch=32, k=8, drop=0.3):
        super().__init__()
        self.c1 = nn.Conv1d(4, ch, k, padding=k // 2)
        self.c2 = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.fc = nn.Sequential(nn.Linear(ch, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, n_classes))

    def forward(self, x):
        h = F.relu(self.c1(x)); h = F.relu(self.c2(h)); h = h.max(dim=2).values
        return self.fc(h)


def train_cnn(Xtr, ytr, Xva, yva, n_classes, epochs=80, lr=1e-3, patience=12, seed=SEED):
    torch.manual_seed(seed)
    m = CNN(n_classes)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.as_tensor(Xtr); yt = torch.as_tensor(ytr)
    cnt = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = torch.as_tensor(len(ytr) / (n_classes * np.maximum(cnt, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    Xv = torch.as_tensor(Xva)
    n = len(Xt); best, best_state, no = -1, None, 0
    for _ in range(epochs):
        m.train(); perm = torch.randperm(n)
        for s in range(0, n, 128):
            idx = perm[s:s + 128]
            opt.zero_grad(); loss = crit(m(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc = accuracy_score(yva, m(Xv).argmax(1).numpy())
        if acc > best:
            best, best_state, no = acc, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            no += 1
            if no >= patience:
                break
    if best_state:
        m.load_state_dict(best_state)
    return m


@torch.no_grad()
def cnn_predict_labels(m, X, train_classes):
    m.eval()
    idx = m(torch.as_tensor(X)).argmax(1).numpy()
    return np.array([train_classes[i] for i in idx])


def fit_predict(Xa, ya, Xb):
    classesA = sorted(set(ya)); l2i = {c: i for i, c in enumerate(classesA)}
    ya_idx = np.array([l2i[c] for c in ya])
    cnt = np.bincount(ya_idx)
    strat = ya_idx if cnt.min() >= 2 else None
    Xtr, Xva, ytr, yva = train_test_split(Xa, ya_idx, test_size=0.15, random_state=SEED, stratify=strat)
    m = train_cnn(Xtr, ytr, Xva, yva, len(classesA))
    return cnn_predict_labels(m, Xb, classesA)


def main():
    df = load_all()
    ids = set(df["id"])
    print(f"Lettura sequenze grezze ({len(ids)} id) dal FASTA, finestra {L} bp...")
    seqs = {}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
        if "N" not in seq and len(seq) >= L:
            seqs[sid] = seq
    df = df[df["id"].isin(seqs)].reset_index(drop=True)
    classes = sorted(df["lab"].unique()); nC = len(classes)
    print(f"Classi: {nC}  campioni con len>={L}: {len(df)}  chance={1/nC:.3f}\n")

    X = np.stack([one_hot(seqs[i]) for i in df["id"]])
    y = df["lab"].to_numpy(); dbcol = df["source_db"].to_numpy()

    # (A) within-study
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    predA = fit_predict(Xtr, ytr, Xte)
    accA = accuracy_score(yte, predA)
    print(f"(A) CNN within-study (split casuale): accuracy={accA:.3f}  ({accA/(1/nC):.1f}x)")

    # (B) cross-study
    a = dbcol == "eccDNABase"; b = dbcol == "CircleBaseV2"
    p_ab = fit_predict(X[a], y[a], X[b]); p_ba = fit_predict(X[b], y[b], X[a])
    acc_ab = accuracy_score(y[b], p_ab); acc_ba = accuracy_score(y[a], p_ba)
    accB = (acc_ab + acc_ba) / 2
    print(f"(B) CNN cross-study (onesto): accuracy={accB:.3f}  (ecc->CB2={acc_ab:.3f}, CB2->ecc={acc_ba:.3f})  ({accB/(1/nC):.1f}x)")

    print("\n=== Confronto col tabellare (dagli esperimenti precedenti, stesse classi) ===")
    print("  RandomForest 78 feature: within~0.349  cross-study~0.290")
    print(f"  CNN sequenza grezza    : within {accA:.3f}  cross-study {accB:.3f}")
    print("  -> se CNN <= RF, la rappresentazione non tabellare NON aggiunge segnale.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([{"modello": "CNN sequenza grezza (non tabellare)", "within_study": round(accA, 3),
                   "cross_study": round(accB, 3), "x_caso_cross": round(accB / (1 / nC), 2)}]
                 ).to_csv(os.path.join(RESULTS_DIR, "cnn_multiclass.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/cnn_multiclass.tsv")


if __name__ == "__main__":
    main()
