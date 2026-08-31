"""Modello FINALE per la classificazione multiclasse della malattia (17 classi):
rete siamese prototipica-coseno su una rappresentazione RICCA della sequenza =
spettro dei 3-mer (64 valori) + i 10 descrittori biologico/statistici (74 feature).

Perche' questa configurazione (esiti degli esperimenti):
- Sui soli 10 descrittori scalari la siamese non batte un softmax bilanciato:
  con poche feature tabellari vince il classificatore diretto (vedi train_multiclass.py).
- Dando alla siamese lo SPETTRO k-mer completo, invece dei riassunti scalari, il
  metric learning trova finalmente segnale sulle malattie RARE che gli altri
  modelli ignorano (es. Liver, Systemic lupus). E' il compito in cui la rete
  siamese e' strutturalmente adatta (experiment_kmer_siamese.py).
- Concatenare spettro + descrittori (74) da' il miglior compromesso: massimo F1
  sulle rare e recupero delle classi comuni.
- Aggiungere altre feature hand-crafted (ripetizioni dirette/invertite,
  experiment_repeat_features.py) NON aiuta (ridondante o addirittura dannoso):
  la leva utile e' la rappresentazione ricca, non piu' descrittori.

Selezione del modello su BALANCED ACCURACY (obiettivo: non perdere i tumori
minori). Metriche riportate: globali (accuracy, balanced accuracy, macro-F1,
top-3) e per-classe, con aggregati separati per classi rare e comuni.

LIMITE DICHIARATO: il metodo di sequenziamento e' quasi un proxy della malattia;
parte del segnale multiclasse puo' derivare dal protocollo (vedi multiclass_data.py).
"""

import os
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score, f1_score

from eccdna_utils import DESCRIPTOR_NAMES, read_fasta_stream, _kmer_counts, BASES
from multiclass_data import build_multiclass_splits
from train_multiclass import (
    standardize, train_prototypical, predict_prototypical,
    _class_prototypes, evaluate, SEED, DEVICE,
)
from models_pytorch import _to_tensor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTA = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection.body.fa")
OUT_DIR = os.path.join(SCRIPT_DIR, "data/processed")
KMER_COLS = ["".join(t) for t in product(BASES, repeat=3)]  # 64 tri-nucleotidi
FEATURE_COLS = KMER_COLS + DESCRIPTOR_NAMES                  # 74 feature
RARE_MAX_TRAIN = 800


def kmer_spectrum(sequence):
    """Frequenze dei 64 tri-nucleotidi sovrapposti (rappresentazione ricca:
    lo spettro completo di cui entropia e Karlin sono solo riassunti scalari)."""
    tri, tot = _kmer_counts(sequence, 3)
    if tot == 0:
        return {c: 0.0 for c in KMER_COLS}
    return {c: tri.get(c, 0) / tot for c in KMER_COLS}


def _attach_spectrum(df, table):
    r = table.reindex([str(i) for i in df["id"]]).reset_index(drop=True)
    return pd.concat([df.reset_index(drop=True), r], axis=1).dropna(subset=KMER_COLS)


def build_rich_splits():
    """(train, val, test) con lo spettro 3-mer unito ai 10 descrittori."""
    tr, va, te = build_multiclass_splits()
    all_ids = set(tr["id"]) | set(va["id"]) | set(te["id"])
    spec = {}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=all_ids):
        if "N" not in seq and len(seq) >= 3:
            spec[sid] = kmer_spectrum(seq)
    tab = pd.DataFrame.from_dict(spec, orient="index")[KMER_COLS]
    return _attach_spectrum(tr, tab), _attach_spectrum(va, tab), _attach_spectrum(te, tab)


def per_class_report(y_true, proba, classes, support_train):
    y_pred = proba.argmax(1)
    rec = recall_score(y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    return pd.DataFrame({
        "malattia": classes,
        "train_n": [support_train[c] for c in classes],
        "recall": rec, "f1": f1,
    }).set_index("malattia").sort_values("train_n")


def main():
    print("--- Modello finale: siamese prototipica-coseno su 74 feature ---\n")
    tr, va, te = build_rich_splits()
    classes = sorted(tr["disease"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    support_train = tr["disease"].value_counts().to_dict()

    Xtr = tr[FEATURE_COLS].to_numpy(np.float32)
    Xva = va[FEATURE_COLS].to_numpy(np.float32)
    Xte = te[FEATURE_COLS].to_numpy(np.float32)
    ytr = tr["disease"].map(c2i).to_numpy()
    yva = va["disease"].map(c2i).to_numpy()
    yte = te["disease"].map(c2i).to_numpy()

    mean, std = Xtr.mean(0), Xtr.std(0); std[std == 0] = 1.0
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)

    print(f"Feature: {len(FEATURE_COLS)} (64 spettro 3-mer + 10 descrittori)")
    print(f"Campioni: train={len(Xtr)} val={len(Xva)} test={len(Xte)}  classi={n_classes}\n")

    enc, val_score = train_prototypical(
        Xtr, ytr, Xva, yva, n_classes, cosine=True, seed=SEED, select_metric="balanced")
    print(f"Balanced accuracy in validation (miglior epoca): {val_score:.3f}\n")

    proba = predict_prototypical(enc, Xtr, ytr, Xte, n_classes, cosine=True)
    print("--- Metriche globali (test) ---")
    evaluate("Siamese 74 (finale)", yte, proba, n_classes)

    rep = per_class_report(yte, proba, classes, support_train)
    print("\n--- Per classe (dalla piu' rara alla piu' comune) ---")
    print(rep.round(3).to_string())
    rare = rep[rep["train_n"] <= RARE_MAX_TRAIN]
    com = rep[rep["train_n"] > RARE_MAX_TRAIN]
    print(f"\nRARE ({len(rare)}): recall medio={rare['recall'].mean():.3f}  F1 medio={rare['f1'].mean():.3f}")
    print(f"COMUNI ({len(com)}): recall medio={com['recall'].mean():.3f}  F1 medio={com['f1'].mean():.3f}")

    # salva modello (encoder + prototipi + standardizzazione) e metriche
    with torch.no_grad():
        C = _class_prototypes(enc(_to_tensor(Xtr, DEVICE)),
                              torch.as_tensor(ytr, device=DEVICE), n_classes, cosine=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({
        "encoder_state": enc.state_dict(),
        "prototypes": C.cpu(),
        "classes": classes,
        "feature_cols": FEATURE_COLS,
        "mean": mean, "std": std,
    }, os.path.join(OUT_DIR, "siamese_multiclass_final.pt"))
    rep.to_csv(os.path.join(OUT_DIR, "siamese_multiclass_per_class.tsv"), sep="\t")
    print(f"\nModello salvato in {os.path.join(OUT_DIR, 'siamese_multiclass_final.pt')}")
    print(f"Metriche per-classe in {os.path.join(OUT_DIR, 'siamese_multiclass_per_class.tsv')}")


if __name__ == "__main__":
    main()
