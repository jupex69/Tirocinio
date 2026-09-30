"""Come experiment_richer_kmers.py ma con le RETI SIAMESI nelle 3 loss (prototipico
euclideo, prototipico coseno, triplet batch-hard), sugli stessi set di feature
(10 -> +3mer -> +4mer -> +5mer). Within-study e cross-studio, length-matched.
Riusa la cache dei k-meri costruita dal run RandomForest.
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from train_multiclass import (standardize, train_prototypical, predict_prototypical,
                              _class_prototypes, _proto_logits, SEED, DEVICE)
from experiment_losses_fewshot import train_metric_encoder
from models_pytorch import _to_tensor
from full_classifier import RESULTS_DIR
from experiment_richer_kmers import load, length_match, SETS


@torch.no_grad()
def predict_triplet(enc, Xtr, ytr, Xte, nC):
    enc.eval()
    C = _class_prototypes(enc(_to_tensor(Xtr, DEVICE)), torch.as_tensor(ytr, device=DEVICE), nC, cosine=False)
    return _proto_logits(enc(_to_tensor(Xte, DEVICE)), C, cosine=False).argmax(1).cpu().numpy()


def train_and_pred(kind, Xtr, ytr, Xte, nC):
    va = int(0.85 * len(Xtr))
    if kind in ("euclideo", "coseno"):
        cos = kind == "coseno"
        enc, _ = train_prototypical(Xtr[:va], ytr[:va], Xtr[va:], ytr[va:], nC, cosine=cos, seed=SEED,
                                    select_metric="balanced", epochs=60, episodes=30, patience=8)
        return predict_prototypical(enc, Xtr, ytr, Xte, nC, cos).argmax(1)
    enc, _ = train_metric_encoder(Xtr[:va], ytr[:va], Xtr[va:], ytr[va:], nC, "triplet", margin=0.3,
                                  epochs=60, batches=30, patience=8)
    return predict_triplet(enc, Xtr, ytr, Xte, nC)


def main():
    df = length_match(load())
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    chance = 1 / nC
    y = df["label"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()
    a = db == "eccDNABase"; b = db == "CircleBaseV2"
    print(f"9 malattie, {len(df)} seq (length-matched), caso={chance:.3f}. Reti siamesi, 3 loss.\n")

    rows = []
    # shuffle una volta per lo split within (i dati sono raggruppati per classe)
    for name, cols in SETS.items():
        X = df[cols].to_numpy(np.float32)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        Xtr_s, Xte_s = standardize(Xtr, Xte)
        XA_s, XB_s = standardize(X[a], X[b]); XB2_s, XA2_s = standardize(X[b], X[a])
        for loss in ["euclideo", "coseno", "triplet"]:
            wp = train_and_pred(loss, Xtr_s, ytr, Xte_s, nC)
            w_acc = accuracy_score(yte, wp); w_f1 = f1_score(yte, wp, average="macro", zero_division=0)
            p1 = train_and_pred(loss, XA_s, y[a], XB_s, nC)
            p2 = train_and_pred(loss, XB2_s, y[b], XA2_s, nC)
            c_acc = (accuracy_score(y[b], p1) + accuracy_score(y[a], p2)) / 2
            c_f1 = (f1_score(y[b], p1, average="macro", zero_division=0) + f1_score(y[a], p2, average="macro", zero_division=0)) / 2
            rows.append({"set": name, "loss": loss, "within_acc": round(w_acc, 3), "within_F1": round(w_f1, 3),
                         "within_xcaso": round(w_acc / chance, 2), "cross_acc": round(c_acc, 3),
                         "cross_F1": round(c_f1, 3), "cross_xcaso": round(c_acc / chance, 2)})
            print(f"  {name:18s} {loss:9s} within: acc={w_acc:.3f} F1={w_f1:.3f} ({w_acc/chance:.1f}x) | cross: acc={c_acc:.3f} F1={c_f1:.3f} ({c_acc/chance:.1f}x)")

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "richer_kmers_siamese.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/richer_kmers_siamese.tsv")


if __name__ == "__main__":
    main()
