"""Confronto DEFINITIVO per la tesi: RandomForest vs le 3 loss siamesi (euclideo,
coseno, triplet) in ONE-SHOT e FEW-SHOT. Stesso preprocessing per tutti: 9 malattie,
10 descrittori, CAP=600. Within-study (split casuale) e cross-studio (leave-one-DB-out).
Caso = 1/9 = 0.111. Un'unica tabella coerente.
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from eccdna_utils import DESCRIPTOR_NAMES
from train_multiclass import (standardize, train_prototypical, _class_prototypes, _proto_logits, SEED, DEVICE)
from experiment_losses_fewshot import train_metric_encoder
from models_pytorch import _to_tensor
from experiment_oneshot import _episode, RESULTS_DIR
from experiment_richer_kmers import load as load_km, length_match

DBS = ["eccDNABase", "CircleBaseV2"]


def std_by(Xfit, *arrays):
    m, s = Xfit.mean(0), Xfit.std(0); s = np.where(s == 0, 1, s)
    return [((A - m) / s).astype(np.float32) for A in arrays]


def train_enc(kind, Xtr, ytr, nC):
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(Xtr))   # shuffle: dati raggruppati per classe
    Xtr, ytr = Xtr[perm], ytr[perm]
    va = int(0.85 * len(Xtr))
    if kind in ("euclideo", "coseno"):
        enc, _ = train_prototypical(Xtr[:va], ytr[:va], Xtr[va:], ytr[va:], nC, cosine=(kind == "coseno"), seed=SEED,
                                    select_metric="balanced", n_support=1, n_query=5, epochs=80, episodes=30, patience=10)
    else:
        enc, _ = train_metric_encoder(Xtr[:va], ytr[:va], Xtr[va:], ytr[va:], nC, "triplet", margin=0.3,
                                      epochs=80, batches=30, patience=10)
    return enc


@torch.no_grad()
def eval_shot(enc, Xs, ys, Xq, yq, k, cosine, n_ep=600, seed=SEED):
    rng = np.random.default_rng(seed); enc.eval()
    Es = enc(_to_tensor(Xs, DEVICE)); Eq = enc(_to_tensor(Xq, DEVICE))
    classes = np.array([c for c in np.unique(ys) if (ys == c).sum() >= 1 and (yq == c).sum() >= 1])
    accs = []
    for _ in range(n_ep):
        si, sl, qi, ql, _ = _episode(rng, ys, classes, len(classes), k, 10, y2=yq)
        C = _class_prototypes(Es[si], torch.as_tensor(sl, device=DEVICE), len(classes), cosine)
        pred = _proto_logits(Eq[qi], C, cosine).argmax(1).cpu().numpy()
        accs.append((pred == ql).mean())
    return float(np.mean(accs))


def main():
    df = length_match(load_km())
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    chance = 1 / nC
    X = df[DESCRIPTOR_NAMES].to_numpy(np.float32); y = df["label"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()
    a = db == DBS[0]; b = db == DBS[1]
    print(f"9 malattie, {len(df)} sequenze (length-matched), 10 descrittori, caso={chance:.3f}\n")
    rows = []

    # --- RandomForest (usa TUTTI gli esempi) ---
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(Xtr_s, ytr)
    w_rf = accuracy_score(yte, rf.predict(Xte_s))
    XA, XB = standardize(X[a], X[b]); XB2, XA2 = standardize(X[b], X[a])
    rfa = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(XA, y[a])
    rfb = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(XB2, y[b])
    c_rf = (accuracy_score(y[b], rfa.predict(XB)) + accuracy_score(y[a], rfb.predict(XA2))) / 2
    rows.append({"metodo": "RandomForest (tutti gli esempi)", "regime": "standard",
                 "within": round(w_rf, 3), "within_x": round(w_rf / chance, 2),
                 "cross": round(c_rf, 3), "cross_x": round(c_rf / chance, 2)})
    print(f"  RandomForest        within={w_rf:.3f} ({w_rf/chance:.1f}x)  cross={c_rf:.3f} ({c_rf/chance:.1f}x)")

    # --- 3 loss siamesi, one-shot (K=1) e few-shot (K=5) ---
    for loss in ["euclideo", "coseno", "triplet"]:
        cos = (loss == "coseno")
        # within: encoder su train, episodi dal test
        enc_w = train_enc(loss, Xtr_s, ytr, nC)
        # cross: encoder LOSO per direzione
        encA = train_enc(loss, XA, y[a], nC); encB = train_enc(loss, XB2, y[b], nC)
        for k, reg in [(1, "one-shot"), (5, "few-shot")]:
            w = eval_shot(enc_w, Xtr_s, ytr, Xte_s, yte, k, cos)
            c = (eval_shot(encA, XA, y[a], XB, y[b], k, cos) + eval_shot(encB, XB2, y[b], XA2, y[a], k, cos)) / 2
            rows.append({"metodo": f"Siamese {loss}", "regime": reg,
                         "within": round(w, 3), "within_x": round(w / chance, 2),
                         "cross": round(c, 3), "cross_x": round(c / chance, 2)})
            print(f"  Siamese {loss:9s} {reg:9s} within={w:.3f} ({w/chance:.1f}x)  cross={c:.3f} ({c/chance:.1f}x)")

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "final_comparison.tsv"), sep="\t", index=False)
    print("\n" + res.to_string(index=False))
    print(f"\ncaso={chance:.3f}. Salvato in {RESULTS_DIR}/final_comparison.tsv")


if __name__ == "__main__":
    main()
