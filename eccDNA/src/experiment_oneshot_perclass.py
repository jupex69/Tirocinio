"""Quante malattie identifica il one-shot? Recall PER-CLASSE in one-shot full-way
(9 classi, 1 esempio di supporto per classe), col miglior encoder (coseno,
1-shot training, 10 descrittori). Within-study e cross-studio. Caso = 1/9 = 0.111.
"""

import os
import numpy as np
import pandas as pd
import torch

from train_multiclass import standardize, train_prototypical, _class_prototypes, _proto_logits, SEED, DEVICE
from models_pytorch import _to_tensor
from eccdna_utils import DESCRIPTOR_NAMES
from experiment_oneshot import load_data, RESULTS_DIR


@torch.no_grad()
def perclass_oneshot(enc, Xa, ya, Xb, yb, n_classes, n_draws=300, seed=SEED):
    rng = np.random.default_rng(seed)
    Ea = enc(_to_tensor(Xa, DEVICE)); Eb = enc(_to_tensor(Xb, DEVICE))
    rec = np.full((n_draws, n_classes), np.nan)
    for d in range(n_draws):
        s_idx, s_lab = [], []
        for c in range(n_classes):
            pool = np.where(ya == c)[0]
            if len(pool) == 0:
                continue
            s_idx.append(rng.choice(pool)); s_lab.append(c)
        C = _class_prototypes(Ea[s_idx], torch.as_tensor(s_lab, device=DEVICE), n_classes, cosine=True)
        pred = _proto_logits(Eb, C, cosine=True).argmax(1).cpu().numpy()
        for c in range(n_classes):
            m = yb == c
            if m.any():
                rec[d, c] = (pred[m] == c).mean()
    return np.nanmean(rec, axis=0)


def main():
    df = load_data()
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    X = df[DESCRIPTOR_NAMES].to_numpy(np.float32); y = df["label"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()
    chance = 1 / nC

    rng = np.random.default_rng(SEED); idx = rng.permutation(len(df)); cut = int(0.7 * len(df))
    tr = idx[:cut]
    m, s = X[tr].mean(0), X[tr].std(0); s[s == 0] = 1.0
    Z = (X - m) / s
    va = int(0.85 * len(tr))
    enc, _ = train_prototypical(Z[tr][:va], y[tr][:va], Z[tr][va:], y[tr][va:], nC, cosine=True, seed=SEED,
                               select_metric="balanced", n_support=1, n_query=5)

    a = db == "eccDNABase"; b = db == "CircleBaseV2"
    # within-study: usa il test set (support+query dallo stesso pool test)
    te = idx[cut:]
    rec_w = perclass_oneshot(enc, Z[te], y[te], Z[te], y[te], nC)
    # cross-studio: support da A, query da B (media delle due direzioni)
    rec_ab = perclass_oneshot(enc, Z[a], y[a], Z[b], y[b], nC)
    rec_ba = perclass_oneshot(enc, Z[b], y[b], Z[a], y[a], nC)
    rec_c = np.nanmean([rec_ab, rec_ba], axis=0)

    res = pd.DataFrame({"malattia": classes,
                        "recall_within": np.round(rec_w, 3),
                        "recall_cross": np.round(rec_c, 3)}).sort_values("recall_cross", ascending=False)
    print(f"One-shot full-way ({nC} classi), 1 esempio/classe. Caso = {chance:.3f}\n")
    print(res.to_string(index=False))
    n_w = int((res["recall_within"] > chance * 1.5).sum())
    n_c = int((res["recall_cross"] > chance * 1.5).sum())
    print(f"\nMalattie con recall > 1.5x il caso ({chance*1.5:.3f}):  within={n_w}  cross-studio={n_c}")
    print(">>> cross-studio: sono queste le malattie che il one-shot identifica davvero.")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "oneshot_perclass.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/oneshot_perclass.tsv")


if __name__ == "__main__":
    main()
