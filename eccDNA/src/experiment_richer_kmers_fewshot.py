"""K-meri di ordine crescente x reti siamesi, in ONE-SHOT e FEW-SHOT (3 loss).

Per ogni set di feature (10 -> +3mer -> +4mer) e per ogni loss (euclideo, coseno,
triplet), encoder addestrato in modo episodico su UN database (leave-one-study-out),
poi valutazione CROSS-STUDIO a K=1 (one-shot) e K=5 (few-shot), media delle due
direzioni. Domanda: i k-meri lunghi aiutano il one/few-shot a identificare la malattia?
Caso = 1/9. Riusa la cache dei k-meri.
"""
import os
import numpy as np
import pandas as pd
import torch

from train_multiclass import (train_prototypical, _class_prototypes, _proto_logits, SEED, DEVICE)
from experiment_losses_fewshot import train_metric_encoder
from models_pytorch import _to_tensor
from experiment_oneshot import _episode
from full_classifier import RESULTS_DIR
from experiment_richer_kmers import load, length_match, SETS

DBS = ["eccDNABase", "CircleBaseV2"]
SETS3 = {k: v for k, v in SETS.items() if not k.startswith("D")}   # 10, +3mer, +4mer (no 5mer: overfit/lento)
SHOTS = [1, 5]


def std_by(Xfit, *arrays):
    m, s = Xfit.mean(0), Xfit.std(0); s = np.where(s == 0, 1, s)
    return [((A - m) / s).astype(np.float32) for A in arrays]


def train_enc(kind, Xtr, ytr, nC):
    va = int(0.85 * len(Xtr))
    if kind in ("euclideo", "coseno"):
        enc, _ = train_prototypical(Xtr[:va], ytr[:va], Xtr[va:], ytr[va:], nC, cosine=(kind == "coseno"), seed=SEED,
                                    select_metric="balanced", n_support=1, n_query=5, epochs=60, episodes=30, patience=8)
    else:
        enc, _ = train_metric_encoder(Xtr[:va], ytr[:va], Xtr[va:], ytr[va:], nC, "triplet", margin=0.3,
                                      epochs=60, batches=30, patience=8)
    return enc


@torch.no_grad()
def eval_shot(enc, Xs, ys, Xq, yq, k, cosine, nC, n_ep=600, seed=SEED):
    rng = np.random.default_rng(seed); enc.eval()
    Es = enc(_to_tensor(Xs, DEVICE)); Eq = enc(_to_tensor(Xq, DEVICE))
    classes = np.array([c for c in np.unique(ys) if (ys == c).sum() >= 1 and (yq == c).sum() >= 1])
    accs = []
    for _ in range(n_ep):
        si, sl, qi, ql, _ = _episode(rng, ys, classes, len(classes), k, 8, y2=yq)
        C = _class_prototypes(Es[si], torch.as_tensor(sl, device=DEVICE), len(classes), cosine)
        pred = _proto_logits(Eq[qi], C, cosine).argmax(1).cpu().numpy()
        accs.append((pred == ql).mean())
    return float(np.mean(accs))


def main():
    df = length_match(load())
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    chance = 1 / nC
    y = df["label"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()
    a = db == DBS[0]; b = db == DBS[1]
    print(f"9 malattie, {len(df)} seq (length-matched), caso={chance:.3f}. One-shot & few-shot, 3 loss.\n")

    rows = []
    for name, cols in SETS3.items():
        X = df[cols].to_numpy(np.float32)
        for loss in ["euclideo", "coseno", "triplet"]:
            cos = (loss == "coseno")
            accs = {k: [] for k in SHOTS}
            for S, Q in [(a, b), (b, a)]:
                Xs, Xq = std_by(X[S], X[S], X[Q]); ys, yq = y[S], y[Q]
                enc = train_enc(loss, Xs, ys, nC)
                for k in SHOTS:
                    accs[k].append(eval_shot(enc, Xs, ys, Xq, yq, k, cos, nC))
            one = float(np.mean(accs[1])); few = float(np.mean(accs[5]))
            rows.append({"set": name, "loss": loss, "one_shot": round(one, 3), "one_xcaso": round(one / chance, 2),
                         "few_shot_K5": round(few, 3), "few_xcaso": round(few / chance, 2)})
            print(f"  {name:16s} {loss:9s}  one-shot={one:.3f} ({one/chance:.1f}x)   few-shot(K5)={few:.3f} ({few/chance:.1f}x)")

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "richer_kmers_fewshot.tsv"), sep="\t", index=False)
    print("\n" + res.to_string(index=False))
    print(f"\ncaso={chance:.3f}. Salvato in {RESULTS_DIR}/richer_kmers_fewshot.tsv")


if __name__ == "__main__":
    main()
