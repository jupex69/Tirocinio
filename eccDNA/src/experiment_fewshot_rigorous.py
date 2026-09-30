"""Confronto RIGOROSO one-shot vs few-shot sulle 9 malattie, al massimo rigore.

Massimo rigore = cross-studio con LEAVE-ONE-STUDY-OUT sull'encoder:
  - l'encoder e' addestrato su UN SOLO database (mai vede lo studio di test);
  - i prototipi si costruiscono da K esempi del database di SUPPORTO;
  - le query vengono dall'ALTRO database;
  - train-as-you-test: per il few-shot l'encoder e' addestrato con lo stesso K;
  - si mediano le due direzioni (A->B e B->A);
  - intervallo di confidenza bootstrap sugli episodi.

Confronto: K=1 (one-shot) vs K=5,10 (few-shot). Encoder coseno (il migliore).
Feature: i 10 descrittori. Caso = 1/9 = 0.111.
"""
import os
import numpy as np
import pandas as pd
import torch

from eccdna_utils import DESCRIPTOR_NAMES
from train_multiclass import (train_prototypical, _class_prototypes, _proto_logits, SEED, DEVICE)
from models_pytorch import _to_tensor
from experiment_oneshot import load_data, _episode, RESULTS_DIR

DBS = ["eccDNABase", "CircleBaseV2"]
SHOTS = [1, 5, 10]
N_WAY = 9
Q = 10
N_EP = 800


def std_by(Xfit, *arrays):
    m, s = Xfit.mean(0), Xfit.std(0); s = np.where(s == 0, 1, s)
    return [((A - m) / s).astype(np.float32) for A in arrays]


@torch.no_grad()
def eval_ci(enc, Xs, ys, Xq, yq, k, n_ep=N_EP, seed=SEED):
    """Per-episodio: prototipi da K support (studio S), query da studio Q. Ritorna
    media e IC 95% (percentili) sugli episodi."""
    rng = np.random.default_rng(seed)
    enc.eval()
    Es = enc(_to_tensor(Xs, DEVICE)); Eq = enc(_to_tensor(Xq, DEVICE))
    classes = np.array([c for c in np.unique(ys) if (ys == c).sum() >= 1 and (yq == c).sum() >= 1])
    n_way = min(N_WAY, len(classes)); accs = []
    for _ in range(n_ep):
        si, sl, qi, ql, _ = _episode(rng, ys, classes, n_way, k, Q, y2=yq)
        C = _class_prototypes(Es[si], torch.as_tensor(sl, device=DEVICE), n_way, cosine=True)
        pred = _proto_logits(Eq[qi], C, cosine=True).argmax(1).cpu().numpy()
        accs.append((pred == ql).mean())
    a = np.array(accs)
    return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5), n_way


def train_enc_on(Xtr, ytr, nC, k):
    """Encoder coseno addestrato SOLO sullo studio di supporto, a K-shot."""
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Xtr))          # shuffle: i dati arrivano raggruppati per classe
    Xtr, ytr = Xtr[perm], ytr[perm]
    va = int(0.85 * len(Xtr))
    enc, _ = train_prototypical(Xtr[:va], ytr[:va], Xtr[va:], ytr[va:], nC, cosine=True, seed=SEED,
                                select_metric="balanced", n_support=k, n_query=5,
                                epochs=80, episodes=30, patience=10)
    return enc


def main():
    df = load_data()
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    X = df[DESCRIPTOR_NAMES].to_numpy(np.float32); y = df["label"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()
    chance = 1 / nC
    print(f"9 malattie, {len(df)} sequenze, caso={chance:.3f}. Massimo rigore: LOSO sull'encoder.\n")

    masks = {d: (db == d) for d in DBS}
    rows = []
    for k in SHOTS:
        dir_res = []
        for S, Qd in [(DBS[0], DBS[1]), (DBS[1], DBS[0])]:
            a, b = masks[S], masks[Qd]
            # standardizzo sullo studio di supporto S (niente info dallo studio di test)
            Xs, Xq = std_by(X[a], X[a], X[b])
            ys, yq = y[a], y[b]
            # encoder addestrato SOLO su S, a K-shot
            enc = train_enc_on(Xs, ys, nC, k)
            m, lo, hi, nw = eval_ci(enc, Xs, ys, Xq, yq, k)
            dir_res.append((m, lo, hi))
            print(f"  K={k:<2d}  {S[:6]}->{Qd[:6]}  acc={m:.3f}  IC95=[{lo:.3f},{hi:.3f}]  ({m/chance:.1f}x)")
        mean_dir = float(np.mean([r[0] for r in dir_res]))
        rows.append({"shot": k, "regime": "one-shot" if k == 1 else f"{k}-shot",
                     "cross_media": round(mean_dir, 3),
                     "dir_A2B": round(dir_res[0][0], 3), "IC_A2B": f"[{dir_res[0][1]:.3f},{dir_res[0][2]:.3f}]",
                     "dir_B2A": round(dir_res[1][0], 3), "IC_B2A": f"[{dir_res[1][1]:.3f},{dir_res[1][2]:.3f}]",
                     "x_caso": round(mean_dir / chance, 2)})
        print(f"  --> K={k}: media cross-studio = {mean_dir:.3f} ({mean_dir/chance:.1f}x il caso)\n")

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "fewshot_rigorous.tsv"), sep="\t", index=False)
    print(res.to_string(index=False))
    print(f"\ncaso = {chance:.3f}. Salvato in {RESULTS_DIR}/fewshot_rigorous.tsv")


if __name__ == "__main__":
    main()
