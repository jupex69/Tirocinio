"""Analisi few-shot APPROFONDITA sulle 9 malattie, al massimo rigore.

Curva degli shot completa K = 1, 3, 5, 9, 15 (include one-shot, K=N=9 -- il punto
suggerito dal tutor -- e oltre), valutazione a 9-WAY PIENA (ogni malattia e' sempre
candidata: il "doppio controllo" -- agganciare la giusta E respingere tutte le
altre), encoder LEAVE-ONE-STUDY-OUT, cross-studio nelle due direzioni, IC 95%
bootstrap sugli episodi. In piu': recall per malattia a K=9.
"""
import os
import numpy as np
import pandas as pd
import torch

from eccdna_utils import DESCRIPTOR_NAMES
from train_multiclass import _class_prototypes, _proto_logits, SEED, DEVICE
from models_pytorch import _to_tensor
from experiment_oneshot import load_data, _episode, RESULTS_DIR
from experiment_fewshot_rigorous import std_by, eval_ci, train_enc_on

DBS = ["eccDNABase", "CircleBaseV2"]
SHOTS = [1, 3, 5, 9, 15]


@torch.no_grad()
def perclass_recall(enc, Xs, ys, Xq, yq, k, nC, n_draws=300, seed=SEED):
    """Recall per classe a 9-way piena: prototipi da k support (studio S), query
    da studio Q, ogni malattia sempre candidata."""
    rng = np.random.default_rng(seed); enc.eval()
    Es = enc(_to_tensor(Xs, DEVICE)); Eq = enc(_to_tensor(Xq, DEVICE))
    rec = np.full((n_draws, nC), np.nan)
    for d in range(n_draws):
        s_idx, s_lab = [], []
        for c in range(nC):
            pool = np.where(ys == c)[0]
            if len(pool) == 0:
                continue
            s_idx += list(rng.choice(pool, k, replace=len(pool) < k)); s_lab += [c] * k
        C = _class_prototypes(Es[s_idx], torch.as_tensor(s_lab, device=DEVICE), nC, cosine=True)
        pred = _proto_logits(Eq, C, cosine=True).argmax(1).cpu().numpy()
        for c in range(nC):
            m = yq == c
            if m.any():
                rec[d, c] = (pred[m] == c).mean()
    return np.nanmean(rec, axis=0)


def main():
    df = load_data()
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    X = df[DESCRIPTOR_NAMES].to_numpy(np.float32); y = df["label"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()
    chance = 1 / nC
    print(f"9 malattie, caso={chance:.3f}. Curva degli shot a 9-way piena, LOSO, cross-studio.\n")
    masks = {d: (db == d) for d in DBS}

    rows, perclass_rows = [], None
    for k in SHOTS:
        dir_res = []
        for S, Qd in [(DBS[0], DBS[1]), (DBS[1], DBS[0])]:
            a, b = masks[S], masks[Qd]
            Xs, Xq = std_by(X[a], X[a], X[b]); ys, yq = y[a], y[b]
            enc = train_enc_on(Xs, ys, nC, k)
            m, lo, hi, nw = eval_ci(enc, Xs, ys, Xq, yq, k)
            dir_res.append((m, lo, hi))
            print(f"  K={k:<2d}  {S[:6]}->{Qd[:6]}  acc={m:.3f}  IC95=[{lo:.3f},{hi:.3f}]  ({m/chance:.1f}x)")
            if k == 9:   # recall per malattia al punto K=N suggerito dal tutor
                rc = perclass_recall(enc, Xs, ys, Xq, yq, k, nC)
                col = f"recall_{S[:6]}2{Qd[:6]}"
                pr = pd.DataFrame({"malattia": classes, col: np.round(rc, 3)})
                perclass_rows = pr if perclass_rows is None else perclass_rows.merge(pr, on="malattia")
        mean_dir = float(np.mean([r[0] for r in dir_res]))
        rows.append({"shot_K": k, "regime": "one-shot" if k == 1 else f"{k}-shot",
                     "cross_media": round(mean_dir, 3),
                     "dir_A2B": round(dir_res[0][0], 3), "IC_A2B": f"[{dir_res[0][1]:.3f},{dir_res[0][2]:.3f}]",
                     "dir_B2A": round(dir_res[1][0], 3), "IC_B2A": f"[{dir_res[1][1]:.3f},{dir_res[1][2]:.3f}]",
                     "x_caso": round(mean_dir / chance, 2)})
        print(f"  --> K={k}: media cross-studio = {mean_dir:.3f} ({mean_dir/chance:.1f}x)\n")

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "fewshot_deep.tsv"), sep="\t", index=False)
    print(res.to_string(index=False))
    if perclass_rows is not None:
        perclass_rows["media"] = perclass_rows.iloc[:, 1:].mean(axis=1).round(3)
        perclass_rows = perclass_rows.sort_values("media", ascending=False)
        perclass_rows.to_csv(os.path.join(RESULTS_DIR, "fewshot_deep_perclass_K9.tsv"), sep="\t", index=False)
        print(f"\n--- Recall per malattia a K=9 (9-way piena, caso {chance:.3f}) ---")
        print(perclass_rows.to_string(index=False))
    print(f"\ncaso = {chance:.3f}. Salvato in {RESULTS_DIR}/fewshot_deep.tsv")


if __name__ == "__main__":
    main()
