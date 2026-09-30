"""Diagnostico: per Ovaio/Prostata/Colon-retto, piu' esempi di supporto migliorano
il cross-studio? Fisso l'encoder (LOSO, addestrato a 5-shot) e faccio crescere gli
esempi di supporto usati ALL'INFERENZA per costruire il prototipo (K=1..200),
misurando il recall per malattia (9-way piena, cross-studio, media delle 2 direzioni).

Se il recall di queste classi resta piatto mentre K cresce, il limite non e' la
numerosita' ma il trasferimento (cross-metodo) e la sovrapposizione biologica.
Riusa la cache (CAP=600), nessun ricalcolo dal FASTA.
"""
import os
import numpy as np
import pandas as pd
import torch

from eccdna_utils import DESCRIPTOR_NAMES
from train_multiclass import _class_prototypes, _proto_logits, SEED, DEVICE
from models_pytorch import _to_tensor
from experiment_oneshot import load_data, RESULTS_DIR
from experiment_fewshot_rigorous import std_by, train_enc_on

DBS = ["eccDNABase", "CircleBaseV2"]
INFER_K = [1, 5, 20, 50, 100, 200]
FOCUS = ["Ovarian Cancer", "Prostate Cancer", "Colorectal Cancer"]


@torch.no_grad()
def perclass_recall(enc, Xs, ys, Xq, yq, k, nC, n_draws=200, seed=SEED):
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
    print(f"9 malattie, caso={chance:.3f}. Recall per malattia al crescere degli esempi di supporto (inferenza).\n")
    masks = {d: (db == d) for d in DBS}

    # per ogni K, media del recall sulle due direzioni
    acc = {k: [] for k in INFER_K}
    for S, Qd in [(DBS[0], DBS[1]), (DBS[1], DBS[0])]:
        a, b = masks[S], masks[Qd]
        Xs, Xq = std_by(X[a], X[a], X[b]); ys, yq = y[a], y[b]
        enc = train_enc_on(Xs, ys, nC, 5)          # encoder fisso (5-shot), LOSO
        for k in INFER_K:
            acc[k].append(perclass_recall(enc, Xs, ys, Xq, yq, k, nC))
    rec_by_k = {k: np.nanmean(acc[k], axis=0) for k in INFER_K}

    tab = pd.DataFrame({"malattia": classes})
    for k in INFER_K:
        tab[f"K={k}"] = np.round(rec_by_k[k], 3)
    tab = tab.set_index("malattia")
    print("--- Recall per malattia vs esempi di supporto (media 2 direzioni, caso {:.3f}) ---".format(chance))
    print(tab.to_string())
    print("\n>>> FOCUS (Ovaio / Prostata / Colon-retto):")
    print(tab.loc[FOCUS].to_string())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tab.to_csv(os.path.join(RESULTS_DIR, "moreexamples_perclass.tsv"), sep="\t")
    print(f"\nSalvato in {RESULTS_DIR}/moreexamples_perclass.tsv")


if __name__ == "__main__":
    main()
