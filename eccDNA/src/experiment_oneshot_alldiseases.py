"""One-shot N-way su TUTTE le malattie: confrontare una query con il prototipo di
OGNI malattia e assegnarla alla piu' vicina (inferenza prototipica estesa a tutte
le classi, non solo alle 9 cross-testabili).

Regime: WITHIN-STUDY (support+query dallo stesso pool). E' l'unico possibile su
tutte le malattie, perche' 46 su 55 esistono in un solo studio -> la validazione
cross-studio (onesta) resta possibile solo sulle 9 multi-studio (vedi
experiment_oneshot.py). Questo numero va quindi letto come OTTIMISTICO/confuso dal
batch: serve a mostrare quanto la sola composizione separi APPARENTEMENTE le
malattie quando le si confronta tutte insieme.

Feature: i 10 descrittori. Encoder: coseno (il migliore), addestrato a 1-shot.
Caso = 1/(numero di malattie incluse).
"""
import os
import numpy as np
import pandas as pd
import torch

from eccdna_utils import DESCRIPTOR_NAMES, read_fasta_stream, compute_sequence_descriptors
from train_multiclass import (Encoder, standardize, train_prototypical,
                              _class_prototypes, _proto_logits, SEED, DEVICE)
from models_pytorch import _to_tensor
from full_classifier import FASTA, RESULTS_DIR
from experiment_oneshot import oneshot, COMPACT

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/alldisease_oneshot_features.tsv")
CAP = 300      # campioni per malattia
MIN_N = 50     # una malattia entra solo se ha >= MIN_N campioni


def load_all():
    counts, buf = {}, {}
    for ch in pd.read_csv(COMPACT, sep="\t", usecols=["id", "label", "is_disease", "source_db"],
                          chunksize=400000, low_memory=False):
        ch = ch[ch["is_disease"] == 1]
        for d, g in ch.groupby("label"):
            counts[d] = counts.get(d, 0) + len(g)
            b = buf.setdefault(d, [])
            if sum(len(x) for x in b) < CAP:
                b.append(g.head(CAP)[["id", "label", "source_db"]])
    keep = sorted([d for d in counts if counts[d] >= MIN_N])
    print(f"Malattie con >= {MIN_N} campioni: {len(keep)} (su {len(counts)} totali)")
    meta = pd.concat([pd.concat(buf[d]).head(CAP) for d in keep], ignore_index=True)
    meta["id"] = meta["id"].astype(str)
    ids = set(meta["id"]); ft = None
    if os.path.exists(CACHE):
        ft = pd.read_csv(CACHE, sep="\t", index_col=0); ft.index = ft.index.astype(str); ids = ids - set(ft.index)
    if ids:
        feats = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
            if "N" not in seq and len(seq) >= 4:
                feats[sid] = compute_sequence_descriptors(seq)
        new = pd.DataFrame.from_dict(feats, orient="index")
        ft = new if ft is None else pd.concat([ft, new]); ft = ft[~ft.index.duplicated()]; ft.to_csv(CACHE, sep="\t")
    ftr = ft.reset_index().rename(columns={ft.reset_index().columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    return meta.merge(ftr[["id"] + DESCRIPTOR_NAMES], on="id", how="inner").dropna(subset=DESCRIPTOR_NAMES)


@torch.no_grad()
def perclass_recall(enc, Xte, yte, nC, n_draws=200, seed=SEED):
    rng = np.random.default_rng(seed)
    E = enc(_to_tensor(Xte, DEVICE))
    rec = np.full((n_draws, nC), np.nan)
    for d in range(n_draws):
        s_idx, s_lab = [], []
        for c in range(nC):
            pool = np.where(yte == c)[0]
            if len(pool) < 2:      # serve 1 per supporto + almeno 1 query
                continue
            s_idx.append(rng.choice(pool)); s_lab.append(c)
        C = _class_prototypes(E[s_idx], torch.as_tensor(s_lab, device=DEVICE), nC, cosine=True)
        pred = _proto_logits(E, C, cosine=True).argmax(1).cpu().numpy()
        for c in range(nC):
            m = yte == c
            if m.any():
                rec[d, c] = (pred[m] == c).mean()
    return np.nanmean(rec, axis=0)


def main():
    df = load_all()
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    chance = 1 / nC
    X = df[DESCRIPTOR_NAMES].to_numpy(np.float32); y = df["label"].map(c2i).to_numpy()
    print(f"\nOne-shot N-way su TUTTE le malattie: {nC} classi, {len(df)} sequenze, caso={chance:.4f}\n")

    rng = np.random.default_rng(SEED); idx = rng.permutation(len(df)); cut = int(0.7 * len(df)); tr, te = idx[:cut], idx[cut:]
    Xtr_s, Xte_s = standardize(X[tr], X[te]); ytr, yte = y[tr], y[te]
    va = int(0.85 * len(Xtr_s))
    enc, _ = train_prototypical(Xtr_s[:va], ytr[:va], Xtr_s[va:], ytr[va:], nC, cosine=True, seed=SEED,
                                select_metric="balanced", n_support=1, n_query=5)

    acc = oneshot(enc, Xte_s, yte, True, nC, k=1, n_ep=500)
    print(f"Accuratezza one-shot {nC}-way (within-study) = {acc:.3f}  ({acc/chance:.1f}x il caso)\n")

    rec = perclass_recall(enc, Xte_s, yte, nC)
    per = pd.DataFrame({"malattia": classes, "recall_within": np.round(rec, 3)}).sort_values("recall_within", ascending=False)
    n_above = int((per["recall_within"] > chance * 2).sum())
    print(f"Malattie con recall > 2x il caso ({chance*2:.3f}): {n_above}/{nC}\n")
    print(per.head(20).to_string(index=False))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    per.to_csv(os.path.join(RESULTS_DIR, "oneshot_alldiseases_perclass.tsv"), sep="\t", index=False)
    pd.DataFrame([{"n_classi": nC, "accuratezza": round(acc, 3), "caso": round(chance, 4),
                   "x_caso": round(acc / chance, 2), "n_sopra_2xcaso": n_above}]).to_csv(
        os.path.join(RESULTS_DIR, "oneshot_alldiseases.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/oneshot_alldiseases.tsv")


if __name__ == "__main__":
    main()
