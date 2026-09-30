"""One-shot vs few-shot AL MASSIMO RIGORE, sul set ESTESO di malattie.

Il classificatore pieno richiede >=150 campioni per DB (-> 9 malattie). Il few-shot
ne richiede molti meno per il SUPPORTO: qui includiamo tutte le malattie con >=10
campioni nel database piu' debole (~12). Per ogni direzione (supporto da un DB,
query dall'altro) l'encoder e' addestrato LEAVE-ONE-STUDY-OUT (solo sul DB di
supporto, mai vede il DB di test). Si confrontano K=1 (one-shot) e K=5,10 (few-shot),
con IC 95% bootstrap sugli episodi. Query-set minimo QMIN per una stima sensata.

Riusa i descrittori gia' in cache (9 CROSS + 40 malattie); calcola i mancanti dal FASTA.
"""
import os
import numpy as np
import pandas as pd
import torch

from eccdna_utils import DESCRIPTOR_NAMES, read_fasta_stream, compute_sequence_descriptors
from train_multiclass import (train_prototypical, _class_prototypes, _proto_logits, SEED, DEVICE)
from models_pytorch import _to_tensor
from experiment_oneshot import _episode, COMPACT
from full_classifier import FASTA, RESULTS_DIR

DBS = ["eccDNABase", "CircleBaseV2"]
CACHES = ["data/processed/oneshot_desc_features.tsv", "data/processed/alldisease_oneshot_features.tsv"]
EXP_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/fewshot_expanded_features.tsv")
CAP = 600
MINW = 10     # min nel DB piu' debole per includere la malattia
QMIN = 30     # query set minimo per una stima sensata
SHOTS = [1, 5, 10]
Q = 8
N_EP = 800


def load_counts_and_meta():
    cnt, buf = {}, {}
    for ch in pd.read_csv(COMPACT, sep="\t", usecols=["id", "label", "is_disease", "source_db"],
                          chunksize=400000, low_memory=False):
        ch = ch[ch["is_disease"] == 1]
        for (l, s), g in ch.groupby(["label", "source_db"]):
            cnt[(l, s)] = cnt.get((l, s), 0) + len(g)
            b = buf.setdefault((l, s), [])
            if sum(len(x) for x in b) < CAP:
                b.append(g.head(CAP)[["id", "label", "source_db"]])
    per = {}
    for (l, s), n in cnt.items():
        per.setdefault(l, {})[s] = n
    keep = [l for l, d in per.items() if len(d) == 2 and min(d.values()) >= MINW]
    print(f"Malattie con >= {MINW} nel DB piu' debole: {len(keep)}")
    for l in sorted(keep, key=lambda l: -min(per[l].values())):
        print(f"    {l:<32} eccDNABase={per[l].get('eccDNABase',0):>8}  CircleBaseV2={per[l].get('CircleBaseV2',0):>8}")
    meta = pd.concat([pd.concat(buf[(l, s)]).head(CAP) for l in keep for s in DBS if (l, s) in buf],
                     ignore_index=True)
    meta["id"] = meta["id"].astype(str)
    return keep, per, meta


def features_for(meta):
    ft = None
    for c in CACHES:
        if os.path.exists(c):
            t = pd.read_csv(c, sep="\t", index_col=0); t.index = t.index.astype(str)
            ft = t if ft is None else pd.concat([ft, t])
    if ft is not None:
        ft = ft[~ft.index.duplicated()]
    have = set(ft.index) if ft is not None else set()
    ids = set(meta["id"]) - have
    if ids:
        feats = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
            if "N" not in seq and len(seq) >= 4:
                feats[sid] = compute_sequence_descriptors(seq)
        new = pd.DataFrame.from_dict(feats, orient="index")
        ft = new if ft is None else pd.concat([ft, new]); ft = ft[~ft.index.duplicated()]
        new.to_csv(EXP_CACHE, sep="\t")
    ftr = ft.reset_index().rename(columns={ft.reset_index().columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    return meta.merge(ftr[["id"] + DESCRIPTOR_NAMES], on="id", how="inner").dropna(subset=DESCRIPTOR_NAMES)


def std_by(Xfit, *arrays):
    m, s = Xfit.mean(0), Xfit.std(0); s = np.where(s == 0, 1, s)
    return [((A - m) / s).astype(np.float32) for A in arrays]


@torch.no_grad()
def eval_ci(enc, Xs, ys, Xq, yq, k, n_way, n_ep=N_EP, seed=SEED):
    rng = np.random.default_rng(seed); enc.eval()
    Es = enc(_to_tensor(Xs, DEVICE)); Eq = enc(_to_tensor(Xq, DEVICE))
    classes = np.array([c for c in np.unique(ys) if (ys == c).sum() >= 1 and (yq == c).sum() >= 1])
    n_way = min(n_way, len(classes)); accs = []
    for _ in range(n_ep):
        si, sl, qi, ql, _ = _episode(rng, ys, classes, n_way, k, Q, y2=yq)
        C = _class_prototypes(Es[si], torch.as_tensor(sl, device=DEVICE), n_way, cosine=True)
        pred = _proto_logits(Eq[qi], C, cosine=True).argmax(1).cpu().numpy()
        accs.append((pred == ql).mean())
    a = np.array(accs)
    return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5), n_way


def train_enc(Xtr, ytr, nC, k):
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(Xtr))
    Xtr, ytr = Xtr[perm], ytr[perm]; va = int(0.85 * len(Xtr))
    enc, _ = train_prototypical(Xtr[:va], ytr[:va], Xtr[va:], ytr[va:], nC, cosine=True, seed=SEED,
                               select_metric="balanced", n_support=k, n_query=5,
                               epochs=80, episodes=30, patience=10)
    return enc


def main():
    keep, per, meta = load_counts_and_meta()
    df = features_for(meta)
    rows = []
    for S, Qd in [("eccDNABase", "CircleBaseV2"), ("CircleBaseV2", "eccDNABase")]:
        # classi con >= max(SHOTS) nel DB di supporto e >= QMIN nel DB di query
        cls = [l for l in keep if per[l].get(S, 0) >= max(SHOTS) and per[l].get(Qd, 0) >= QMIN]
        sub = df[df["label"].isin(cls)].copy()
        c2i = {c: i for i, c in enumerate(sorted(cls))}
        sub["y"] = sub["label"].map(c2i); nC = len(cls); chance = 1 / nC
        a = sub[sub["source_db"] == S]; b = sub[sub["source_db"] == Qd]
        Xs_raw = a[DESCRIPTOR_NAMES].to_numpy(np.float32); ys = a["y"].to_numpy()
        Xq_raw = b[DESCRIPTOR_NAMES].to_numpy(np.float32); yq = b["y"].to_numpy()
        Xs, Xq = std_by(Xs_raw, Xs_raw, Xq_raw)   # standardizzo sul DB di supporto (no leak)
        print(f"\n=== supporto={S} -> query={Qd} | {nC} malattie | caso={chance:.3f} ===")
        for k in SHOTS:
            enc = train_enc(Xs, ys, nC, k)
            m, lo, hi, nw = eval_ci(enc, Xs, ys, Xq, yq, k, nC)
            print(f"  K={k:<2d}  acc={m:.3f}  IC95=[{lo:.3f},{hi:.3f}]  ({m/chance:.1f}x il caso)")
            rows.append({"direzione": f"{S[:6]}->{Qd[:6]}", "n_classi": nC, "shot": k,
                         "regime": "one-shot" if k == 1 else f"{k}-shot",
                         "acc": round(m, 3), "IC95": f"[{lo:.3f},{hi:.3f}]",
                         "caso": round(chance, 3), "x_caso": round(m / chance, 2)})
    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "fewshot_expanded.tsv"), sep="\t", index=False)
    print("\n" + res.to_string(index=False))
    print(f"\nSalvato in {RESULTS_DIR}/fewshot_expanded.tsv")


if __name__ == "__main__":
    main()
