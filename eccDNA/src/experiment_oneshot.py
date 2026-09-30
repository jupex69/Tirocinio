"""Massimizzare il ONE-SHOT sui 10 descrittori, con le 3 loss siamesi.

Leve applicate: (1) 1-shot episodic training (addestrare come si testa);
(2) le 3 loss -- prototipico euclideo, coseno, triplet; (3) rettifica
TRANSDUTTIVA dei prototipi (raffina il prototipo da 1 esempio usando i query non
etichettati). Baseline: one-shot sullo spazio grezzo dei 10 descrittori.

Valutazione one-shot in due regimi:
  - WITHIN-STUDY (support+query dalla stessa distribuzione) -> ottimistico;
  - CROSS-STUDIO (support da un DB, query dall'altro)       -> onesto.

Feature: SOLO i 10 descrittori (DESCRIPTOR_NAMES). Classi: le 9 cross-testabili.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from eccdna_utils import DESCRIPTOR_NAMES, read_fasta_stream, compute_sequence_descriptors
from train_multiclass import (
    Encoder, standardize, train_prototypical, _class_prototypes, _proto_logits, SEED, DEVICE,
)
from experiment_losses_fewshot import train_metric_encoder
from models_pytorch import _to_tensor
from full_classifier import FASTA

COMPACT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/eccdna_compacted_metadata.tsv")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/oneshot_desc_features.tsv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CROSS = ["Stomach Cancer", "Ovarian Cancer", "Prostate Cancer", "Colorectal Cancer",
         "Hypopharynx Cancer", "Primary Pulmonary Hypertension", "Glioblastoma",
         "Dilated Cardiomyopathy", "Breast Cancer"]
DBS = ["eccDNABase", "CircleBaseV2"]
CAP = 600


def load_data():
    want = {(d, s): [] for d in CROSS for s in DBS}
    for ch in pd.read_csv(COMPACT, sep="\t", usecols=["id", "label", "source_db"], chunksize=400000, low_memory=False):
        ch = ch[ch["label"].isin(CROSS) & ch["source_db"].isin(DBS)]
        for (d, s), g in ch.groupby(["label", "source_db"]):
            buf = want[(d, s)]
            if sum(len(x) for x in buf) < CAP:
                buf.append(g.head(CAP)[["id", "label", "source_db"]])
    meta = pd.concat([pd.concat(v).head(CAP) for v in want.values() if v], ignore_index=True)
    meta["id"] = meta["id"].astype(str)
    ids = set(meta["id"])
    # 10 descrittori calcolati dal FASTA (cache incrementale) -> copre tutte le classi
    ft = None
    if os.path.exists(CACHE):
        ft = pd.read_csv(CACHE, sep="\t", index_col=0); ft.index = ft.index.astype(str)
        ids = ids - set(ft.index)
    if ids:
        feats = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
            if "N" not in seq and len(seq) >= 4:
                feats[sid] = compute_sequence_descriptors(seq)
        new = pd.DataFrame.from_dict(feats, orient="index")
        ft = new if ft is None else pd.concat([ft, new])
        ft = ft[~ft.index.duplicated()]; ft.to_csv(CACHE, sep="\t")
    ftr = ft.reset_index(); ftr = ftr.rename(columns={ftr.columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    return meta.merge(ftr[["id"] + DESCRIPTOR_NAMES], on="id", how="inner").dropna(subset=DESCRIPTOR_NAMES)


@torch.no_grad()
def _emb(enc, X):
    if enc is None:
        return _to_tensor(X, DEVICE)
    enc.eval()
    return enc(_to_tensor(X, DEVICE))


def _rectify(C, emb_q, cosine, iters=1):
    for _ in range(iters):
        w = torch.softmax(_proto_logits(emb_q, C, cosine), dim=1)     # (nq, nway)
        C = C + w.t() @ emb_q
        C = C / (1 + w.sum(0).unsqueeze(1))
        if cosine:
            C = F.normalize(C, dim=-1)
    return C


def _episode(rng, y, classes, n_way, k, q, y2=None, X2emb=None):
    """support (k per classe) e query (q per classe). Se y2/X2emb dati, i query
    vengono da un ALTRO insieme (cross-studio)."""
    chosen = rng.choice(classes, n_way, replace=False)
    s_idx, s_lab, q_idx, q_lab = [], [], [], []
    for j, c in enumerate(chosen):
        ps = np.where(y == c)[0]
        s_idx += list(rng.choice(ps, k, replace=len(ps) < k)); s_lab += [j] * k
        pool_q = np.where((y2 if y2 is not None else y) == c)[0]
        q_idx += list(rng.choice(pool_q, q, replace=len(pool_q) < q)); q_lab += [j] * q
    return np.array(s_idx), np.array(s_lab), np.array(q_idx), np.array(q_lab), chosen


@torch.no_grad()
def oneshot(enc, Xs, ys, cosine, n_way, k=1, q=10, n_ep=400, transductive=False, seed=SEED, Xq=None, yq=None):
    rng = np.random.default_rng(seed)
    Es = _emb(enc, Xs); Eq = _emb(enc, Xq) if Xq is not None else Es
    classes = np.array([c for c in np.unique(ys) if (ys == c).sum() >= 1 and (yq if yq is not None else ys).tolist().count(c) >= 1])
    n_way = min(n_way, len(classes)); accs = []
    for _ in range(n_ep):
        si, sl, qi, ql, _ = _episode(rng, ys, classes, n_way, k, q, y2=yq, X2emb=Eq)
        C = _class_prototypes(Es[si], torch.as_tensor(sl, device=DEVICE), n_way, cosine)
        if transductive:
            C = _rectify(C, Eq[qi], cosine)
        pred = _proto_logits(Eq[qi], C, cosine).argmax(1).cpu().numpy()
        accs.append((pred == ql).mean())
    return float(np.mean(accs))


def main(quick=False):
    ep = dict(epochs=8, episodes=8, patience=4) if quick else dict(epochs=120, episodes=40, patience=15)
    n_ep = 80 if quick else 500
    df = load_data()
    classes = sorted(df["label"].unique()); nC = len(classes); c2i = {c: i for i, c in enumerate(classes)}
    print(f"Classi: {nC}  campioni: {len(df)}  (10 descrittori)\n")
    X = df[DESCRIPTOR_NAMES].to_numpy(np.float32); y = df["label"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()

    # split train/test (per l'encoder e per la valutazione within)
    rng = np.random.default_rng(SEED); idx = rng.permutation(len(df)); cut = int(0.7 * len(df))
    tr, te = idx[:cut], idx[cut:]
    Xtr_s, Xte_s = standardize(X[tr], X[te])
    ytr, yte = y[tr], y[te]
    # per il cross-studio: standardizzo per DB (fit su A, applico a B dentro le funzioni sarebbe ideale; qui uso stat globali del train)
    Xall_s = (X - X[tr].mean(0)) / np.where(X[tr].std(0) == 0, 1, X[tr].std(0))

    n_way = nC  # 9-way: tutte le malattie (caso 1/9)
    rows = []

    def train_enc(kind):
        if kind == "raw":
            return None, (kind == "coseno")
        cos = kind == "coseno"
        va = int(0.85 * len(Xtr_s))
        if kind in ("euclideo", "coseno"):
            enc, _ = train_prototypical(Xtr_s[:va], ytr[:va], Xtr_s[va:], ytr[va:], nC, cosine=cos, seed=SEED,
                                        select_metric="balanced", n_support=1, n_query=5,
                                        epochs=ep["epochs"], episodes=ep["episodes"], patience=ep["patience"])
        else:  # triplet
            enc, _ = train_metric_encoder(Xtr_s[:va], ytr[:va], Xtr_s[va:], ytr[va:], nC, "triplet", margin=0.3,
                                          epochs=ep["epochs"], batches=ep["episodes"], patience=ep["patience"])
        return enc, cos

    print("=== ONE-SHOT (10 descrittori) — 3 loss, within vs cross-studio ===")
    for kind in ["raw", "euclideo", "coseno", "triplet"]:
        enc, cos = train_enc(kind)
        cos_eval = cos if kind != "triplet" else False
        # within-study: support+query dal test
        w_plain = oneshot(enc, Xte_s, yte, cos_eval, n_way, n_ep=n_ep)
        w_trans = oneshot(enc, Xte_s, yte, cos_eval, n_way, n_ep=n_ep, transductive=True)
        # cross-studio: support da eccDNABase, query da CircleBaseV2 (su tutti i dati, standardizzati)
        a = db == "eccDNABase"; b = db == "CircleBaseV2"
        c_plain = oneshot(enc, Xall_s[a], y[a], cos_eval, n_way, n_ep=n_ep, Xq=Xall_s[b], yq=y[b])
        c_trans = oneshot(enc, Xall_s[a], y[a], cos_eval, n_way, n_ep=n_ep, Xq=Xall_s[b], yq=y[b], transductive=True)
        rows.append({"loss": kind, "within_1shot": round(w_plain, 3), "within_+transd": round(w_trans, 3),
                     "cross_1shot": round(c_plain, 3), "cross_+transd": round(c_trans, 3)})
        print(f"  {kind:9s} within={w_plain:.3f} (+transd {w_trans:.3f})   cross={c_plain:.3f} (+transd {c_trans:.3f})")

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "oneshot_10desc.tsv"), sep="\t", index=False)
    print(f"\ncaso {n_way}-way = {1/n_way:.3f}")
    print(f"Salvato in {RESULTS_DIR}/oneshot_10desc.tsv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
