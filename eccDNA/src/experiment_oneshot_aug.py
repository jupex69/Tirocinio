"""Aumento del supporto per il one-shot (miglior config: coseno + 1-shot training,
10 descrittori). Poiche' l'eccDNA e' circolare, si aumenta l'unico esempio di
supporto con:
 - ROTAZIONI CIRCOLARI (offset a 1/4, 1/2, 3/4) -> descrittori ricalcolati e mediati;
 - + REVERSE-COMPLEMENT (strand arbitrario) -> variante aggiuntiva.
Il prototipo del supporto usa la rappresentazione aumentata; i query restano
grezzi. Si confronta il one-shot (within e cross-studio) con supporto grezzo vs
aumentato. Attesa: sui 10 descrittori la rotazione e' ~invariante (poco effetto);
il reverse-complement ribalta gli skew (potenziale danno). Lo si misura.
"""

import os
import numpy as np
import pandas as pd
import torch

from eccdna_utils import DESCRIPTOR_NAMES, read_fasta_stream, compute_sequence_descriptors
from train_multiclass import standardize, train_prototypical, _class_prototypes, _proto_logits, SEED, DEVICE
from models_pytorch import _to_tensor
from full_classifier import FASTA
from experiment_oneshot import load_data, RESULTS_DIR

AUGCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/oneshot_aug_features.tsv")
COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s):
    return s.translate(COMP)[::-1]


def aug_descriptors(seq, use_rc):
    L = len(seq)
    variants = [seq] + [seq[k:] + seq[:k] for k in (L // 4, L // 2, 3 * L // 4)]
    if use_rc:
        variants.append(revcomp(seq))
    vecs = []
    for v in variants:
        if "N" not in v and len(v) >= 4:
            d = compute_sequence_descriptors(v)
            vecs.append([d[c] for c in DESCRIPTOR_NAMES])
    return np.mean(vecs, axis=0) if vecs else None


@torch.no_grad()
def oneshot_sq(enc, Xs, ys, Xq, yq, cosine, n_way, k=1, q=10, n_ep=500, seed=SEED):
    """support da (Xs,ys), query da (Xq,yq). Xs/Xq possono differire (grezzo vs
    aumentato) e provenire da studi diversi (cross)."""
    rng = np.random.default_rng(seed)
    Es = enc(_to_tensor(Xs, DEVICE)) if enc else _to_tensor(Xs, DEVICE)
    Eq = enc(_to_tensor(Xq, DEVICE)) if enc else _to_tensor(Xq, DEVICE)
    classes = np.array([c for c in np.unique(ys) if (ys == c).sum() >= 1 and (yq == c).sum() >= 1])
    n_way = min(n_way, len(classes)); accs = []
    for _ in range(n_ep):
        chosen = rng.choice(classes, n_way, replace=False)
        si, sl, qi, ql = [], [], [], []
        for j, c in enumerate(chosen):
            ps = np.where(ys == c)[0]; pq = np.where(yq == c)[0]
            si += list(rng.choice(ps, k, replace=len(ps) < k)); sl += [j] * k
            qi += list(rng.choice(pq, q, replace=len(pq) < q)); ql += [j] * q
        C = _class_prototypes(Es[si], torch.as_tensor(sl, device=DEVICE), n_way, cosine)
        pred = _proto_logits(Eq[np.array(qi)], C, cosine).argmax(1).cpu().numpy()
        accs.append((pred == np.array(ql)).mean())
    return float(np.mean(accs))


def main():
    df = load_data()  # id, label, source_db, 10 descrittori grezzi
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    df = df.reset_index(drop=True)

    # calcola descrittori aumentati (rotazioni; rotazioni+rc) per gli id del pool
    if os.path.exists(AUGCACHE):
        aug = pd.read_csv(AUGCACHE, sep="\t", index_col=0); aug.index = aug.index.astype(str)
    else:
        rows = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=set(df["id"])):
            if "N" not in seq and len(seq) >= 4:
                a = aug_descriptors(seq, use_rc=False); arc = aug_descriptors(seq, use_rc=True)
                if a is not None:
                    rows[sid] = list(a) + list(arc)
        cols = [f"rot_{c}" for c in DESCRIPTOR_NAMES] + [f"rotrc_{c}" for c in DESCRIPTOR_NAMES]
        aug = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
        aug.to_csv(AUGCACHE, sep="\t")
    aug = aug.reset_index().rename(columns={aug.reset_index().columns[0]: "id"}); aug["id"] = aug["id"].astype(str)
    df = df.merge(aug, on="id", how="inner")

    y = df["label"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()
    Xraw = df[DESCRIPTOR_NAMES].to_numpy(np.float32)
    Xrot = df[[f"rot_{c}" for c in DESCRIPTOR_NAMES]].to_numpy(np.float32)
    Xrotrc = df[[f"rotrc_{c}" for c in DESCRIPTOR_NAMES]].to_numpy(np.float32)

    # standardizza tutto con le statistiche del train grezzo
    rng = np.random.default_rng(SEED); idx = rng.permutation(len(df)); cut = int(0.7 * len(df))
    tr, te = idx[:cut], idx[cut:]
    m, s = Xraw[tr].mean(0), Xraw[tr].std(0); s[s == 0] = 1.0
    Zraw, Zrot, Zrotrc = (Xraw - m) / s, (Xrot - m) / s, (Xrotrc - m) / s

    # miglior encoder: coseno, 1-shot training, sul train grezzo
    va = int(0.85 * len(tr))
    enc, _ = train_prototypical(Zraw[tr][:va], y[tr][:va], Zraw[tr][va:], y[tr][va:], nC, cosine=True, seed=SEED,
                               select_metric="balanced", n_support=1, n_query=5)
    cos = True; n_way = min(5, nC)

    print("=== One-shot (coseno) con supporto GREZZO vs AUMENTATO (10 descrittori) ===")
    rows = []
    for nome, Zs in [("grezzo", Zraw), ("rotazioni", Zrot), ("rotazioni+revcomp", Zrotrc)]:
        # within: support (aug) e query (grezzo) dal test
        w = oneshot_sq(enc, Zs[te], y[te], Zraw[te], y[te], cos, n_way)
        # cross: support (aug) da eccDNABase, query (grezzo) da CircleBaseV2
        a = db == "eccDNABase"; b = db == "CircleBaseV2"
        c = oneshot_sq(enc, Zs[a], y[a], Zraw[b], y[b], cos, n_way)
        rows.append({"supporto": nome, "within_1shot": round(w, 3), "cross_1shot": round(c, 3)})
        print(f"  supporto {nome:18s} within={w:.3f}  cross={c:.3f}")

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "oneshot_augmentation.tsv"), sep="\t", index=False)
    print(f"\ncaso {n_way}-way = {1/n_way:.3f}")
    # verifica invarianza rotazione
    diff = np.abs(Zraw - Zrot).mean()
    print(f"Differenza media |grezzo - rotazioni| (standardizzato): {diff:.4f}  (~0 = rotazione invariante)")
    print(f"Salvato in {RESULTS_DIR}/oneshot_augmentation.tsv")


if __name__ == "__main__":
    main()
