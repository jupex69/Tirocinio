"""Analisi accurata dell'IDENTIFICAZIONE DELLE MALATTIE (9 malattie) + ricerca di
descrittori nuovi utili dalla sequenza.

Leva testata: l'ORDINE dei k-meri. Oggi: 10 descrittori + spettro 3-mer (64). Qui si
aggiungono 4-mer (256) e 5-mer (1024) -> motivi piu' lunghi/specifici. Confronto
ONESTO su 4 set di feature, within-study E cross-studio, con length-matching (i
k-meri di ordine alto sono piu' legati alla lunghezza: va controllato). Piu':
importanza delle feature per capire se i motivi lunghi contano davvero.

Modello: RandomForest (il migliore su tabellari). Metriche: accuratezza, macro-F1.
"""
import os
from itertools import product
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from eccdna_utils import DESCRIPTOR_NAMES, read_fasta_stream, compute_sequence_descriptors
from train_multiclass import SEED
from full_classifier import FASTA, RESULTS_DIR
from classify_which import CROSS, COMPACT

CAP = 400
DBS = ["eccDNABase", "CircleBaseV2"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/richer_kmers_features.tsv")
VOCAB = {k: [''.join(p) for p in product("ACGT", repeat=k)] for k in (3, 4, 5)}
IDX = {k: {km: i for i, km in enumerate(v)} for k, v in VOCAB.items()}


def kmer_freq(seq, k):
    v = np.zeros(len(VOCAB[k]), dtype=np.float32); n = len(seq) - k + 1
    if n <= 0:
        return v
    idx = IDX[k]
    for i in range(n):
        j = idx.get(seq[i:i + k])
        if j is not None:
            v[j] += 1
    return v / n


def featurize(seq):
    d = compute_sequence_descriptors(seq)
    row = [d[c] for c in DESCRIPTOR_NAMES]
    for k in (3, 4, 5):
        row.extend(kmer_freq(seq, k).tolist())
    return row


COLS = list(DESCRIPTOR_NAMES) + [f"k3_{m}" for m in VOCAB[3]] + [f"k4_{m}" for m in VOCAB[4]] + [f"k5_{m}" for m in VOCAB[5]]
SETS = {
    "A: 10 descrittori": list(DESCRIPTOR_NAMES),
    "B: +3-mer (74)": list(DESCRIPTOR_NAMES) + [f"k3_{m}" for m in VOCAB[3]],
    "C: +4-mer (330)": list(DESCRIPTOR_NAMES) + [f"k3_{m}" for m in VOCAB[3]] + [f"k4_{m}" for m in VOCAB[4]],
    "D: +5-mer (1354)": COLS,
}


def load():
    buf = {(d, s): [] for d in CROSS for s in DBS}
    for ch in pd.read_csv(COMPACT, sep="\t", usecols=["id", "label", "source_db", "length"], chunksize=400000, low_memory=False):
        ch = ch[ch["label"].isin(CROSS) & ch["source_db"].isin(DBS)]
        for (d, s), g in ch.groupby(["label", "source_db"]):
            b = buf[(d, s)]
            if sum(len(x) for x in b) < CAP:
                b.append(g.head(CAP)[["id", "label", "source_db", "length"]])
    meta = pd.concat([pd.concat(v).head(CAP) for v in buf.values() if v], ignore_index=True)
    meta["id"] = meta["id"].astype(str)
    ft = None
    if os.path.exists(CACHE):
        ft = pd.read_csv(CACHE, sep="\t", index_col=0); ft.index = ft.index.astype(str)
    ids = set(meta["id"]) - (set(ft.index) if ft is not None else set())
    if ids:
        rows = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
            if "N" not in seq and len(seq) >= 6:
                rows[sid] = featurize(seq)
        new = pd.DataFrame.from_dict(rows, orient="index", columns=COLS)
        ft = new if ft is None else pd.concat([ft, new]); ft = ft[~ft.index.duplicated()]; ft.to_csv(CACHE, sep="\t")
    ftr = ft.reset_index().rename(columns={ft.reset_index().columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    return meta.merge(ftr, on="id", how="inner").dropna(subset=COLS)


def length_match(df, n_bins=6, seed=SEED):
    d = df.copy(); d["lb"] = pd.qcut(d["length"], n_bins, duplicates="drop")
    classes = sorted(d["label"].unique()); parts = []
    for _, g in d.groupby("lb", observed=True):
        vc = g["label"].value_counts()
        if len(vc) == len(classes):
            k = int(vc.min())
            for c in classes:
                parts.append(g[g["label"] == c].sample(k, random_state=seed))
    return pd.concat(parts, ignore_index=True).drop(columns="lb")


def main():
    df = length_match(load())
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    chance = 1 / nC
    y = df["label"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()
    print(f"9 malattie, {len(df)} sequenze (length-matched), caso={chance:.3f}\n")

    def rf_eval(Xtr, ytr, Xte, yte):
        rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(Xtr, ytr)
        pr = rf.predict(Xte)
        return accuracy_score(yte, pr), f1_score(yte, pr, average="macro", zero_division=0), rf

    a = db == DBS[0]; b = db == DBS[1]
    rows = []
    for name, cols in SETS.items():
        X = df[cols].to_numpy(np.float32)
        # within-study
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        w_acc, w_f1, _ = rf_eval(Xtr, ytr, Xte, yte)
        # cross-studio (media 2 direzioni)
        c1 = rf_eval(X[a], y[a], X[b], y[b]); c2 = rf_eval(X[b], y[b], X[a], y[a])
        c_acc = (c1[0] + c2[0]) / 2; c_f1 = (c1[1] + c2[1]) / 2
        rows.append({"set": name, "within_acc": round(w_acc, 3), "within_F1": round(w_f1, 3),
                     "within_xcaso": round(w_acc / chance, 2),
                     "cross_acc": round(c_acc, 3), "cross_F1": round(c_f1, 3),
                     "cross_xcaso": round(c_acc / chance, 2)})
        print(f"  {name:22s} within: acc={w_acc:.3f} F1={w_f1:.3f} ({w_acc/chance:.1f}x) | cross: acc={c_acc:.3f} F1={c_f1:.3f} ({c_acc/chance:.1f}x)")

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "richer_kmers.tsv"), sep="\t", index=False)

    # importanza feature sul set C (10+3mer+4mer), within-study
    X = df[SETS["C: +4-mer (330)"]].to_numpy(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    _, _, rf = rf_eval(Xtr, ytr, Xte, yte)
    imp = pd.Series(rf.feature_importances_, index=SETS["C: +4-mer (330)"]).sort_values(ascending=False)
    print("\n--- Top 20 feature per importanza (set C: 10+3mer+4mer) ---")
    print(imp.head(20).round(4).to_string())
    order = imp.groupby(imp.index.str[:2].where(imp.index.str.startswith("k"), "desc")).sum()
    print("\n--- Importanza totale per famiglia ---")
    print(order.round(3).to_string())
    print(f"\nSalvato in {RESULTS_DIR}/richer_kmers.tsv")


if __name__ == "__main__":
    main()
