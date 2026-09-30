"""Multiclasse a livello di MALATTIA (non tessuto), within-study, per RQ2.

Le 9 malattie cross-testabili (presenti in >=2 studi) sono l'obiettivo del
multiclasse. Qui se ne misura la fattibilita' WITHIN-STUDY (split casuale): quanto
la sola composizione le distingue, in condizioni controllate. Feature: le 74
(64 spettro 3-mer + 10 descrittori). Modelli: RandomForest, softmax, siamese
coseno, e baseline solo-lunghezza. Metriche globali + recall/F1 per malattia.
La generalizzazione cross-studio e' invece oggetto di RQ3.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from eccdna_utils import read_fasta_stream, compute_sequence_descriptors
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from train_multiclass import (standardize, train_prototypical, predict_prototypical,
                              train_softmax, predict_softmax, SEED)
from full_classifier import FASTA, RESULTS_DIR
from classify_which import CROSS, CACHE, COMPACT

CAP = 800
DBS = ["eccDNABase", "CircleBaseV2"]


def load():
    buf = {(d, s): [] for d in CROSS for s in DBS}
    for ch in pd.read_csv(COMPACT, sep="\t", usecols=["id", "label", "source_db", "length"],
                          chunksize=400000, low_memory=False):
        ch = ch[ch["label"].isin(CROSS) & ch["source_db"].isin(DBS)]
        for (d, s), g in ch.groupby(["label", "source_db"]):
            b = buf[(d, s)]
            if sum(len(x) for x in b) < CAP:
                b.append(g.head(CAP)[["id", "label", "source_db", "length"]])
    meta = pd.concat([pd.concat(v).head(CAP) for v in buf.values() if v], ignore_index=True)
    meta["id"] = meta["id"].astype(str)
    ft = None; ids = set(meta["id"])
    if os.path.exists(CACHE):
        ft = pd.read_csv(CACHE, sep="\t", index_col=0); ft.index = ft.index.astype(str); ids = ids - set(ft.index)
    if ids:
        feats = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
            if "N" not in seq and len(seq) >= 4:
                f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq)); feats[sid] = f
        new = pd.DataFrame.from_dict(feats, orient="index")
        ft = new if ft is None else pd.concat([ft, new]); ft = ft[~ft.index.duplicated()]; ft.to_csv(CACHE, sep="\t")
    ftr = ft.reset_index().rename(columns={ft.reset_index().columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    return meta.merge(ftr[["id"] + FEATURE_COLS], on="id", how="inner").dropna(subset=FEATURE_COLS)


def length_match(df, n_bins=6, seed=SEED):
    """Appaiamento per lunghezza tra le malattie (bin di quantile, stesso numero
    per classe in ogni bin dove tutte sono presenti)."""
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
    df = load()
    df = length_match(df)
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    chance = 1 / nC
    X = df[FEATURE_COLS].to_numpy(np.float32); y = df["label"].map(c2i).to_numpy(); L = df["length"].to_numpy().reshape(-1, 1)
    print(f"Multiclasse MALATTIE: {nC} classi, {len(df)} sequenze, caso={chance:.3f}\n")

    Xtr, Xte, ytr, yte, Ltr, Lte = train_test_split(X, y, L, test_size=0.3, random_state=SEED, stratify=y)
    Xtr_s, Xte_s = standardize(Xtr, Xte)

    rows = []
    def ev(name, pred, proba=None):
        acc = accuracy_score(yte, pred); bacc = balanced_accuracy_score(yte, pred)
        mf1 = f1_score(yte, pred, average="macro", zero_division=0)
        rows.append({"modello": name, "accuracy": round(acc, 3), "bal_acc": round(bacc, 3),
                     "macro_f1": round(mf1, 3), "x_caso": round(acc / chance, 2)})
        print(f"  {name:22s} acc={acc:.3f} bal_acc={bacc:.3f} macro-F1={mf1:.3f} ({acc/chance:.1f}x)")
        return pred

    rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(Xtr_s, ytr)
    pred_rf = ev("RandomForest 74", rf.predict(Xte_s))
    sm = train_softmax(Xtr_s, ytr, Xtr_s, ytr, nC, seed=SEED, balanced=True)
    ev("Softmax", predict_softmax(sm, Xte_s).argmax(1))
    enc, _ = train_prototypical(Xtr_s, ytr, Xtr_s, ytr, nC, cosine=True, seed=SEED, select_metric="balanced")
    ev("Siamese coseno", predict_prototypical(enc, Xtr_s, ytr, Xte_s, nC, True).argmax(1))
    rfl = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Ltr, ytr)
    ev("Baseline solo-lunghezza", rfl.predict(Lte))

    # test di permutazione (significativita' vs caso) sul RandomForest
    rng = np.random.default_rng(SEED); real = accuracy_score(yte, pred_rf)
    null = [accuracy_score(rng.permutation(yte), pred_rf) for _ in range(1000)]
    pval = (sum(n >= real for n in null) + 1) / 1001
    print(f"\nTest di permutazione (RF): reale={real:.3f}, caso medio={np.mean(null):.3f}, p={pval:.4f}")

    # per-malattia (RandomForest)
    rec = recall_score(yte, pred_rf, average=None, labels=range(nC), zero_division=0)
    f1 = f1_score(yte, pred_rf, average=None, labels=range(nC), zero_division=0)
    per = pd.DataFrame({"malattia": classes, "recall": np.round(rec, 3), "f1": np.round(f1, 3)}).sort_values("f1", ascending=False)
    print("\n--- Per malattia (RandomForest, within-study) ---")
    print(per.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "disease_multiclass.tsv"), sep="\t", index=False)
    per.to_csv(os.path.join(RESULTS_DIR, "disease_multiclass_perclass.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/disease_multiclass.tsv")


if __name__ == "__main__":
    main()
