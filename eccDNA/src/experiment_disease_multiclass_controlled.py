"""Multiclasse a MALATTIA con METODO E DATABASE COSTANTI (RQ2, versione controllata).

Gemello di experiment_disease_multiclass.py, ma qui il confondente 'metodo' e
'studio/database' sono tenuti COSTANTI per costruzione: si usa il solo strato
source_db == CircleBaseV2 e method == Circle_seq. Cosi' la separabilita' residua
tra le 9 malattie non puo' dipendere dal protocollo di sequenziamento ne' dal
database, ma solo (a) dalla biologia e (b) dal residuo di studio (ogni malattia,
anche qui, resta in larga parte un singolo studio: e' il limite dichiarato).

Le classi sono ancora appaiate per lunghezza (length-matching). Modelli: RandomForest,
softmax, siamese coseno, baseline solo-lunghezza + test di permutazione. Feature: 74
(64 spettro 3-mer + 10 descrittori). Confronto diretto con la versione solo
length-matched su 2 DB (experiment_disease_multiclass.py).
"""

import argparse
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
DB_CONST = "CircleBaseV2"
METHOD_CONST = "Circle_seq"
MIN_N = 120   # una malattia entra solo se ha almeno MIN_N campioni nello strato


def load(with_sano):
    classes = (["Sano"] + CROSS) if with_sano else list(CROSS)
    buf = {d: [] for d in classes}
    counts = {d: 0 for d in classes}
    for ch in pd.read_csv(COMPACT, sep="\t",
                          usecols=["id", "label", "is_disease", "source_db", "method", "length"],
                          chunksize=400000, low_memory=False):
        stratum = ch[(ch["source_db"] == DB_CONST) & (ch["method"] == METHOD_CONST)]
        # malattie
        dis = stratum[stratum["label"].isin(CROSS)]
        for d, g in dis.groupby("label"):
            counts[d] += len(g)
            if sum(len(x) for x in buf[d]) < CAP:
                buf[d].append(g.head(CAP)[["id", "label", "length"]])
        # sani (qualunque etichetta, is_disease == 0) -> classe unica "Sano"
        if with_sano:
            sane = stratum[stratum["is_disease"] == 0]
            counts["Sano"] += len(sane)
            if sum(len(x) for x in buf["Sano"]) < CAP and len(sane):
                s = sane.head(CAP)[["id", "length"]].assign(label="Sano")
                buf["Sano"].append(s[["id", "label", "length"]])
    print("--- Campioni per classe nello strato CircleBaseV2/Circle_seq ---")
    for d in classes:
        print(f"    {d:<32} {counts[d]:>10}")
    keep = [d for d in classes if counts[d] >= MIN_N]
    print(f"\nClassi con >= {MIN_N} campioni nello strato: {len(keep)}/{len(classes)}")
    meta = pd.concat([pd.concat(buf[d]).head(CAP) for d in keep], ignore_index=True)
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


def length_match(df, n_bins=5, seed=SEED):
    d = df.copy(); d["lb"] = pd.qcut(d["length"], n_bins, duplicates="drop")
    classes = sorted(d["label"].unique()); parts = []
    for _, g in d.groupby("lb", observed=True):
        vc = g["label"].value_counts()
        if len(vc) == len(classes):
            k = int(vc.min())
            for c in classes:
                parts.append(g[g["label"] == c].sample(k, random_state=seed))
    return pd.concat(parts, ignore_index=True).drop(columns="lb")


def main(with_sano=False):
    tag = "10way_sano" if with_sano else "9way_disease"
    df = load(with_sano)
    df = length_match(df)
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    chance = 1 / nC
    X = df[FEATURE_COLS].to_numpy(np.float32); y = df["label"].map(c2i).to_numpy()
    L = df["length"].to_numpy().reshape(-1, 1)
    print(f"\nMulticlasse CONTROLLATO (metodo+DB costanti): {nC} classi, {len(df)} sequenze, "
          f"caso={chance:.3f}, per-classe~{len(df)//nC}\n")

    Xtr, Xte, ytr, yte, Ltr, Lte = train_test_split(X, y, L, test_size=0.3, random_state=SEED, stratify=y)
    Xtr_s, Xte_s = standardize(Xtr, Xte)

    rows = []
    def ev(name, pred):
        acc = accuracy_score(yte, pred); bacc = balanced_accuracy_score(yte, pred)
        mf1 = f1_score(yte, pred, average="macro", zero_division=0)
        rows.append({"modello": name, "accuracy": round(acc, 3), "bal_acc": round(bacc, 3),
                     "macro_f1": round(mf1, 3), "x_caso": round(acc / chance, 2)})
        print(f"  {name:24s} acc={acc:.3f} bal_acc={bacc:.3f} macro-F1={mf1:.3f} ({acc/chance:.1f}x)")
        return pred

    rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(Xtr_s, ytr)
    pred_rf = ev("RandomForest 74", rf.predict(Xte_s))
    sm = train_softmax(Xtr_s, ytr, Xtr_s, ytr, nC, seed=SEED, balanced=True)
    ev("Softmax", predict_softmax(sm, Xte_s).argmax(1))
    enc, _ = train_prototypical(Xtr_s, ytr, Xtr_s, ytr, nC, cosine=True, seed=SEED, select_metric="balanced")
    ev("Siamese coseno", predict_prototypical(enc, Xtr_s, ytr, Xte_s, nC, True).argmax(1))
    rfl = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Ltr, ytr)
    ev("Baseline solo-lunghezza", rfl.predict(Lte))

    rng = np.random.default_rng(SEED); real = accuracy_score(yte, pred_rf)
    null = [accuracy_score(rng.permutation(yte), pred_rf) for _ in range(1000)]
    pval = (sum(n >= real for n in null) + 1) / 1001
    print(f"\nTest di permutazione (RF): reale={real:.3f}, caso medio={np.mean(null):.3f}, p={pval:.4f}")

    rec = recall_score(yte, pred_rf, average=None, labels=range(nC), zero_division=0)
    f1 = f1_score(yte, pred_rf, average=None, labels=range(nC), zero_division=0)
    per = pd.DataFrame({"malattia": classes, "recall": np.round(rec, 3), "f1": np.round(f1, 3)}).sort_values("f1", ascending=False)
    print("\n--- Per malattia (RandomForest, metodo+DB costanti) ---")
    print(per.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, f"disease_multiclass_controlled_{tag}.tsv"), sep="\t", index=False)
    per.to_csv(os.path.join(RESULTS_DIR, f"disease_multiclass_controlled_{tag}_perclass.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/disease_multiclass_controlled_{tag}.tsv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_sano", action="store_true", help="aggiunge la classe Sano (10-way)")
    main(**vars(ap.parse_args()))
