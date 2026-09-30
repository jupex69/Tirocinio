"""Multiclasse ONESTO con TUTTE le classi + le 3 reti siamesi.

Risponde a due richieste:
 - tenere TUTTE le malattie (non solo le 10 cross-database): le classi presenti in
   un solo studio, cross-lab, avranno recall ~0 (mai viste in training) -> e' la
   risposta onesta sulla generalizzazione;
 - usare le RETI SIAMESI con le 3 loss (prototipico euclideo, coseno, triplet),
   con RandomForest come riferimento.

Valutazioni:
 (A) WITHIN-STUDY (split casuale): numero confuso di riferimento.
 (B) CROSS-STUDY (train su un database, test sull'altro, media delle 2 direzioni):
     numero ONESTO. Le classi solo-in-un-DB restano nel problema (nello spazio di
     etichette globale) e pesano: cross-lab non sono predicibili.

Feature: 74 + 4 nuovi. Riusa canon()/cache di full_classifier.py.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from full_classifier import canon, get_features, ALLCOLS, META, RESULTS_DIR, CHUNK, SEED
from train_multiclass import standardize, train_prototypical, predict_prototypical
from experiment_losses_fewshot import train_metric_encoder

CAP = 600
MIN_N = 150
DBS = ["eccDNABase", "CircleBaseV2"]


def load_all():
    """TUTTE le classi canonicalizzate (totale >= MIN_N) + Healthy, con tetto per
    (classe, database) per non far dominare le classi giganti."""
    buf = {}
    for ch in pd.read_csv(META, sep="\t",
                          usecols=["id", "disease", "disease_binary_name", "source_db", "length"],
                          chunksize=CHUNK, low_memory=False):
        ch = ch[ch["source_db"].isin(DBS)]
        healthy = ch["disease_binary_name"].astype(str).str.lower() == "healthy"
        ch["lab"] = np.where(healthy, "Healthy", ch["disease"].map(canon))
        ch = ch[ch["lab"].notna() & (ch["lab"] != "None")]
        for (lab, db), g in ch.groupby(["lab", "source_db"]):
            key = (lab, db); cur = buf.get(key); have = 0 if cur is None else len(cur)
            if have < CAP:
                take = g.head(CAP - have)[["id", "lab", "source_db"]]
                buf[key] = take if cur is None else pd.concat([cur, take], ignore_index=True)
    df = pd.concat(buf.values(), ignore_index=True); df["id"] = df["id"].astype(str)
    keep = df["lab"].value_counts(); keep = keep[keep >= MIN_N].index
    return df[df["lab"].isin(keep)].reset_index(drop=True)


def _safe_split(X, yidx, frac=0.15):
    counts = np.bincount(yidx)
    strat = yidx if (counts.min() >= 2) else None
    return train_test_split(X, yidx, test_size=frac, random_state=SEED, stratify=strat)


def train_predict(kind, Xa, ya, Xb):
    """Addestra su A (etichette stringa globali), predice su B. Ritorna etichette
    stringa predette. Le classi assenti in A non sono predicibili (onesto)."""
    Xa_s, Xb_s = standardize(Xa, Xb)
    classesA = sorted(set(ya))
    if kind == "RF":
        rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(Xa_s, ya)
        return rf.predict(Xb_s)
    l2i = {c: i for i, c in enumerate(classesA)}
    ya_idx = np.array([l2i[c] for c in ya])
    Xtr, Xval, ytr, yval = _safe_split(Xa_s, ya_idx)
    if kind in ("euclideo", "coseno"):
        cos = kind == "coseno"
        enc, _ = train_prototypical(Xtr, ytr, Xval, yval, len(classesA), cosine=cos, seed=SEED, select_metric="balanced")
        proba = predict_prototypical(enc, Xa_s, ya_idx, Xb_s, len(classesA), cosine=cos)
    else:  # triplet
        enc, _ = train_metric_encoder(Xtr, ytr, Xval, yval, len(classesA), "triplet", margin=0.3)
        proba = predict_prototypical(enc, Xa_s, ya_idx, Xb_s, len(classesA), cosine=False)
    return np.array([classesA[i] for i in proba.argmax(1)])


def main():
    df = load_all()
    ft = get_features(df)
    ftr = ft.reset_index(); ftr = ftr.rename(columns={ftr.columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    df = df.merge(ftr, on="id", how="inner").dropna(subset=ALLCOLS)
    classes = sorted(df["lab"].unique()); nC = len(classes)
    inboth = df.groupby("lab")["source_db"].nunique()
    n_both = int((inboth >= 2).sum())
    print(f"Classi totali tenute: {nC}  (di cui in entrambi i DB: {n_both})  chance={1/nC:.3f}")
    print(f"Campioni: {len(df)}\n")

    X = df[ALLCOLS].to_numpy(np.float32); y = df["lab"].to_numpy(); dbcol = df["source_db"].to_numpy()
    models = ["RF", "euclideo", "coseno", "triplet"]

    # (A) within-study (split casuale)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    print("=== (A) WITHIN-STUDY (split casuale, confuso) ===")
    withinA = {}
    for mk in models:
        pred = train_predict(mk, Xtr, ytr, Xte)
        a = accuracy_score(yte, pred); b = balanced_accuracy_score(yte, pred)
        withinA[mk] = a
        print(f"  {mk:9s}: accuracy={a:.3f}  bal_acc={b:.3f}  ({a/(1/nC):.1f}x)")

    # (B) cross-study (train un DB, test l'altro; media 2 direzioni)
    print("\n=== (B) CROSS-STUDY (train un lab, test l'altro) — ONESTO, tutte le classi ===")
    a_mask = dbcol == "eccDNABase"; b_mask = dbcol == "CircleBaseV2"
    rows = []
    for mk in models:
        p_ab = train_predict(mk, X[a_mask], y[a_mask], X[b_mask])
        p_ba = train_predict(mk, X[b_mask], y[b_mask], X[a_mask])
        acc_ab = accuracy_score(y[b_mask], p_ab); acc_ba = accuracy_score(y[a_mask], p_ba)
        acc = (acc_ab + acc_ba) / 2
        rows.append({"modello": mk, "within_study": round(withinA[mk], 3),
                     "cross_study": round(acc, 3), "ecc->CB2": round(acc_ab, 3),
                     "CB2->ecc": round(acc_ba, 3), "x_caso_cross": round(acc/(1/nC), 2),
                     "calo": round(withinA[mk] - acc, 3)})
        print(f"  {mk:9s}: cross-study={acc:.3f}  (ecc->CB2={acc_ab:.3f}, CB2->ecc={acc_ba:.3f})  "
              f"({acc/(1/nC):.1f}x)   calo dal within: -{withinA[mk]-acc:.3f}")

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "honest_all_siamese.tsv"), sep="\t", index=False)
    print("\n=== Riepilogo (tutte le classi) ===")
    print(res.to_string(index=False))
    print(f"\nSalvato in {RESULTS_DIR}/honest_all_siamese.tsv")


if __name__ == "__main__":
    main()
