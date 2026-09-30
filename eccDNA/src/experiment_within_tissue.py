"""Primo test ONESTO di 'malato vs sano' DENTRO lo stesso tessuto.

Finora 'sano vs malato' confrontava tessuti diversi (sani = muscolo/plasma/sperma;
malati = tumori solidi) -> separava il tipo di tessuto, non la malattia. Solo due
organi hanno un sano corrispondente, con metodo e database combacianti:

 - CERVELLO / ATAC-seq / eccDNABase: Glioblastoma (3452) vs cervello sano (290)
 - CUORE   / Circle-seq / eccDNABase: Dilated Cardiomyopathy (1513) vs cuore sano (2068)

Tessuto, metodo e database IDENTICI tra sano e malato + length-matching -> un AUC
sopra 0.5 qui e' segnale di MALATTIA vero (non tessuto, non protocollo, non
lunghezza). Sul cervello i sani sono pochi (290): terreno ideale per le reti
siamesi, che lavorano bene con pochi esempi.

Si confrontano i 3 modi di lavorare della rete siamese:
 - prototipico EUCLIDEO, prototipico COSENO, TRIPLET (batch-hard)
piu' softmax e RandomForest come riferimento, e un baseline SOLO-lunghezza (deve
restare ~0.5 se il matching funziona). Metrica: ROC-AUC malato-vs-sano.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from eccdna_utils import read_fasta_stream, compute_sequence_descriptors
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from train_multiclass import (
    standardize, train_prototypical, predict_prototypical,
    train_softmax, predict_softmax, SEED,
)
from experiment_losses_fewshot import train_metric_encoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection_metadata.tsv")
FASTA = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection.body.fa")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CACHE = os.path.join(SCRIPT_DIR, "data/processed/within_tissue_features.tsv")

# (nome, tissue, method, keyword-malattia)
TASKS = [
    ("Cervello/ATAC (Glioblastoma vs sano)", "brain", "atac-seq", "glioblastoma"),
    ("Cuore/Circle-seq (Cardiomiopatia vs sano)", "cardiac", "circle-seq", "dilated cardiomyopathy"),
]


def load_task_ids():
    """Raccoglie gli id (malato/sano) per i due task, filtrando eccDNABase +
    tissue + method combacianti."""
    want = {i: {"pos": [], "neg": []} for i in range(len(TASKS))}
    for ch in pd.read_csv(META, sep="\t",
                          usecols=["id", "disease", "disease_binary_name", "tissue", "source_db", "method", "length"],
                          chunksize=400000, low_memory=False):
        ch = ch[ch["source_db"] == "eccDNABase"].copy()
        ch["tl"] = ch["tissue"].astype(str).str.lower()
        ch["ml"] = ch["method"].astype(str).str.lower()
        ch["dl"] = ch["disease"].astype(str).str.lower()
        ch["is_h"] = ch["disease_binary_name"].astype(str).str.lower() == "healthy"
        for i, (_, tis, met, dkw) in enumerate(TASKS):
            base = ch[(ch["tl"] == tis) & (ch["ml"] == met)]
            want[i]["neg"].append(base[base["is_h"]][["id", "length"]])
            want[i]["pos"].append(base[~base["is_h"] & base["dl"].str.contains(dkw)][["id", "length"]])
    tasks = []
    for i in range(len(TASKS)):
        pos = pd.concat(want[i]["pos"], ignore_index=True).assign(y=1)
        neg = pd.concat(want[i]["neg"], ignore_index=True).assign(y=0)
        df = pd.concat([pos, neg], ignore_index=True)
        df["id"] = df["id"].astype(str)
        tasks.append(df)
    return tasks


def get_features(tasks):
    all_ids = set().union(*[set(t["id"]) for t in tasks])
    if os.path.exists(CACHE):
        ftab = pd.read_csv(CACHE, sep="\t", index_col=0); ftab.index = ftab.index.astype(str)
        if all_ids.issubset(set(ftab.index)):
            return ftab
    feats = {}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=all_ids):
        if "N" not in seq and len(seq) >= 4:
            f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq)); feats[sid] = f
    ftab = pd.DataFrame.from_dict(feats, orient="index")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True); ftab.to_csv(CACHE, sep="\t")
    return ftab


def length_match(df, caliper=0.10, seed=SEED):
    """Abbinamento per lunghezza piu' vicina (caliper): ogni campione della
    classe minoritaria viene accoppiato al piu' vicino per lunghezza della
    maggioritaria non ancora usato, se entro il caliper relativo. Garantisce
    distribuzioni di lunghezza quasi identiche (il controllo solo-lunghezza
    deve scendere a ~0.5)."""
    pos = df[df.y == 1]; neg = df[df.y == 0]
    minority, majority = (neg, pos) if len(neg) <= len(pos) else (pos, neg)
    maj = majority.sort_values("length").reset_index(drop=True)
    maj_len = maj["length"].to_numpy(np.float64)
    used = np.zeros(len(maj), bool)
    keep_min, keep_maj = [], []
    for _, r in minority.iterrows():
        d = np.abs(maj_len - r["length"]); d[used] = np.inf
        j = int(d.argmin())
        if d[j] <= caliper * max(r["length"], 1):  # entro il 10% della lunghezza
            used[j] = True; keep_min.append(r); keep_maj.append(maj.iloc[j])
    out = pd.concat([pd.DataFrame(keep_min), pd.DataFrame(keep_maj)], ignore_index=True)
    return out


def split(df, seed=SEED):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(df))
    n = len(df); nte = int(0.2 * n); nva = int(0.15 * n)
    s = np.array(["train"] * n, dtype=object); s[idx[:nte]] = "test"; s[idx[nte:nte + nva]] = "val"
    df = df.copy(); df["split"] = s
    return df


def auc_disease(name, ytr, ptr_fn, yte, pte):
    return roc_auc_score(yte, pte)


def run_task(nome, df, ftab):
    df = df.set_index("id").join(ftab, how="inner").reset_index().dropna(subset=FEATURE_COLS)
    df = length_match(df)
    df = split(df)
    tr, va, te = df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]
    print(f"\n### {nome}")
    print(f"    dopo length-matching: malati={int((df.y==1).sum())} sani={int((df.y==0).sum())}  "
          f"(train={len(tr)} val={len(va)} test={len(te)})")
    Xtr = tr[FEATURE_COLS].to_numpy(np.float32); Xva = va[FEATURE_COLS].to_numpy(np.float32); Xte = te[FEATURE_COLS].to_numpy(np.float32)
    ytr = tr["y"].to_numpy(); yva = va["y"].to_numpy(); yte = te["y"].to_numpy()
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)

    res = {}
    # baseline solo-lunghezza (deve restare ~0.5)
    rf_len = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(tr[["length"]], ytr)
    res["solo-lunghezza (controllo)"] = roc_auc_score(yte, rf_len.predict_proba(te[["length"]])[:, 1])
    # RandomForest sulle 74 feature
    rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    res["RandomForest 74"] = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    # softmax
    sm = train_softmax(Xtr, ytr, Xva, yva, 2, seed=SEED, balanced=True)
    res["Softmax"] = roc_auc_score(yte, predict_softmax(sm, Xte)[:, 1])
    # 3 modi siamese
    enc, _ = train_prototypical(Xtr, ytr, Xva, yva, 2, cosine=False, seed=SEED, select_metric="balanced")
    res["Siamese prototipico euclideo"] = roc_auc_score(yte, predict_prototypical(enc, Xtr, ytr, Xte, 2, False)[:, 1])
    enc, _ = train_prototypical(Xtr, ytr, Xva, yva, 2, cosine=True, seed=SEED, select_metric="balanced")
    res["Siamese prototipico coseno"] = roc_auc_score(yte, predict_prototypical(enc, Xtr, ytr, Xte, 2, True)[:, 1])
    enc, _ = train_metric_encoder(Xtr, ytr, Xva, yva, 2, "triplet", margin=0.3)
    res["Siamese triplet"] = roc_auc_score(yte, predict_prototypical(enc, Xtr, ytr, Xte, 2, False)[:, 1])
    return res


def main():
    tasks = load_task_ids()
    ftab = get_features(tasks)
    allres = {}
    for (nome, *_), df in zip(TASKS, tasks):
        allres[nome] = run_task(nome, df, ftab)
    out = pd.DataFrame(allres).round(3)
    print("\n=== ROC-AUC malato-vs-sano DENTRO lo stesso tessuto (0.5 = caso) ===")
    print(out.to_string())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(RESULTS_DIR, "within_tissue_disease.tsv"), sep="\t")
    print(f"\nSalvato in {RESULTS_DIR}/within_tissue_disease.tsv")


if __name__ == "__main__":
    main()
