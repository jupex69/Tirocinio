"""OBIETTIVO MALATTIA (non tessuto): distinguere STATO di malattia dentro lo stesso
tessuto e protocollo. Caso disponibile e controllabile:

  Colon-retto, eccDNABase, Circle-seq:
    Colorectal Adenoma (benigno, 1803)  vs  Colorectal Cancer (maligno, 1636)

Stesso tessuto, stesso database, stesso metodo, lunghezze simili (555 vs 606) ->
un AUC sopra 0.5 qui e' segnale di MALATTIA/MALIGNITA' vero (benigno vs maligno),
non tessuto, non salute generica, non protocollo. E' la domanda clinicamente
rilevante ('la lesione e' cancerosa?').

Stessi criteri rigorosi degli altri test: caliper length-matching, split
stratificato per lunghezza, IC 95% bootstrap, controllo solo-lunghezza, e i 3
modi delle reti siamesi + RandomForest/softmax.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from eccdna_utils import read_fasta_stream, compute_sequence_descriptors
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from train_multiclass import (
    standardize, train_prototypical, predict_prototypical, train_softmax, predict_softmax, SEED,
)
from experiment_losses_fewshot import train_metric_encoder
from cardiac_certify import caliper_match, strat_split, boot_ci

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection_metadata.tsv")
FASTA = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection.body.fa")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CACHE = os.path.join(SCRIPT_DIR, "data/processed/disease_state_features.tsv")


def load_ids():
    """Colorectal Adenoma (y=0) vs Colorectal Cancer (y=1), eccDNABase/Circle-seq."""
    parts = []
    for ch in pd.read_csv(META, sep="\t",
                          usecols=["id", "disease", "source_db", "method", "length"],
                          chunksize=400000, low_memory=False):
        ch = ch[(ch["source_db"] == "eccDNABase") & (ch["method"].astype(str).str.lower() == "circle-seq")]
        dl = ch["disease"].astype(str).str.lower()
        ade = ch[dl == "colorectal adenoma"].assign(y=0)
        can = ch[dl == "colorectal cancer"].assign(y=1)
        parts.append(pd.concat([ade, can])[["id", "y", "length"]])
    df = pd.concat(parts, ignore_index=True); df["id"] = df["id"].astype(str)
    return df


def get_features(df):
    if os.path.exists(CACHE):
        ftab = pd.read_csv(CACHE, sep="\t", index_col=0); ftab.index = ftab.index.astype(str)
        if set(df["id"]).issubset(set(ftab.index)):
            return ftab
    feats = {}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=set(df["id"])):
        if "N" not in seq and len(seq) >= 4:
            f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq)); feats[sid] = f
    ftab = pd.DataFrame.from_dict(feats, orient="index")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True); ftab.to_csv(CACHE, sep="\t")
    return ftab


def main():
    df = load_ids()
    print(f"Adenoma (benigno): {int((df.y==0).sum())}   Cancer (maligno): {int((df.y==1).sum())}")
    ftab = get_features(df)
    ft = ftab.reset_index(); ft = ft.rename(columns={ft.columns[0]: "id"}); ft["id"] = ft["id"].astype(str)
    df = df.merge(ft, on="id", how="inner").dropna(subset=FEATURE_COLS)

    df = caliper_match(df); df = strat_split(df)
    tr, va, te = df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]
    print(f"dopo length-matching: benigni={int((df.y==0).sum())} maligni={int((df.y==1).sum())}  "
          f"(train={len(tr)} val={len(va)} test={len(te)})")
    print(f"len mediana test: maligni={te[te.y==1]['length'].median():.0f} benigni={te[te.y==0]['length'].median():.0f}\n")

    Xtr = tr[FEATURE_COLS].to_numpy(np.float32); Xva = va[FEATURE_COLS].to_numpy(np.float32); Xte = te[FEATURE_COLS].to_numpy(np.float32)
    ytr, yva, yte = tr.y.to_numpy(), va.y.to_numpy(), te.y.to_numpy()
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)

    def show(name, p):
        m, lo, hi = boot_ci(yte, p)
        flag = "> 0.5 (segnale)" if lo > 0.5 else "include 0.5"
        print(f"  {name:30s} AUC={m:.3f}  IC95%=[{lo:.3f},{hi:.3f}]  {flag}")
        return {"modello": name, "AUC": round(m, 3), "IC": f"[{lo:.3f},{hi:.3f}]"}

    rows = []
    # controllo lunghezza
    pl = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(tr[["length"]], ytr).predict_proba(te[["length"]])[:, 1]
    rows.append(show("solo-lunghezza (controllo)", pl))
    # RF
    rows.append(show("RandomForest 74", RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
    # softmax
    sm = train_softmax(Xtr, ytr, Xva, yva, 2, seed=SEED, balanced=True)
    rows.append(show("Softmax", predict_softmax(sm, Xte)[:, 1]))
    # 3 modi siamese
    enc, _ = train_prototypical(Xtr, ytr, Xva, yva, 2, cosine=False, seed=SEED, select_metric="balanced")
    rows.append(show("Siamese prototipico euclideo", predict_prototypical(enc, Xtr, ytr, Xte, 2, False)[:, 1]))
    enc, _ = train_prototypical(Xtr, ytr, Xva, yva, 2, cosine=True, seed=SEED, select_metric="balanced")
    rows.append(show("Siamese prototipico coseno", predict_prototypical(enc, Xtr, ytr, Xte, 2, True)[:, 1]))
    enc, _ = train_metric_encoder(Xtr, ytr, Xva, yva, 2, "triplet", margin=0.3)
    rows.append(show("Siamese triplet", predict_prototypical(enc, Xtr, ytr, Xte, 2, False)[:, 1]))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "disease_state_colorectal.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/disease_state_colorectal.tsv")


if __name__ == "__main__":
    main()
