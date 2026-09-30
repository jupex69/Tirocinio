"""ELIMINARE metodo E database: validazione CROSS-SOURCE (train su un laboratorio,
test sull'altro). Il test biologico piu' pulito del progetto.

Idea (complementare a experiment_single_method.py): invece di TENERE COSTANTE il
protocollo, lo si INCROCIA. Sei tessuti compaiono in ENTRAMBI i database
(Colorectal, Glioblastoma, Hypopharynx, Ovarian, Prostate, Stomach). Si addestra
il classificatore su un database (es. eccDNABase) e lo si valuta sull'altro
(CircleBaseV2). Poiche' train e test differiscono SIA per database SIA per
metodo, qualunque cosa trasferisca NON puo' essere una scorciatoia di
protocollo: e' segnale indipendente dal batch -> il candidato piu' credibile a
'biologia'.

Confronto interno per ogni direzione:
 - within-source (train/val nello stesso DB): il tetto, con il batch disponibile;
 - transfer (test sull'ALTRO DB): quanto sopravvive togliendo metodo+database.
Il DIVARIO within-source - transfer misura la quota di batch/protocollo.

Le lunghezze sono matchate per quantili tra i tessuti dentro ogni database
(anche la lunghezza non deve fare da scorciatoia). Caso = 1/6 = 0.167.

LIMITE DICHIARATO: i tessuti cross-source restano pochi (6) e in un DB alcuni
sono piccoli (Ovarian in CircleBaseV2 ~464); i numeri di transfer vanno letti con
il loro intervallo. Ma la DIREZIONE del risultato (trasferisce o crolla) e'
informativa a prescindere.

Risultati in results/ con prefisso 'cross_source_'.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score

from eccdna_utils import read_fasta_stream, compute_sequence_descriptors
from gold_standard_data import canon_tissue
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from train_multiclass import (
    standardize, train_prototypical, predict_prototypical,
    train_softmax, predict_softmax, evaluate, SEED, DEVICE,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection_metadata.tsv")
FASTA = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection.body.fa")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CACHE = os.path.join(SCRIPT_DIR, "data/processed/cross_source_features.tsv")

CHUNK = 400000
CAP = 3000          # tetto per (tessuto, database)
N_BINS = 5          # bin di quantile per il length-matching dentro ogni DB
SOURCES = ["eccDNABase", "CircleBaseV2"]
# tessuti presenti in entrambi i DB con abbastanza campioni (verificato sui metadati)
CROSS_TISSUES = ["Colorectal", "Glioblastoma", "Hypopharynx", "Ovarian", "Prostate", "Stomach"]


def _length_match(df, n_bins=N_BINS, seed=SEED):
    """Campiona i tessuti alla STESSA distribuzione di lunghezza (bin di quantile,
    stesso numero per tessuto in ogni bin dove tutti sono presenti)."""
    tissues = sorted(df["tessuto"].unique())
    d = df.copy()
    d["lb"] = pd.qcut(d["length"], n_bins, duplicates="drop")
    parts = []
    for _, g in d.groupby("lb", observed=True):
        vc = g["tessuto"].value_counts()
        if len(vc) == len(tissues):
            k = int(vc.min())
            for t in tissues:
                parts.append(g[g["tessuto"] == t].sample(k, random_state=seed))
    return pd.concat(parts, ignore_index=True).drop(columns="lb")


def load_cross_base():
    """Per (tessuto cross-source, database): campiona fino a CAP righe, poi
    length-match dei tessuti dentro ogni database. Ritorna id, tessuto, source_db."""
    keep = {t: {s: [] for s in SOURCES} for t in CROSS_TISSUES}
    for ch in pd.read_csv(META, sep="\t",
                          usecols=["id", "disease", "disease_binary_name", "source_db", "length"],
                          chunksize=CHUNK, low_memory=False):
        ch = ch[ch["source_db"].isin(SOURCES)]
        ch = ch[ch["disease_binary_name"].astype(str).str.lower() != "healthy"]
        ch = ch.assign(tessuto=ch["disease"].map(canon_tissue))
        ch = ch[ch["tessuto"].isin(CROSS_TISSUES)]
        for (t, s), g in ch.groupby(["tessuto", "source_db"]):
            buf = keep[t][s]
            if sum(len(b) for b in buf) < CAP * 3:  # raccogli con margine, poi si campiona
                buf.append(g[["id", "tessuto", "source_db", "length"]])
    frames = []
    for t in CROSS_TISSUES:
        for s in SOURCES:
            if keep[t][s]:
                d = pd.concat(keep[t][s], ignore_index=True)
                frames.append(d.sample(n=min(len(d), CAP), random_state=SEED))
    base = pd.concat(frames, ignore_index=True)
    base["id"] = base["id"].astype(str)
    # length-match dentro ciascun database
    out = [_length_match(base[base["source_db"] == s]) for s in SOURCES]
    return pd.concat(out, ignore_index=True)


def get_features(base, force=False):
    if os.path.exists(CACHE) and not force:
        ftab = pd.read_csv(CACHE, sep="\t", index_col=0); ftab.index = ftab.index.astype(str)
    else:
        ids = set(base["id"])
        feats = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
            if "N" not in seq and len(seq) >= 4:
                f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq)); feats[sid] = f
        ftab = pd.DataFrame.from_dict(feats, orient="index")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True); ftab.to_csv(CACHE, sep="\t")
    return base.set_index("id").join(ftab, how="inner").reset_index().dropna(subset=FEATURE_COLS)


def run_direction(A, B, mc, c2i, n_classes, ep):
    """Addestra su A (train/val), valuta su A-val (within-source) e su TUTTO B
    (transfer). Ritorna righe di risultato per softmax e prototipico-coseno."""
    a = mc[mc.source_db == A]; b = mc[mc.source_db == B]
    Xa = a[FEATURE_COLS].to_numpy(np.float32); ya = a["tessuto"].map(c2i).to_numpy()
    Xb = b[FEATURE_COLS].to_numpy(np.float32); yb = b["tessuto"].map(c2i).to_numpy()
    Xtr, Xval, ytr, yval = train_test_split(Xa, ya, test_size=0.2, random_state=SEED, stratify=ya)
    Xtr, Xval, Xb_s = standardize(Xtr, Xval, Xb)  # standardizza con statistiche del TRAIN (A)
    chance = 1 / n_classes
    rows = []

    enc, _ = train_prototypical(Xtr, ytr, Xval, yval, n_classes, cosine=True, seed=SEED,
                                select_metric="balanced", epochs=ep["epochs"], episodes=ep["episodes"], patience=ep["patience"])
    sm = train_softmax(Xtr, ytr, Xval, yval, n_classes, seed=SEED, balanced=True,
                       epochs=ep["epochs"] * (25 if ep["epochs"] < 20 else 1), patience=ep["patience"])
    for name, pf in [("Prototipico coseno", lambda X: predict_prototypical(enc, Xtr, ytr, X, n_classes, True)),
                     ("Softmax bilanciato", lambda X: predict_softmax(sm, X))]:
        acc_in = evaluate(f"[{A}->{A}val] {name}", yval, pf(Xval), n_classes)["accuracy"]
        r = evaluate(f"[{A}->{B}] {name}", yb, pf(Xb_s), n_classes)
        rows.append({"direzione": f"{A}->{B}", "modello": name,
                     "within_source_acc": round(acc_in, 3), "transfer_acc": round(r["accuracy"], 3),
                     "transfer_bal_acc": round(r["balanced_accuracy"], 3),
                     "transfer_macroF1": round(r["macro_f1"], 3),
                     "transfer_x_chance": round(r["accuracy"] / chance, 2),
                     "gap_batch": round(acc_in - r["accuracy"], 3)})
    return rows


def main(quick=False, force=False):
    ep = dict(epochs=8, episodes=8, patience=4) if quick else dict(epochs=200, episodes=40, patience=20)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("--- CROSS-SOURCE: eliminare metodo+database incrociandoli ---")
    base = load_cross_base()
    mc = get_features(base, force=force)
    classes = sorted(mc["tessuto"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    print(f"Tessuti cross-source ({n_classes}): {classes}  chance={1/n_classes:.3f}")
    print(pd.crosstab(mc["tessuto"], mc["source_db"]).to_string())
    print()

    rows = []
    rows += run_direction("eccDNABase", "CircleBaseV2", mc, c2i, n_classes, ep)
    rows += run_direction("CircleBaseV2", "eccDNABase", mc, c2i, n_classes, ep)
    res = pd.DataFrame(rows)
    print("\n=== TRANSFER CROSS-SOURCE (caso {:.3f}) ===".format(1/n_classes))
    print(res.to_string(index=False))
    res.to_csv(os.path.join(RESULTS_DIR, "cross_source_transfer.tsv"), sep="\t", index=False)

    print(f"\nRisultati salvati in {RESULTS_DIR}/cross_source_transfer.tsv")
    print("\nLettura: transfer >> caso -> segnale biologico indipendente dal protocollo.")
    print("         transfer ~ caso  -> il segnale within-source era batch/protocollo.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
