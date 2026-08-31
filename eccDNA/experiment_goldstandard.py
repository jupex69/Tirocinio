"""Esperimento GOLD-STANDARD (confounder-controlled): binario e multiclasse dentro
lo strato omogeneo (CircleBaseV2 + Circle_seq), con LENGTH-MATCHING e baseline
solo-lunghezza. E' la versione scientificamente onesta di entrambi i task.

Per ogni task si confronta il modello 74-feature (spettro 3-mer + 10 descrittori,
che NON contengono la lunghezza grezza) con una baseline che usa SOLO la lunghezza:
 - se dopo il length-matching la baseline solo-lunghezza scende al caso (conferma
   che il matching ha funzionato) e il 74-feature resta sopra il caso -> esiste
   segnale di COMPOSIZIONE reale, indipendente da lunghezza/metodo/studio.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

from eccdna_utils import read_fasta_stream, compute_sequence_descriptors
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from train_multiclass import (
    standardize, train_prototypical, predict_prototypical,
    train_softmax, predict_softmax, evaluate, SEED,
)
from gold_standard_data import (
    load_stratum_table, length_matched_binary, length_matched_multiclass, length_diagnostics,
)

FASTA = "data/processed/eccdna_disease_detection.body.fa"


def features_for(ids):
    feats = {}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=set(ids)):
        if "N" in seq or len(seq) < 3:
            continue
        f = kmer_spectrum(seq)
        f.update(compute_sequence_descriptors(seq))
        feats[sid] = f
    return pd.DataFrame.from_dict(feats, orient="index")


def attach(df, ftab):
    return df.set_index("id").join(ftab, how="inner").reset_index().dropna(subset=FEATURE_COLS)


def main():
    print("--- Caricamento strato e costruzione tabelle gold-standard ---")
    base = load_stratum_table()
    length_diagnostics(base)

    bn = length_matched_binary(base)
    mc = length_matched_multiclass(base)
    print("\nBINARIO length-matched:", dict(bn["is_disease"].value_counts()))
    print("MULTICLASSE length-matched per tessuto:")
    print(mc["gruppo"].value_counts().to_string())

    all_ids = set(bn["id"]) | set(mc["id"])
    print(f"\nLettura FASTA e calcolo feature per {len(all_ids)} id...")
    ftab = features_for(all_ids)
    print(f"Feature calcolate per {len(ftab)} sequenze.")

    bn = attach(bn, ftab); mc = attach(mc, ftab)

    # ===================== BINARIO =====================
    print("\n" + "="*60)
    print("A) BINARIO sano/malato (strato + length-matched)")
    tr, va, te = bn[bn.split == "train"], bn[bn.split == "val"], bn[bn.split == "test"]
    Xtr, Xte = standardize(tr[FEATURE_COLS].to_numpy(np.float32), te[FEATURE_COLS].to_numpy(np.float32))
    ytr, yte = tr["is_disease"].to_numpy(), te["is_disease"].to_numpy()
    gbm = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    print(f"  74-feature  AUC  GBM={roc_auc_score(yte, gbm.predict_proba(Xte)[:,1]):.3f}  "
          f"RF={roc_auc_score(yte, rf.predict_proba(Xte)[:,1]):.3f}")
    # baseline solo-lunghezza (deve stare ~0.5 dopo il matching)
    lrf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(
        tr[["length"]].to_numpy(), ytr)
    print(f"  solo-lunghezza AUC = {roc_auc_score(yte, lrf.predict_proba(te[['length']].to_numpy())[:,1]):.3f}  (deve ~0.5)")

    # ===================== MULTICLASSE =====================
    print("\n" + "="*60)
    print("B) MULTICLASSE tessuto (strato + length-matched)")
    tessuti = sorted(mc["gruppo"].unique())
    t2i = {t: i for i, t in enumerate(tessuti)}
    n_cls = len(tessuti)
    tr, va, te = mc[mc.split == "train"], mc[mc.split == "val"], mc[mc.split == "test"]
    Xtr, Xva, Xte = standardize(tr[FEATURE_COLS].to_numpy(np.float32),
                                va[FEATURE_COLS].to_numpy(np.float32),
                                te[FEATURE_COLS].to_numpy(np.float32))
    ytr = tr["gruppo"].map(t2i).to_numpy(); yva = va["gruppo"].map(t2i).to_numpy(); yte = te["gruppo"].map(t2i).to_numpy()
    print(f"  tessuti={tessuti}  chance={1/n_cls:.3f}")

    enc, _ = train_prototypical(Xtr, ytr, Xva, yva, n_cls, cosine=True, seed=SEED, select_metric="balanced")
    evaluate("  Siamese 74", yte, predict_prototypical(enc, Xtr, ytr, Xte, n_cls, cosine=True), n_cls)
    sm = train_softmax(Xtr, ytr, Xva, yva, n_cls, seed=SEED, balanced=True)
    evaluate("  Softmax bilanciato", yte, predict_softmax(sm, Xte), n_cls)
    # baseline solo-lunghezza (deve ~chance dopo il matching)
    lrf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(
        tr[["length"]].to_numpy(), ytr)
    acc_l = accuracy_score(yte, lrf.predict(te[["length"]].to_numpy()))
    print(f"  solo-lunghezza  acc={acc_l:.3f}  (deve ~chance={1/n_cls:.3f})")

    print("\n--- LETTURA ---")
    print("  Se solo-lunghezza ~ caso (matching ok) e 74-feature > caso => segnale di")
    print("  composizione reale, indipendente da lunghezza/metodo/studio. Quello e' il")
    print("  numero onesto di quanto la SEQUENZA sa dire sul tessuto.")


if __name__ == "__main__":
    main()
