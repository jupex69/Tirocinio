"""Sistema a due stadi (cascade) dentro lo strato gold-standard.

STADIO 1 (binario): sano vs malato, length-matched -> risposta affidabile.
STADIO 2 (multiclasse, solo sui malati): 5 tessuti, con ASTENSIONE. Se la
confidenza (max probabilita' softmax) supera una soglia -> assegna il tessuto;
altrimenti resta "malato (tessuto non determinato)". Si mostra la curva
precisione/copertura: quanto sale l'accuratezza su cio' che il modello risponde
man mano che si astiene di piu'.

Idea: il sistema non e' mai peggio del binario, e aggiunge il tessuto solo quando
ne e' sicuro (tipicamente i tessuti biologicamente distinti: colon-retto, cataratta).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from eccdna_utils import read_fasta_stream, compute_sequence_descriptors
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from train_multiclass import standardize, train_softmax, predict_softmax, SEED
from gold_standard_data import load_stratum_table, length_matched_binary, length_matched_multiclass

FASTA = "data/processed/eccdna_disease_detection.body.fa"


def main():
    print("--- Sistema a due stadi (binario -> tessuto con astensione) ---")
    base = load_stratum_table()
    bn = length_matched_binary(base)
    mc = length_matched_multiclass(base)

    ids = set(bn["id"]) | set(mc["id"])
    print(f"Lettura FASTA e calcolo feature per {len(ids)} sequenze...")
    feats = {}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
        if "N" not in seq and len(seq) >= 4:
            f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq)); feats[sid] = f
    ft = pd.DataFrame.from_dict(feats, orient="index")
    bn = bn.set_index("id").join(ft, how="inner").reset_index().dropna(subset=FEATURE_COLS)
    mc = mc.set_index("id").join(ft, how="inner").reset_index().dropna(subset=FEATURE_COLS)

    # ---------------- STADIO 1: binario sano/malato ----------------
    print("\n=== STADIO 1: sano vs malato (length-matched) ===")
    btr, bte = bn[bn.split == "train"], bn[bn.split == "test"]
    Xtr, Xte = standardize(btr[FEATURE_COLS].to_numpy(np.float32), bte[FEATURE_COLS].to_numpy(np.float32))
    ytr, yte = btr["is_disease"].to_numpy(), bte["is_disease"].to_numpy()
    gbm = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
    p1 = gbm.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p1); acc1 = accuracy_score(yte, (p1 >= 0.5).astype(int))
    print(f"  ROC-AUC={auc:.3f}   accuratezza={acc1:.3f}   (questo e' il 'pavimento' affidabile del sistema)")

    # ---------------- STADIO 2: tessuto con astensione ----------------
    print("\n=== STADIO 2: quale tessuto (5 classi) con ASTENSIONE ===")
    tess = sorted(mc["gruppo"].unique()); t2i = {t: i for i, t in enumerate(tess)}
    n = len(tess); chance = 1 / n
    mtr, mva, mte = mc[mc.split == "train"], mc[mc.split == "val"], mc[mc.split == "test"]
    Xmtr, Xmva, Xmte = standardize(mtr[FEATURE_COLS].to_numpy(np.float32),
                                   mva[FEATURE_COLS].to_numpy(np.float32),
                                   mte[FEATURE_COLS].to_numpy(np.float32))
    ymtr = mtr["gruppo"].map(t2i).to_numpy(); ymva = mva["gruppo"].map(t2i).to_numpy(); ymte = mte["gruppo"].map(t2i).to_numpy()
    sm = train_softmax(Xmtr, ymtr, Xmva, ymva, n, seed=SEED, balanced=True)
    proba = predict_softmax(sm, Xmte)
    conf = proba.max(1); pred = proba.argmax(1); correct = (pred == ymte)

    print(f"  Senza astensione: accuratezza={correct.mean():.3f} su tutti (caso={chance:.3f})\n")
    print("  Curva precisione/copertura (si risponde solo ai piu' sicuri):")
    print(f"  {'copertura':>10} {'accuratezza-su-risposti':>24} {'tessuti risposti (top-2)':>34}")
    order = np.argsort(-conf)
    for cov in [1.00, 0.75, 0.50, 0.25]:
        k = max(1, int(cov * len(conf))); idx = order[:k]
        acc_cov = correct[idx].mean()
        top = pd.Series([tess[i] for i in pred[idx]]).value_counts(normalize=True).head(2)
        top_s = ", ".join(f"{t} {f*100:.0f}%" for t, f in top.items())
        print(f"  {cov*100:>8.0f}% {acc_cov:>22.3f}   {top_s:>34}")

    # ---------------- vista di sistema ----------------
    print("\n=== VISTA DI SISTEMA (esempio a copertura ~50%) ===")
    k = len(conf) // 2; idx = order[:k]
    thr = conf[order[k - 1]]
    print(f"  Soglia di confidenza ~{thr:.2f}:")
    print(f"   - lo Stadio 1 dice 'malata' in modo affidabile (AUC {auc:.3f});")
    print(f"   - per il ~50% dei malati piu' chiari, lo Stadio 2 nomina il tessuto con "
          f"accuratezza {correct[idx].mean():.3f} (vs {chance:.3f} del caso);")
    print(f"   - per l'altro ~50% il sistema resta a 'malata (tessuto non determinato)'.")
    print("  => il sistema non e' mai peggio del binario e aggiunge il tessuto solo quando e' sicuro.")


if __name__ == "__main__":
    main()
