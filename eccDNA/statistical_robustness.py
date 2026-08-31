"""Robustezza statistica dei risultati: intervalli di confidenza al 95% e test di
significativita', per il binario e per il multiclasse.

BINARIO: bootstrap sull'insieme di test (2000 ricampionamenti) -> IC 95% del
ROC-AUC dei modelli deterministici (GBM, RF). Verifica che l'IC escluda 0.5.

MULTICLASSE: ripetizioni su piu' ricampionamenti (seed diversi) del pipeline
gold-standard length-matched -> media, deviazione standard e IC 95% (t di Student)
di accuratezza e macro-F1, per siamese e softmax. Test di permutazione (1000
rimescolamenti delle etichette) per la significativita' contro il caso (0.20).
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from eccdna_utils import DESCRIPTOR_NAMES, read_fasta_stream, compute_sequence_descriptors
from training_data import build_balanced_splits
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from train_multiclass import (
    standardize, train_prototypical, predict_prototypical, train_softmax, predict_softmax, SEED,
)
from gold_standard_data import load_stratum_table, length_matched_multiclass

FASTA = "data/processed/eccdna_disease_detection.body.fa"


def bootstrap_ci(metric_fn, *arrays, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(arrays[0]); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric_fn(*[a[idx] for a in arrays]))
        except ValueError:
            continue
    return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def t_ci(vals):
    vals = np.asarray(vals); m, s, n = vals.mean(), vals.std(ddof=1), len(vals)
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=s / np.sqrt(n))
    return m, s, lo, hi


# ============================ BINARIO ============================
def binario():
    print("=== BINARIO: IC 95% del ROC-AUC (bootstrap sul test) ===")
    tr, va, te = build_balanced_splits()
    Xtr, ytr = tr[DESCRIPTOR_NAMES].to_numpy(np.float64), tr["y"].to_numpy()
    Xte, yte = te[DESCRIPTOR_NAMES].to_numpy(np.float64), te["y"].to_numpy()
    for nome, mdl in [("Gradient Boosting", HistGradientBoostingClassifier(random_state=SEED)),
                      ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]:
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        _, lo, hi = bootstrap_ci(roc_auc_score, yte, p)
        sig = "SI' (IC esclude 0.5)" if lo > 0.5 else "no"
        print(f"  {nome:18s} AUC={auc:.3f}  IC95%=[{lo:.3f}, {hi:.3f}]  significativo vs 0.5: {sig}")


# ============================ MULTICLASSE ============================
def multiclasse(n_rep=10):
    print(f"\n=== MULTICLASSE: {n_rep} ricampionamenti -> IC 95% (t) ===")
    base = load_stratum_table()
    # feature per tutti gli id dei tessuti candidati (calcolate una volta)
    cand = length_matched_multiclass(base, seed=SEED)  # per conoscere i tessuti tenuti
    tessuti = sorted(cand["gruppo"].unique())
    ids = set(base[base["gruppo"].isin(tessuti) & (base["is_disease"] == 1)]["id"])
    print(f"Tessuti: {tessuti}  chance={1/len(tessuti):.3f}")
    print(f"Calcolo feature per {len(ids)} sequenze (una volta)...")
    feats = {}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
        if "N" not in seq and len(seq) >= 4:
            f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq)); feats[sid] = f
    ftab = pd.DataFrame.from_dict(feats, orient="index")
    t2i = {t: i for i, t in enumerate(tessuti)}; n = len(tessuti)

    acc_s, mf1_s, acc_m, mf1_m = [], [], [], []
    last = None
    for s in range(n_rep):
        mc = length_matched_multiclass(base, seed=SEED + s)
        mc = mc.set_index("id").join(ftab, how="inner").reset_index().dropna(subset=FEATURE_COLS)
        tr, va, te = mc[mc.split == "train"], mc[mc.split == "val"], mc[mc.split == "test"]
        Xtr, Xva, Xte = standardize(tr[FEATURE_COLS].to_numpy(np.float32),
                                    va[FEATURE_COLS].to_numpy(np.float32),
                                    te[FEATURE_COLS].to_numpy(np.float32))
        ytr = tr["gruppo"].map(t2i).to_numpy(); yva = va["gruppo"].map(t2i).to_numpy(); yte = te["gruppo"].map(t2i).to_numpy()
        enc, _ = train_prototypical(Xtr, ytr, Xva, yva, n, cosine=True, seed=SEED, select_metric="balanced")
        ps = predict_prototypical(enc, Xtr, ytr, Xte, n, cosine=True).argmax(1)
        sm = train_softmax(Xtr, ytr, Xva, yva, n, seed=SEED, balanced=True)
        pm = predict_softmax(sm, Xte).argmax(1)
        acc_s.append(accuracy_score(yte, ps)); mf1_s.append(f1_score(yte, ps, average="macro", zero_division=0))
        acc_m.append(accuracy_score(yte, pm)); mf1_m.append(f1_score(yte, pm, average="macro", zero_division=0))
        last = (yte, ps)

    for nome, acc, mf1 in [("Siamese", acc_s, mf1_s), ("Softmax", acc_m, mf1_m)]:
        m, sd, lo, hi = t_ci(acc); m2, sd2, lo2, hi2 = t_ci(mf1)
        print(f"  {nome:8s} acc={m:.3f}±{sd:.3f} IC95%=[{lo:.3f},{hi:.3f}]   macro-F1={m2:.3f}±{sd2:.3f} IC95%=[{lo2:.3f},{hi2:.3f}]")

    # test di permutazione (significativita' vs caso) sull'ultima esecuzione
    yte, ps = last
    real = accuracy_score(yte, ps)
    rng = np.random.default_rng(SEED); null = []
    for _ in range(1000):
        null.append(accuracy_score(rng.permutation(yte), ps))
    p_val = (np.sum(np.array(null) >= real) + 1) / (1000 + 1)
    print(f"\n  Test di permutazione (accuratezza vs caso): reale={real:.3f}, "
          f"caso medio={np.mean(null):.3f}, p={p_val:.4f}  -> {'significativo' if p_val < 0.05 else 'non significativo'}")


def main():
    binario()
    multiclasse()


if __name__ == "__main__":
    main()
