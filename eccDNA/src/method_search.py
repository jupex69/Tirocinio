"""RICERCA DEL METODO PIU' RIGOROSO per individuare malattie dall'eccDNA in modo
indipendente dallo studio/laboratorio (valido per dati futuri, non ottimizzato
per fare numeri alti su questo dataset).

Principio guida: l'UNICA valutazione onesta e' leave-one-study-out (train su un
laboratorio, test su un altro mai visto). Tutto il resto (split casuale) e'
gonfiato dalla memorizzazione dello studio (dimostrato in
experiment_merged_split.py).

Griglia di ablation sulle leve del metodo, misurata come rilevabilita' per
tessuto (one-vs-rest, AUC cross-lab in ENTRAMBE le direzioni; robusto = min):
 1. RAPPRESENTAZIONE: 10 descrittori | spettro 3-mer (64) | tutte (74)
 2. CORREZIONE BATCH: nessuna | z-score PER-STUDIO (rimuove shift di laboratorio)
 3. PESI: nessuno | bilanciati (le classi/studi minori pesano di piu')
 4. CLASSIFICATORE: GBM | Logistica | Random Forest

Obiettivo: la configurazione che massimizza il numero di tessuti rilevabili
(AUC_min >= 0.60) e l'AUC media robusta. I risultati completi vanno in results/.

Usa le feature in cache di experiment_cross_source.py (6 tessuti cross-source).
"""

import os
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from experiment_cross_source import load_cross_base, get_features, RESULTS_DIR
from train_siamese_multiclass import KMER_COLS
from eccdna_utils import DESCRIPTOR_NAMES

warnings.filterwarnings("ignore")
SEED = 42
FEATSETS = {"desc10": DESCRIPTOR_NAMES, "kmer64": KMER_COLS, "all74": KMER_COLS + DESCRIPTOR_NAMES}


def make_clf(kind):
    if kind == "GBM":
        return HistGradientBoostingClassifier(random_state=SEED)
    if kind == "LogReg":
        return LogisticRegression(max_iter=2000, C=1.0)
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)


def per_study_zscore(mc, cols):
    """Rimuove media/scala PER DATABASE (batch correction non supervisionata:
    a test si usano le statistiche del database di test, ammesso)."""
    X = mc[cols].to_numpy(np.float64).copy()
    for s in mc["source_db"].unique():
        m = (mc["source_db"] == s).to_numpy()
        mu, sd = X[m].mean(0), X[m].std(0); sd[sd == 0] = 1.0
        X[m] = (X[m] - mu) / sd
    return X


def cross_auc(Xall, mc, tissue, A, B, clf_kind, weight):
    ia = (mc["source_db"] == A).to_numpy(); ib = (mc["source_db"] == B).to_numpy()
    ya = (mc.loc[ia, "tessuto"] == tissue).astype(int).to_numpy()
    yb = (mc.loc[ib, "tessuto"] == tissue).astype(int).to_numpy()
    Xa, Xb = Xall[ia], Xall[ib]
    if ya.sum() == 0 or yb.sum() == 0 or ya.sum() == len(ya):
        return np.nan
    clf = make_clf(clf_kind)
    sw = compute_sample_weight("balanced", ya) if weight == "balanced" else None
    clf.fit(Xa, ya, sample_weight=sw)
    p = clf.predict_proba(Xb)[:, 1]
    return roc_auc_score(yb, p)


def main():
    mc = get_features(load_cross_base())
    tissues = sorted(mc["tessuto"].unique())
    print(f"Tessuti cross-source: {tissues}  (n={len(tissues)})\n")

    # pre-standardizzazione globale (per 'none') e per-studio (per 'perstudy')
    # per 'none' standardizziamo col train dentro la funzione? per semplicita' e
    # coerenza usiamo z-score globale per 'none' e per-studio per 'perstudy'
    def global_z(cols):
        X = mc[cols].to_numpy(np.float64)
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
        return (X - mu) / sd

    grid = list(itertools.product(FEATSETS, ["none", "perstudy"], ["none", "balanced"], ["GBM", "LogReg", "RF"]))
    print(f"Configurazioni: {len(grid)}  (feature x batch x pesi x classificatore)\n")

    rows = []
    for feat, batch, weight, clf in grid:
        cols = FEATSETS[feat]
        Xall = per_study_zscore(mc, cols) if batch == "perstudy" else global_z(cols)
        aucs = {}
        for t in tissues:
            ab = cross_auc(Xall, mc, t, "eccDNABase", "CircleBaseV2", clf, weight)
            ba = cross_auc(Xall, mc, t, "CircleBaseV2", "eccDNABase", clf, weight)
            aucs[t] = min(ab, ba)  # robusto = peggiore direzione
        vals = np.array(list(aucs.values()))
        rows.append({"feature": feat, "batch": batch, "pesi": weight, "clf": clf,
                     "AUC_min_media": round(float(np.nanmean(vals)), 3),
                     "n_rilevabili_0.60": int((vals >= 0.60).sum()),
                     "n_deboli_0.55": int((vals >= 0.55).sum()),
                     **{f"AUC_{t}": round(float(aucs[t]), 3) for t in tissues}})
    res = pd.DataFrame(rows).sort_values(["n_rilevabili_0.60", "AUC_min_media"], ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "method_search_grid.tsv"), sep="\t", index=False)

    cols_show = ["feature", "batch", "pesi", "clf", "AUC_min_media", "n_rilevabili_0.60", "n_deboli_0.55"]
    pd.set_option("display.width", 200)
    print("=== TOP 15 configurazioni (per n. tessuti rilevabili robusti, poi AUC media) ===")
    print(res[cols_show].head(15).to_string(index=False))
    print("\n=== Le 5 peggiori (per contrasto) ===")
    print(res[cols_show].tail(5).to_string(index=False))

    best = res.iloc[0]
    print(f"\n>>> MIGLIORE: feature={best['feature']} batch={best['batch']} pesi={best['pesi']} clf={best['clf']}")
    print(f">>> rileva {best['n_rilevabili_0.60']} tessuti (AUC_min>=0.60), AUC media robusta {best['AUC_min_media']}")
    tcols = [c for c in res.columns if c.startswith("AUC_") and c != "AUC_min_media"]
    detail = best[tcols].sort_values(ascending=False)
    print(">>> per tessuto (AUC robusta, min tra le due direzioni):")
    print(detail.to_string())
    print(f"\nGriglia completa salvata in {RESULTS_DIR}/method_search_grid.tsv")


if __name__ == "__main__":
    main()
