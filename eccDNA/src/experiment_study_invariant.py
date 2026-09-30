"""Due tecniche NON ancora provate per ridurre l'influenza dello studio, sui
tessuti cross-source (le uniche classi presenti in >=2 studi, dove ha senso):

 1. NORMALIZZAZIONE PER RANGO dentro ogni studio: ogni feature -> rango
    percentile all'interno del suo database. Rimuove TUTTI gli spostamenti
    monotoni di studio (non solo media/scala come lo z-score gia' provato).
 2. SELEZIONE DI FEATURE STUDY-INVARIANTI: si tengono le feature con alto potere
    discriminante sulla CLASSE (F rispetto al tessuto) e basso sullo STUDIO (F
    rispetto al database) -> rapporto F_classe/F_studio.

Si valuta la certificazione cross-lab per tessuto (one-vs-rest, AUC robusta = min
tra le due direzioni), confrontando: baseline (74 raw), rango, selezione, e
rango+selezione. Domanda: si rileva piu' di un tessuto (Hypopharynx)?
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from experiment_cross_source import load_cross_base, get_features, RESULTS_DIR
from train_siamese_multiclass import FEATURE_COLS

SEED = 42
mc_g = None
tissues_g = None


def rank_within_db(mc, cols):
    X = mc[cols].copy()
    for s in mc["source_db"].unique():
        m = mc["source_db"] == s
        X.loc[m, cols] = X.loc[m, cols].rank(pct=True)
    return X.to_numpy(np.float64)


def study_invariant_cols(mc, cols, k=30):
    ycls = mc["tessuto"].astype("category").cat.codes.to_numpy()
    ystd = (mc["source_db"] == "eccDNABase").astype(int).to_numpy()
    Fc, _ = f_classif(mc[cols].to_numpy(np.float64), ycls)
    Fs, _ = f_classif(mc[cols].to_numpy(np.float64), ystd)
    ratio = np.nan_to_num(Fc) / (np.nan_to_num(Fs) + 1e-9)
    order = np.argsort(-ratio)
    return [cols[i] for i in order[:k]]


def cross_auc(X, mc, t, A, B):
    ia = (mc["source_db"] == A).to_numpy(); ib = (mc["source_db"] == B).to_numpy()
    ya = (mc.loc[ia, "tessuto"] == t).astype(int).to_numpy()
    yb = (mc.loc[ib, "tessuto"] == t).astype(int).to_numpy()
    if ya.sum() == 0 or yb.sum() == 0:
        return np.nan
    clf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1)
    clf.fit(X[ia], ya, sample_weight=compute_sample_weight("balanced", ya))
    return roc_auc_score(yb, clf.predict_proba(X[ib])[:, 1])


def evaluate(X, name):
    aucs = {}
    for t in tissues_g:
        ab = cross_auc(X, mc_g, t, "eccDNABase", "CircleBaseV2")
        ba = cross_auc(X, mc_g, t, "CircleBaseV2", "eccDNABase")
        aucs[t] = min(ab, ba)
    vals = np.array(list(aucs.values()))
    print(f"\n{name}:  AUC_min media={np.nanmean(vals):.3f}  rilevabili(>=0.60)={int((vals>=0.60).sum())}")
    for t, v in sorted(aucs.items(), key=lambda x: -x[1]):
        print(f"    {t:14s} {v:.3f}")
    return aucs


def main():
    global mc_g, tissues_g
    mc_g = get_features(load_cross_base())
    tissues_g = sorted(mc_g["tessuto"].unique())
    print(f"Tessuti cross-source: {tissues_g}\n")

    def zscore(cols):
        X = mc_g[cols].to_numpy(np.float64); m, s = X.mean(0), X.std(0); s[s == 0] = 1
        return (X - m) / s

    print("=== Confronto tecniche anti-studio (certificazione cross-lab) ===")
    a = evaluate(zscore(FEATURE_COLS), "baseline (74 raw z-score)")
    b = evaluate(rank_within_db(mc_g, FEATURE_COLS), "rango-per-studio (74)")
    sel = study_invariant_cols(mc_g, FEATURE_COLS, k=30)
    c = evaluate(zscore(sel), "selezione study-invariante (top 30)")
    # rango + selezione
    Xr = pd.DataFrame(rank_within_db(mc_g, FEATURE_COLS), columns=FEATURE_COLS)
    d = evaluate(Xr[sel].to_numpy(np.float64), "rango + selezione")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for name, res in [("baseline", a), ("rango", b), ("selezione", c), ("rango+selezione", d)]:
        vals = np.array(list(res.values()))
        rows.append({"tecnica": name, "AUC_min_media": round(float(np.nanmean(vals)), 3),
                     "n_rilevabili": int((vals >= 0.60).sum()), "Hypopharynx": round(res.get("Hypopharynx", np.nan), 3)})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "study_invariant.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/study_invariant.tsv")
    print(">>> Se n_rilevabili non sale oltre la baseline, nemmeno queste tecniche recuperano lo studio.")


if __name__ == "__main__":
    main()
