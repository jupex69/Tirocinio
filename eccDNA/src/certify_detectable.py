"""CERTIFICAZIONE del set di malattie rilevabili cross-studio, col metodo migliore
trovato dalla ricerca (method_search.py): RandomForest + pesi bilanciati + 74
feature, valutazione leave-one-database-out.

Per ogni tessuto e ogni direzione (ecc->CB2, CB2->ecc) si calcola l'AUC one-vs-
rest con IC 95% bootstrap sull'insieme di test (2000 ricampionamenti). Un tessuto
e' CERTIFICATO rilevabile se il limite inferiore dell'IC supera 0.5 in ENTRAMBE
le direzioni (segnale sopra il caso, statisticamente, indipendente dal laboratorio).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from experiment_cross_source import load_cross_base, get_features, RESULTS_DIR
from train_siamese_multiclass import FEATURE_COLS

SEED = 42
B = 2000


def _std(tr, te):
    m, s = tr.mean(0), tr.std(0); s[s == 0] = 1.0
    return (tr - m) / s, (te - m) / s


def boot_auc(y, p, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def direction(mc, tissue, A, Bs):
    a = mc[mc.source_db == A]; b = mc[mc.source_db == Bs]
    Xa = a[FEATURE_COLS].to_numpy(np.float64); ya = (a["tessuto"] == tissue).astype(int).to_numpy()
    Xb = b[FEATURE_COLS].to_numpy(np.float64); yb = (b["tessuto"] == tissue).astype(int).to_numpy()
    Xa, Xb = _std(Xa, Xb)
    clf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1)
    clf.fit(Xa, ya, sample_weight=compute_sample_weight("balanced", ya))
    p = clf.predict_proba(Xb)[:, 1]
    auc = roc_auc_score(yb, p); lo, hi = boot_auc(yb, p)
    return auc, lo, hi


def main():
    mc = get_features(load_cross_base())
    tissues = sorted(mc["tessuto"].unique())
    print("Metodo: RandomForest + pesi bilanciati + 74 feature, leave-one-DB-out\n")
    rows = []
    for t in tissues:
        a1, l1, h1 = direction(mc, t, "eccDNABase", "CircleBaseV2")
        a2, l2, h2 = direction(mc, t, "CircleBaseV2", "eccDNABase")
        cert = (l1 > 0.5) and (l2 > 0.5)
        rows.append({"tessuto": t,
                     "AUC_ecc->CB2": round(a1, 3), "IC_ecc": f"[{l1:.3f},{h1:.3f}]",
                     "AUC_CB2->ecc": round(a2, 3), "IC_CB2": f"[{l2:.3f},{h2:.3f}]",
                     "IC_min_basso": round(min(l1, l2), 3),
                     "CERTIFICATO": "SI" if cert else "no"})
    res = pd.DataFrame(rows).sort_values("IC_min_basso", ascending=False)
    print(res.to_string(index=False))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "certified_detectable.tsv"), sep="\t", index=False)
    n = (res["CERTIFICATO"] == "SI").sum()
    print(f"\n>>> MALATTIE CERTIFICATE rilevabili cross-studio (IC 95% > 0.5 in entrambe le direzioni): {n}/{len(res)}")
    print(">>> " + ", ".join(res[res.CERTIFICATO == "SI"]["tessuto"].tolist()))
    print(f"\nSalvato in {RESULTS_DIR}/certified_detectable.tsv")


if __name__ == "__main__":
    main()
