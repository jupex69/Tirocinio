"""Certificazione rigorosa dell'UNICO task 'malato vs sano within-tessuto'
controllabile: Cuore/Circle-seq (Dilated Cardiomyopathy vs cuore sano, eccDNABase).

(Il task Cervello e' scartato: sano e Glioblastoma hanno lunghezze che non si
sovrappongono -> non e' length-controllabile.)

Matching stretto per lunghezza (caliper 5%), split STRATIFICATO per lunghezza
(il test resta bilanciato in lunghezza), IC 95% bootstrap sull'AUC malato-vs-sano
del modello migliore E sul controllo solo-lunghezza. Verdetto:
 - AUC malattia con IC che esclude 0.5  E  controllo-lunghezza con IC che include
   0.5  -> segnale di malattia reale (per quanto debole);
 - altrimenti -> indistinguibile dal caso / dal residuo di lunghezza.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from experiment_within_tissue import load_task_ids, get_features, FEATURE_COLS
from train_multiclass import standardize, train_prototypical, predict_prototypical, SEED
from experiment_losses_fewshot import train_metric_encoder

B = 2000


def caliper_match(df, caliper=0.05, seed=SEED):
    pos = df[df.y == 1]; neg = df[df.y == 0]
    minority, majority = (neg, pos) if len(neg) <= len(pos) else (pos, neg)
    maj = majority.sort_values("length").reset_index(drop=True)
    ml = maj["length"].to_numpy(np.float64); used = np.zeros(len(maj), bool)
    km, kM = [], []
    for _, r in minority.iterrows():
        d = np.abs(ml - r["length"]); d[used] = np.inf; j = int(d.argmin())
        if d[j] <= caliper * max(r["length"], 1):
            used[j] = True; km.append(r); kM.append(maj.iloc[j])
    return pd.concat([pd.DataFrame(km), pd.DataFrame(kM)], ignore_index=True)


def strat_split(df, seed=SEED):
    """Split train/val/test stratificato per (classe, quintile di lunghezza),
    cosi' il test e' bilanciato sia in classe sia in lunghezza."""
    rng = np.random.default_rng(seed); df = df.copy()
    df["lq"] = pd.qcut(df["length"], 5, duplicates="drop").astype(str)
    df["split"] = "train"
    for _, g in df.groupby(["y", "lq"]):
        idx = g.index.to_numpy(); rng.shuffle(idx)
        n = len(idx); nte = max(1, int(0.2 * n)); nva = max(1, int(0.15 * n))
        df.loc[idx[:nte], "split"] = "test"; df.loc[idx[nte:nte + nva], "split"] = "val"
    return df


def boot_ci(y, p, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            v.append(roc_auc_score(y[i], p[i]))
    return np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    tasks = load_task_ids()
    ftab = get_features(tasks)
    df = tasks[1].set_index("id").join(ftab, how="inner").reset_index().dropna(subset=FEATURE_COLS)  # cuore
    df = caliper_match(df); df = strat_split(df)
    tr, va, te = df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]
    print("Cuore/Circle-seq — Dilated Cardiomyopathy vs cuore sano (eccDNABase)")
    print(f"coppie matchate: {int((df.y==1).sum())}+{int((df.y==0).sum())}  "
          f"(train={len(tr)} val={len(va)} test={len(te)})")
    print(f"lunghezza mediana test: malati={te[te.y==1]['length'].median():.0f} sani={te[te.y==0]['length'].median():.0f}\n")

    Xtr = tr[FEATURE_COLS].to_numpy(np.float32); Xva = va[FEATURE_COLS].to_numpy(np.float32); Xte = te[FEATURE_COLS].to_numpy(np.float32)
    ytr = tr.y.to_numpy(); yva = va.y.to_numpy(); yte = te.y.to_numpy()
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)

    # controllo solo-lunghezza
    rfl = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(tr[["length"]], ytr)
    pl = rfl.predict_proba(te[["length"]])[:, 1]
    m, lo, hi = boot_ci(yte, pl)
    print(f"CONTROLLO solo-lunghezza : AUC={m:.3f}  IC95%=[{lo:.3f},{hi:.3f}]  {'~0.5 OK' if lo <= 0.5 <= hi else 'ANCORA confuso'}")

    # RandomForest 74
    rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    pr = rf.predict_proba(Xte)[:, 1]
    m, lo, hi = boot_ci(yte, pr)
    print(f"RandomForest 74          : AUC={m:.3f}  IC95%=[{lo:.3f},{hi:.3f}]  {'> 0.5' if lo > 0.5 else 'include 0.5'}")

    # siamese triplet (il migliore tra i 3 modi sul cuore)
    enc, _ = train_metric_encoder(Xtr, ytr, Xva, yva, 2, "triplet", margin=0.3)
    ps = predict_prototypical(enc, Xtr, ytr, Xte, 2, False)[:, 1]
    m, lo, hi = boot_ci(yte, ps)
    print(f"Siamese triplet          : AUC={m:.3f}  IC95%=[{lo:.3f},{hi:.3f}]  {'> 0.5' if lo > 0.5 else 'include 0.5'}")


if __name__ == "__main__":
    main()
