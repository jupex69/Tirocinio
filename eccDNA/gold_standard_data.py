"""Costruzione del dataset GOLD-STANDARD (confounder-controlled) e diagnostica lunghezza.

Principio: tenere COSTANTI le variabili tecniche invece di sperare che il modello
le ignori. Concretamente:
 - strato omogeneo per protocollo+studio: source_db=CircleBaseV2 AND method=Circle_seq
   (elimina i confondenti metodo, database, e il gc-via-metodo)
 - igiene etichette: fusione dei doppioni in tessuto canonico (case-insensitive),
   unificazione dei sani, scarto di 'Multiple Diseases'/label non mappabili e delle
   classi troppo piccole
 - lunghezza: qui la si DIAGNOSTICA e (per il binario) la si MATCHA per quantili
 - split per cluster genomico (gia' privo di leakage) con riparazione classi rare

Espone:
 - load_stratum_table(): id, gruppo (tessuto|Healthy), is_disease, length, split
 - length_diagnostics(): quanto la lunghezza da sola separa sano/malato e i tessuti
 - length_matched_binary(): campione sano/malato con stessa distribuzione di lunghezza
 - multiclass_table(): solo tessuti, con tetto e split riparato
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

META = "data/processed/eccdna_disease_detection_metadata.tsv"
CHUNK = 250000
SOURCE = "CircleBaseV2"
COLLECT_CAP = 20000
CAP_PER_CLASS = 3000
MIN_CLASS = 200
SEED = 42

TISSUE_KW = {"gastric": "Stomach", "stomach": "Stomach", "colorect": "Colorectal",
             "prostate": "Prostate", "cataract": "Cataract", "hypophar": "Hypopharynx",
             "glioblastoma": "Glioblastoma", "ovarian": "Ovarian", "ovary": "Ovarian",
             "breast": "Breast"}


def canon_tissue(disease):
    d = str(disease).lower()
    for kw, t in TISSUE_KW.items():
        if kw in d:
            return t
    return None


def load_stratum_table():
    parts = []
    for ch in pd.read_csv(META, sep="\t",
                          usecols=["id", "disease", "disease_binary_name", "source_db",
                                   "method", "length", "split_cluster"],
                          chunksize=CHUNK, low_memory=False):
        ch = ch[ch["source_db"] == SOURCE]
        ch = ch[ch["method"].astype(str).str.lower().str.contains("circle")]
        healthy = ch["disease_binary_name"].astype(str).str.lower() == "healthy"
        grp = np.where(healthy, "Healthy", ch["disease"].map(canon_tissue))
        ch = ch.assign(gruppo=grp, is_disease=np.where(healthy, 0, 1))
        ch = ch[ch["gruppo"].notna()]
        parts.append(ch[["id", "gruppo", "is_disease", "length", "split_cluster"]]
                     .rename(columns={"split_cluster": "split"}))
    df = pd.concat(parts, ignore_index=True)
    df["id"] = df["id"].astype(str)
    df["split"] = df["split"].astype(str)
    # tetto di raccolta per gruppo (bilancia le classi giganti prima del resto)
    df = df.groupby("gruppo", group_keys=False).head(COLLECT_CAP)
    keep = df["gruppo"].value_counts()
    keep = keep[keep >= MIN_CLASS].index
    return df[df["gruppo"].isin(keep)].reset_index(drop=True)


def _repair_split(df, seed=SEED):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df.loc[~df["split"].isin(["train", "val", "test"]), "split"] = np.nan
    for g, grp in df.groupby("gruppo"):
        c = grp["split"].value_counts()
        if grp["split"].isna().any() or c.get("val", 0) == 0 or c.get("test", 0) == 0:
            idx = grp.index.to_numpy(); rng.shuffle(idx)
            n = len(idx); nte = max(1, int(0.15*n)); nva = max(1, int(0.15*n))
            df.loc[idx[:nte], "split"] = "test"
            df.loc[idx[nte:nte+nva], "split"] = "val"
            df.loc[idx[nte+nva:], "split"] = "train"
    return df


def _cap(df, seed=SEED):
    parts = []
    for (g, sp), grp in df.groupby(["gruppo", "split"]):
        cap = CAP_PER_CLASS if sp == "train" else max(1, CAP_PER_CLASS // 4)
        parts.append(grp.sample(n=min(len(grp), cap), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def multiclass_table(df):
    d = _repair_split(df[df["is_disease"] == 1].copy())
    return _cap(d)


def length_matched_binary(df, n_bins=20, seed=SEED):
    d = df.copy()
    d["lb"] = pd.qcut(d["length"], n_bins, duplicates="drop")
    parts = []
    for _, g in d.groupby("lb", observed=True):
        h = g[g["is_disease"] == 0]; s = g[g["is_disease"] == 1]
        k = min(len(h), len(s))
        if k > 0:
            parts.append(h.sample(k, random_state=seed))
            parts.append(s.sample(k, random_state=seed))
    out = pd.concat(parts, ignore_index=True).drop(columns="lb")
    return _repair_split(out)


def length_matched_multiclass(df, min_n=2000, n_bins=10, seed=SEED):
    """Tessuti con la STESSA distribuzione di lunghezza (per bin di quantile,
    campionamento uguale tra le classi presenti). Solo tessuti ben popolati
    (>= min_n) per non decimare il campione."""
    d = df[df["is_disease"] == 1].copy()
    keep = d["gruppo"].value_counts()
    keep = keep[keep >= min_n].index
    d = d[d["gruppo"].isin(keep)]
    tissues = sorted(d["gruppo"].unique())
    d["lb"] = pd.qcut(d["length"], n_bins, duplicates="drop")
    parts = []
    for _, g in d.groupby("lb", observed=True):
        counts = g["gruppo"].value_counts()
        if len(counts) == len(tissues):
            k = int(counts.min())
            for t in tissues:
                parts.append(g[g["gruppo"] == t].sample(k, random_state=seed))
    out = pd.concat(parts, ignore_index=True).drop(columns="lb")
    return _repair_split(out)


def length_diagnostics(df):
    print("--- DIAGNOSTICA LUNGHEZZA (dentro lo strato pulito) ---")
    print("\nLunghezza mediana per gruppo:")
    print(df.groupby("gruppo")["length"].median().round(0).sort_values().to_string())

    hd = df.groupby("is_disease")["length"].median()
    print(f"\nSano vs malato (mediana): sano={hd.get(0, float('nan')):.0f}  malato={hd.get(1, float('nan')):.0f}")

    # quanto la LUNGHEZZA DA SOLA separa sano/malato (AUC)
    d = df.dropna(subset=["length"])
    Xtr, Xte, ytr, yte = train_test_split(d[["length"]].to_numpy(), d["is_disease"].to_numpy(),
                                          test_size=0.3, random_state=SEED, stratify=d["is_disease"])
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    print(f"\nAUC binario usando SOLO la lunghezza: {auc:.3f}  (0.5=lunghezza non separa)")

    # quanto la LUNGHEZZA DA SOLA separa i TESSUTI (accuratezza)
    md = df[df["is_disease"] == 1].dropna(subset=["length"])
    tess = sorted(md["gruppo"].unique())
    t2i = {t: i for i, t in enumerate(tess)}
    y = md["gruppo"].map(t2i).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(md[["length"]].to_numpy(), y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    acc = accuracy_score(yte, rf.predict(Xte))
    print(f"Accuratezza multiclasse ({len(tess)} tessuti) usando SOLO la lunghezza: {acc:.3f}  (chance={1/len(tess):.3f})")
    print("  -> se alta, la lunghezza va matchata/controllata anche nella multiclasse.")


def main():
    df = load_stratum_table()
    print("Gruppi tenuti (>= MIN_CLASS) e numerosita' grezza nello strato:")
    print(df["gruppo"].value_counts().to_string())
    print(f"\nTotale righe strato (dopo collect cap {COLLECT_CAP}): {len(df)}\n")
    length_diagnostics(df)

    print("\n--- Tabelle finali ---")
    mc = multiclass_table(df)
    print("MULTICLASSE (tessuti) per split:")
    print(pd.crosstab(mc["gruppo"], mc["split"]).to_string())
    bn = length_matched_binary(df)
    print("\nBINARIO length-matched per classe e split:")
    print(pd.crosstab(bn["is_disease"], bn["split"]).to_string())
    print(f"  lunghezza mediana dopo matching: sano={bn[bn.is_disease==0]['length'].median():.0f} "
          f"malato={bn[bn.is_disease==1]['length'].median():.0f}")


if __name__ == "__main__":
    main()
