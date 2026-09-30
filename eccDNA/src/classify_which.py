"""Quali delle malattie compattate sono DAVVERO classificabili (cross-studio)?

Si considerano le 9 malattie cross-testabili (presenti in >=2 database). Per
ciascuna si stima la rilevabilita' one-vs-rest con validazione leave-one-database-
out (train su un lab, test sull'altro, entrambe le direzioni), IC 95% bootstrap
sull'AUC, e un CONTROLLO solo-lunghezza per smascherare gli artefatti di
tipo-campione (es. cfDNA corto da siero).

Verdetto per malattia:
  CLASSIFICABILE  : AUC robusta (min tra direzioni) con IC che esclude 0.5 E
                    controllo-lunghezza non altrettanto alto (segnale non di sola lunghezza);
  ARTEFATTO LUNGHEZZA: AUC alta ma spiegata dalla lunghezza;
  NON classificabile : IC include 0.5 (al caso cross-studio).

Le 46 malattie mono-studio non sono validabili cross-studio (1 sola coorte).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from eccdna_utils import read_fasta_stream, compute_sequence_descriptors
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from full_classifier import FASTA, RESULTS_DIR, SEED

COMPACT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data/processed/eccdna_compacted_metadata.tsv")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data/processed/classify_which_features.tsv")
CROSS = ["Stomach Cancer", "Ovarian Cancer", "Prostate Cancer", "Colorectal Cancer",
         "Hypopharynx Cancer", "Primary Pulmonary Hypertension", "Glioblastoma",
         "Dilated Cardiomyopathy", "Breast Cancer"]
DBS = ["eccDNABase", "CircleBaseV2"]
CAP_POS = 800     # per (malattia target, db)
CAP_BG = 2500     # background di altre malattie per db
B = 1500


def sample_pool():
    """Positivi (9 cross) + background di altre malattie, per database."""
    pos = {(d, db): [] for d in CROSS for db in DBS}
    bg = {db: [] for db in DBS}
    for ch in pd.read_csv(COMPACT, sep="\t",
                          usecols=["id", "label", "is_disease", "source_db", "length"],
                          chunksize=400000, low_memory=False):
        ch = ch[(ch["is_disease"] == 1) & ch["source_db"].isin(DBS)]
        for (lab, db), g in ch.groupby(["label", "source_db"]):
            if lab in CROSS:
                buf = pos[(lab, db)]
                if sum(len(x) for x in buf) < CAP_POS:
                    buf.append(g.head(CAP_POS)[["id", "label", "source_db", "length"]])
            else:
                buf = bg[db]
                if sum(len(x) for x in buf) < CAP_BG:
                    buf.append(g.head(200)[["id", "label", "source_db", "length"]])
    frames = []
    for k, v in pos.items():
        if v:
            frames.append(pd.concat(v, ignore_index=True).head(CAP_POS))
    for db, v in bg.items():
        if v:
            frames.append(pd.concat(v, ignore_index=True).head(CAP_BG))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    df["id"] = df["id"].astype(str)
    return df


def get_features(df):
    need = set(df["id"]); ft = None
    if os.path.exists(CACHE):
        ft = pd.read_csv(CACHE, sep="\t", index_col=0); ft.index = ft.index.astype(str)
        need = need - set(ft.index)
    if need:
        feats = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=need):
            if "N" not in seq and len(seq) >= 4:
                f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq)); feats[sid] = f
        new = pd.DataFrame.from_dict(feats, orient="index")
        ft = new if ft is None else pd.concat([ft, new])
        ft = ft[~ft.index.duplicated()]; ft.to_csv(CACHE, sep="\t")
    return ft


def boot_ci(y, p, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            v.append(roc_auc_score(y[i], p[i]))
    return (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan, np.nan)


def one_dir(df, disease, A, B_, cols):
    a = df[df.source_db == A]; b = df[df.source_db == B_]
    Xa = a[cols].to_numpy(np.float64); ya = (a["label"] == disease).astype(int).to_numpy()
    Xb = b[cols].to_numpy(np.float64); yb = (b["label"] == disease).astype(int).to_numpy()
    if ya.sum() == 0 or yb.sum() == 0:
        return np.nan, np.nan, np.nan
    m, s = Xa.mean(0), Xa.std(0); s[s == 0] = 1.0
    clf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1)
    clf.fit((Xa - m) / s, ya, sample_weight=compute_sample_weight("balanced", ya))
    p = clf.predict_proba((Xb - m) / s)[:, 1]
    return roc_auc_score(yb, p), *boot_ci(yb, p)[1:]


def main():
    df = sample_pool()
    ft = get_features(df)
    ftr = ft.reset_index(); ftr = ftr.rename(columns={ftr.columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    df = df.merge(ftr, on="id", how="inner").dropna(subset=FEATURE_COLS)
    print(f"Pool: {len(df)} sequenze  (positivi 9 malattie + background)\n")

    rows = []
    for d in CROSS:
        a1, l1, h1 = one_dir(df, d, "eccDNABase", "CircleBaseV2", FEATURE_COLS)
        a2, l2, h2 = one_dir(df, d, "CircleBaseV2", "eccDNABase", FEATURE_COLS)
        # controllo solo-lunghezza (robusto = max delle due direzioni, per essere prudenti)
        la1, _, _ = one_dir(df, d, "eccDNABase", "CircleBaseV2", ["length"])
        la2, _, _ = one_dir(df, d, "CircleBaseV2", "eccDNABase", ["length"])
        auc_min = np.nanmin([a1, a2]); cil_min = np.nanmin([l1, l2]); len_max = np.nanmax([la1, la2])
        if cil_min > 0.5 and auc_min - len_max > 0.05:
            verdict = "CLASSIFICABILE"
        elif len_max >= 0.65 and auc_min - len_max < 0.05:
            verdict = "artefatto lunghezza"
        elif cil_min > 0.5:
            verdict = "debole/dubbio"
        else:
            verdict = "no (al caso)"
        rows.append({"malattia": d, "AUC_ecc->CB2": round(a1, 3), "AUC_CB2->ecc": round(a2, 3),
                     "AUC_robusta": round(auc_min, 3), "IC_low_robusto": round(cil_min, 3),
                     "AUC_solo_lunghezza": round(len_max, 3), "verdetto": verdict})
    res = pd.DataFrame(rows).sort_values("AUC_robusta", ascending=False)
    print("=== Classificabilita' cross-studio delle 9 malattie cross-testabili ===")
    print(res.to_string(index=False))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "classify_which.tsv"), sep="\t", index=False)
    ok = res[res.verdetto == "CLASSIFICABILE"]["malattia"].tolist()
    print(f"\n>>> CLASSIFICABILI cross-studio (segnale reale, non lunghezza): {len(ok)} -> {ok}")
    print(">>> Le 46 malattie mono-studio non sono validabili cross-studio (1 sola coorte).")
    print(f"\nSalvato in {RESULTS_DIR}/classify_which.tsv")


if __name__ == "__main__":
    main()
