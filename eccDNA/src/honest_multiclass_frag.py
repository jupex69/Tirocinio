"""Fragmentomica applicata al MULTICLASSE onesto (cross-studio).

Aggiunge le 7 feature fragmentomiche (end-motif, breakpoint, microomologia) alle
78 (74+4) e riconfronta il RandomForest, sia within-study sia cross-studio
(train un lab, test l'altro), su TUTTE le classi. Domanda: la fragmentomica
migliora il multiclasse onesto (baseline cross-studio ~0.29)?
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from eccdna_utils import read_fasta_stream
from honest_all_siamese import load_all
from full_classifier import get_features, ALLCOLS, FASTA, RESULTS_DIR, SEED
from experiment_fragmentomics import fragmentomics, FRAG_COLS


def rf_cross(Xa, ya, Xb):
    m, s = Xa.mean(0), Xa.std(0); s[s == 0] = 1.0
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, class_weight="balanced").fit((Xa - m) / s, ya)
    return rf.predict((Xb - m) / s)


def main():
    df = load_all()
    ft = get_features(df)  # 78 in cache
    ftr = ft.reset_index(); ftr = ftr.rename(columns={ftr.columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)

    print("Calcolo fragmentomica dalle sequenze grezze...")
    frag = {}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=set(df["id"])):
        if "N" not in seq and len(seq) >= 4:
            frag[sid] = fragmentomics(seq)
    fr = pd.DataFrame.from_dict(frag, orient="index"); fr.index = fr.index.astype(str)
    fr = fr.reset_index().rename(columns={"index": "id"})

    df = df.merge(ftr, on="id", how="inner").merge(fr, on="id", how="inner").dropna(subset=ALLCOLS + FRAG_COLS)
    classes = sorted(df["lab"].unique()); nC = len(classes); chance = 1 / nC
    print(f"Classi: {nC}  campioni: {len(df)}  chance={chance:.3f}\n")

    y = df["lab"].to_numpy(); dbcol = df["source_db"].to_numpy()
    a = dbcol == "eccDNABase"; b = dbcol == "CircleBaseV2"

    for nome, cols in [("78 (74+4) baseline", ALLCOLS), ("85 (+fragmentomica)", ALLCOLS + FRAG_COLS)]:
        X = df[cols].to_numpy(np.float32)
        # within-study
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        mm, ss = Xtr.mean(0), Xtr.std(0); ss[ss == 0] = 1.0
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, class_weight="balanced").fit((Xtr - mm) / ss, ytr)
        accW = accuracy_score(yte, rf.predict((Xte - mm) / ss))
        # cross-study
        acc_ab = accuracy_score(y[b], rf_cross(X[a], y[a], X[b]))
        acc_ba = accuracy_score(y[a], rf_cross(X[b], y[b], X[a]))
        accX = (acc_ab + acc_ba) / 2
        print(f"{nome:22s}: within={accW:.3f} ({accW/chance:.1f}x)   cross-studio={accX:.3f} ({accX/chance:.1f}x)   "
              f"[ecc->CB2={acc_ab:.3f}, CB2->ecc={acc_ba:.3f}]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n>>> Se il cross-studio non sale, la fragmentomica non aiuta il multiclasse (segnale assente).")


if __name__ == "__main__":
    main()
