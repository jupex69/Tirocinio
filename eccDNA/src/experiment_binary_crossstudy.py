"""Controllo cross-studio del CLASSIFICATORE BINARIO (sano vs malato).

RQ1 nella tesi da' ROC-AUC ~0.78 within-domain. Domanda: sotto lo STESSO
controllo del multiclasse -- leave-one-database-out -- il binario regge?

Protocollo: dentro ciascun database si appaiano sano e malato per lunghezza
(caliper), si addestra su un DB e si valuta sull'altro (entrambe le direzioni),
con IC 95% bootstrap e controllo solo-lunghezza. Feature: i 10 descrittori.

Caveat noto: i sani provengono da tessuti diversi dai malati (muscolo, plasma,
sperma) -> il binario separa in parte il TIPO di campione (tessuto normale vs
tumorale), non la malattia in senso clinico. Il test cross-studio dice se quella
separazione e' comunque riproducibile tra laboratori.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from eccdna_utils import DESCRIPTOR_NAMES, read_fasta_stream, compute_sequence_descriptors
from full_classifier import FASTA, RESULTS_DIR, SEED
from cardiac_certify import caliper_match, boot_ci

COMPACT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/eccdna_compacted_metadata.tsv")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/binary_crossstudy_features.tsv")
DBS = ["eccDNABase", "CircleBaseV2"]
CAP = 5000


def load():
    buf = {(d, s): [] for d in (0, 1) for s in DBS}
    for ch in pd.read_csv(COMPACT, sep="\t", usecols=["id", "is_disease", "source_db", "length"],
                          chunksize=400000, low_memory=False):
        ch = ch[ch["source_db"].isin(DBS)]
        for (d, s), g in ch.groupby(["is_disease", "source_db"]):
            b = buf[(d, s)]
            if sum(len(x) for x in b) < CAP:
                b.append(g.head(CAP)[["id", "is_disease", "source_db", "length"]])
    meta = pd.concat([pd.concat(v).head(CAP) for v in buf.values() if v], ignore_index=True)
    meta["id"] = meta["id"].astype(str)
    # length-match sano/malato DENTRO ogni database
    parts = []
    for s in DBS:
        d = meta[meta.source_db == s].rename(columns={"is_disease": "y"})
        parts.append(caliper_match(d).assign(source_db=s))
    m = pd.concat(parts, ignore_index=True)
    # feature: 10 descrittori dal FASTA (cache)
    ft = None; ids = set(m["id"])
    if os.path.exists(CACHE):
        ft = pd.read_csv(CACHE, sep="\t", index_col=0); ft.index = ft.index.astype(str); ids = ids - set(ft.index)
    if ids:
        feats = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
            if "N" not in seq and len(seq) >= 4:
                feats[sid] = compute_sequence_descriptors(seq)
        new = pd.DataFrame.from_dict(feats, orient="index")
        ft = new if ft is None else pd.concat([ft, new]); ft = ft[~ft.index.duplicated()]; ft.to_csv(CACHE, sep="\t")
    ftr = ft.reset_index().rename(columns={ft.reset_index().columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    return m.merge(ftr[["id"] + DESCRIPTOR_NAMES], on="id", how="inner").dropna(subset=DESCRIPTOR_NAMES)


def rf_auc(Xtr, ytr, Xte, yte):
    m, s = Xtr.mean(0), Xtr.std(0); s[s == 0] = 1.0
    rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit((Xtr-m)/s, ytr)
    p = rf.predict_proba((Xte-m)/s)[:, 1]
    return roc_auc_score(yte, p), boot_ci(yte, p)


def main():
    df = load()
    X = df[DESCRIPTOR_NAMES].to_numpy(np.float32); y = df["y"].to_numpy(); db = df["source_db"].to_numpy()
    L = df["length"].to_numpy().reshape(-1, 1)
    print(f"Binario sano/malato, length-matched: {len(df)} sequenze "
          f"(sani={int((y==0).sum())}, malati={int((y==1).sum())})\n")

    # within-study (pool, split casuale)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    aw, (mw, lw, hw) = rf_auc(Xtr, ytr, Xte, yte)
    print(f"WITHIN-STUDY (split casuale) : AUC={aw:.3f}  IC95%=[{lw:.3f},{hw:.3f}]")

    # cross-studio (train un DB, test l'altro)
    a = db == "eccDNABase"; b = db == "CircleBaseV2"
    ab, (mab, lab, hab) = rf_auc(X[a], y[a], X[b], y[b])
    ba, (mba, lba, hba) = rf_auc(X[b], y[b], X[a], y[a])
    print(f"CROSS ecc->CB2              : AUC={ab:.3f}  IC95%=[{lab:.3f},{hab:.3f}]")
    print(f"CROSS CB2->ecc              : AUC={ba:.3f}  IC95%=[{lba:.3f},{hba:.3f}]")
    print(f"CROSS-STUDIO (media)        : {(ab+ba)/2:.3f}")

    # controllo solo-lunghezza cross-studio
    la, _ = rf_auc(L[a], y[a], L[b], y[b]); lb, _ = rf_auc(L[b], y[b], L[a], y[a])
    print(f"\nControllo solo-lunghezza cross-studio: {(la+lb)/2:.3f}  (~0.5 se il matching ha funzionato)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([{"valutazione": "within_study", "AUC": round(aw, 3)},
                  {"valutazione": "cross_ecc->CB2", "AUC": round(ab, 3)},
                  {"valutazione": "cross_CB2->ecc", "AUC": round(ba, 3)},
                  {"valutazione": "cross_media", "AUC": round((ab+ba)/2, 3)},
                  {"valutazione": "solo_lunghezza_cross", "AUC": round((la+lb)/2, 3)}]
                 ).to_csv(os.path.join(RESULTS_DIR, "binary_crossstudy.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/binary_crossstudy.tsv")


if __name__ == "__main__":
    main()
