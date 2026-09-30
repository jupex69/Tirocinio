"""Il multiclasse che PASSA il controllo dei confondenti: stesso pipeline del
sistema completo, ma valutato leave-one-database-out (train su un laboratorio,
test sull'altro) invece che con split casuale.

Dopo la fusione delle etichette, alcune malattie esistono in ENTRAMBI i database
(eccDNABase e CircleBaseV2) e diventano testabili cross-studio. Su queste (+
Healthy) si confronta:
  (A) SPLIT CASUALE (stessi studi in train e test) -> numero gonfiato;
  (B) SPLIT PER DATABASE (train un lab, test l'altro) -> numero ONESTO.
Il divario (A - B) e' la quota di confondente. Ci si aspetta B molto piu' basso.

Feature: 74 + 4 nuovi. Modello: RandomForest bilanciato. Riusa canon() e la cache
incrementale di full_classifier.py.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from full_classifier import canon, get_features, ALLCOLS, META, RESULTS_DIR, CHUNK, SEED

CAP = 1000
MIN_PER_DB = 150
DBS = ["eccDNABase", "CircleBaseV2"]


def load_both_db():
    buf = {}
    for ch in pd.read_csv(META, sep="\t",
                          usecols=["id", "disease", "disease_binary_name", "source_db", "length"],
                          chunksize=CHUNK, low_memory=False):
        ch = ch[ch["source_db"].isin(DBS)]
        healthy = ch["disease_binary_name"].astype(str).str.lower() == "healthy"
        ch["lab"] = np.where(healthy, "Healthy", ch["disease"].map(canon))
        ch = ch[ch["lab"].notna() & (ch["lab"] != "None")]
        for (lab, db), g in ch.groupby(["lab", "source_db"]):
            key = (lab, db); cur = buf.get(key); have = 0 if cur is None else len(cur)
            if have < CAP:
                take = g.head(CAP - have)[["id", "lab", "source_db", "length"]]
                buf[key] = take if cur is None else pd.concat([cur, take], ignore_index=True)
    df = pd.concat(buf.values(), ignore_index=True); df["id"] = df["id"].astype(str)
    counts = df.groupby(["lab", "source_db"]).size().unstack(fill_value=0)
    keep = counts[(counts.get("eccDNABase", 0) >= MIN_PER_DB) & (counts.get("CircleBaseV2", 0) >= MIN_PER_DB)].index
    return df[df["lab"].isin(keep)].reset_index(drop=True), counts.loc[keep]


def std_fit(Xtr, Xte):
    m, s = Xtr.mean(0), Xtr.std(0); s[s == 0] = 1.0
    return (Xtr - m) / s, (Xte - m) / s


def rf_eval(Xtr, ytr, Xte, yte, classes):
    rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(Xtr, ytr)
    pred = rf.predict(Xte)
    return accuracy_score(yte, pred), balanced_accuracy_score(yte, pred), \
        dict(zip(classes, recall_score(yte, pred, average=None, labels=classes, zero_division=0)))


def main():
    df, counts = load_both_db()
    ft = get_features(df)
    ftr = ft.reset_index(); ftr = ftr.rename(columns={ftr.columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    df = df.merge(ftr, on="id", how="inner").dropna(subset=ALLCOLS)
    classes = sorted(df["lab"].unique()); nC = len(classes)
    print(f"Classi testabili cross-studio (in entrambi i DB): {nC}  chance={1/nC:.3f}")
    print("Numerosita' per (classe, database):"); print(counts.to_string(), "\n")

    X = df[ALLCOLS].to_numpy(np.float32); y = df["lab"].to_numpy(); dbcol = df["source_db"].to_numpy()

    # (A) split casuale (within-study)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    Xtr_s, Xte_s = std_fit(Xtr, Xte)
    accA, baccA, _ = rf_eval(Xtr_s, ytr, Xte_s, yte, classes)

    # (B) leave-one-database-out
    def crossdb(train_db, test_db):
        tr = dbcol == train_db; te = dbcol == test_db
        Xtr_s, Xte_s = std_fit(X[tr], X[te])
        return rf_eval(Xtr_s, y[tr], Xte_s, y[te], classes)
    accB1, baccB1, rec1 = crossdb("eccDNABase", "CircleBaseV2")
    accB2, baccB2, rec2 = crossdb("CircleBaseV2", "eccDNABase")
    accB, baccB = (accB1 + accB2) / 2, (baccB1 + baccB2) / 2

    print("=== CONFRONTO: split casuale (confuso) vs per-studio (onesto) ===")
    print(f"(A) split CASUALE  : accuracy={accA:.3f}  bal_acc={baccA:.3f}  ({accA/(1/nC):.1f}x il caso)")
    print(f"(B) per-STUDIO     : accuracy={accB:.3f}  bal_acc={baccB:.3f}  ({accB/(1/nC):.1f}x il caso)")
    print(f"      ecc->CB2: acc={accB1:.3f}   CB2->ecc: acc={accB2:.3f}")
    print(f"\n>>> CALO passando al controllo dei confondenti: {accA:.3f} -> {accB:.3f}  (-{accA-accB:.3f})")
    print(f">>> il {100*(accA-accB)/max(accA,1e-9):.0f}% dell'accuratezza era confondente di studio")

    # per-classe: chi sopravvive al cross-studio
    rec = pd.DataFrame({"classe": classes,
                        "recall_ecc->CB2": [round(rec1[c], 3) for c in classes],
                        "recall_CB2->ecc": [round(rec2[c], 3) for c in classes]})
    rec["recall_min"] = rec[["recall_ecc->CB2", "recall_CB2->ecc"]].min(axis=1)
    rec = rec.sort_values("recall_min", ascending=False)
    print("\nPer-classe (recall cross-studio, min tra le due direzioni):")
    print(rec.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([{"valutazione": "split_casuale", "accuracy": round(accA, 3), "bal_acc": round(baccA, 3), "x_caso": round(accA/(1/nC), 2)},
                  {"valutazione": "per_studio_onesto", "accuracy": round(accB, 3), "bal_acc": round(baccB, 3), "x_caso": round(accB/(1/nC), 2)}]
                 ).to_csv(os.path.join(RESULTS_DIR, "honest_multiclass.tsv"), sep="\t", index=False)
    rec.to_csv(os.path.join(RESULTS_DIR, "honest_multiclass_per_class.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/honest_multiclass.tsv")


if __name__ == "__main__":
    main()
