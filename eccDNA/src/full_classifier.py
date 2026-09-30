"""IL SISTEMA COMPLETO (obiettivo ultimo): data una sequenza eccDNA, estrarre i
descrittori (tabellari + non) e dire SE e QUALE malattia, tra TUTTE quelle del
dataset (+ una classe 'Healthy' per il 'se').

Pipeline end-to-end:
  sequenza --> 64 spettro 3-mer + 10 descrittori + 4 nuovi descrittori (78) -->
               RandomForest bilanciato --> (malattia | Healthy) + confidenza

Valutazione ONESTA e DOPPIA:
  (1) WITHIN-STUDY (split stratificato casuale): numero OTTIMISTICO. Attenzione:
      qui train e test contengono gli STESSI studi -> in larga parte il modello
      riconosce protocollo/studio, non biologia (dimostrato in tutta l'indagine).
  (2) Rinvio CROSS-STUDY: per le malattie presenti in un solo studio (quasi tutte)
      il transfer a un laboratorio nuovo crolla al caso (cross_source_*). Quindi
      il numero (1) NON e' l'accuratezza attesa su un paziente di un lab nuovo.

Espone anche predict(seq) per l'uso reale. Salva modello e metriche in results/.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from eccdna_utils import read_fasta_stream, compute_sequence_descriptors
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from experiment_richer_descriptors import new_descriptors, NEW_COLS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection_metadata.tsv")
FASTA = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection.body.fa")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CACHE = os.path.join(SCRIPT_DIR, "data/processed/full_classifier_features.tsv")
ALLCOLS = FEATURE_COLS + NEW_COLS
CHUNK = 400000
CAP_DIS = 1200
CAP_HEALTHY = 3000
MIN_N = 200
SEED = 42


# canonicalizzazione etichette: unisce i duplicati sotto un'unica etichetta.
# 1) case/spazi normalizzati -> unisce i duplicati di sola maiuscola;
# 2) SYN: varianti della STESSA malattia (parola ridondante 'cancer', sinonimi);
#    NB si tengono separati gli STATI diversi (adenoma benigno != cancro maligno).
# 3) EXCLUDE: etichette spurie (Multiple Diseases = mix non interpretabile).
SYN = {
    "glioblastoma cancer": "glioblastoma",
    "gastric cancer": "stomach cancer", "stomach": "stomach cancer",
    "hypopharyngeal squamous cell carcinoma": "hypopharynx cancer",
    "colon cancer": "colorectal cancer",
    "multiple myeloma cancer": "multiple myeloma",
    "thyroid": "thyroid cancer", "liver": "liver cancer", "liver disease": "liver cancer",
    "lung": "lung cancer", "brain cancer": "glioblastoma",
}
EXCLUDE = {"multiple diseases"}


def canon(disease):
    k = " ".join(str(disease).strip().lower().split())
    if k in EXCLUDE:
        return None
    return SYN.get(k, k).title()


def load_labels():
    """Tutte le malattie canonicalizzate (>= MIN_N) + Healthy, con tetto per classe."""
    buf = {}
    for ch in pd.read_csv(META, sep="\t",
                          usecols=["id", "disease", "disease_binary_name", "source_db", "length"],
                          chunksize=CHUNK, low_memory=False):
        healthy = ch["disease_binary_name"].astype(str).str.lower() == "healthy"
        ch["lab"] = np.where(healthy, "Healthy", ch["disease"].map(canon))
        ch = ch[ch["lab"].notna() & (ch["lab"] != "None")]
        for lab, g in ch.groupby("lab"):
            cap = CAP_HEALTHY if lab == "Healthy" else CAP_DIS
            cur = buf.get(lab)
            have = 0 if cur is None else len(cur)
            if have < cap:
                take = g.head(cap - have)[["id", "lab", "length", "source_db"]]
                buf[lab] = take if cur is None else pd.concat([cur, take], ignore_index=True)
    df = pd.concat(buf.values(), ignore_index=True)
    df["id"] = df["id"].astype(str)
    keep = df["lab"].value_counts(); keep = keep[keep >= MIN_N].index
    return df[df["lab"].isin(keep)].reset_index(drop=True)


def get_features(df):
    """Cache INCREMENTALE: riusa gli id gia' calcolati, calcola solo i mancanti."""
    need = set(df["id"]); ft = None
    if os.path.exists(CACHE):
        ft = pd.read_csv(CACHE, sep="\t", index_col=0); ft.index = ft.index.astype(str)
        missing = need - set(ft.index)
    else:
        missing = need
    if missing:
        feats = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=missing):
            if "N" not in seq and len(seq) >= 4:
                f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq)); f.update(new_descriptors(seq))
                feats[sid] = f
        new = pd.DataFrame.from_dict(feats, orient="index")
        ft = new if ft is None else pd.concat([ft, new])
        ft = ft[~ft.index.duplicated()]
        os.makedirs(os.path.dirname(CACHE), exist_ok=True); ft.to_csv(CACHE, sep="\t")
    return ft


def main():
    df = load_labels()
    ft = get_features(df)
    ftr = ft.reset_index(); ftr = ftr.rename(columns={ftr.columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    df = df.merge(ftr, on="id", how="inner").dropna(subset=ALLCOLS)

    classes = sorted(df["lab"].unique())
    n_dis = len([c for c in classes if c != "Healthy"])
    print(f"Classi DOPO fusione etichette: {len(classes)}  ({n_dis} malattie + Healthy)")
    print(f"Campioni: {len(df)}   feature: {len(ALLCOLS)} (74 + {len(NEW_COLS)} nuovi)")
    print(f"chance (uniforme) = {1/len(classes):.3f}")
    vc = df["lab"].value_counts()
    print("Numerosita' per classe:"); print(vc.to_string(), "\n")

    X = df[ALLCOLS].to_numpy(np.float32); y = df["lab"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    m, s = Xtr.mean(0), Xtr.std(0); s[s == 0] = 1.0
    Xtr_s, Xte_s = (Xtr - m) / s, (Xte - m) / s
    from sklearn.metrics import recall_score, precision_score
    train_counts = pd.Series(ytr).value_counts()
    rare_thresh = float(train_counts.quantile(0.4))            # 40% inferiore per numerosita'
    rare = set(train_counts[train_counts <= rare_thresh].index)

    def heavy_weights(alpha):
        """peso ∝ (N/(K*n_c))^alpha : alpha=1 = 'balanced' standard, alpha>1 =
        peso PIU' aggressivo sulle classi rare (richiesta esplicita)."""
        N, K = len(ytr), len(train_counts)
        return {c: (N / (K * train_counts[c])) ** alpha for c in train_counts.index}

    def fit_eval(alpha, nome):
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1,
                                    class_weight=heavy_weights(alpha)).fit(Xtr_s, ytr)
        proba = rf.predict_proba(Xte_s); pred = rf.classes_[proba.argmax(1)]
        acc = accuracy_score(yte, pred); bacc = balanced_accuracy_score(yte, pred)
        top3 = top_k_accuracy_score(yte, proba, k=3, labels=rf.classes_)
        recmap = dict(zip(rf.classes_, recall_score(yte, pred, average=None, labels=rf.classes_, zero_division=0)))
        rr = np.mean([recmap[c] for c in rf.classes_ if c in rare])
        cr = np.mean([recmap[c] for c in rf.classes_ if c not in rare])
        print(f"  [{nome:20s}] acc={acc:.3f} bal_acc={bacc:.3f} top3={top3:.3f}  "
              f"recall RARE={rr:.3f}  recall comuni={cr:.3f}")
        return rf, proba, pred, acc, bacc, top3

    print("=== (1) WITHIN-STUDY — bilanciamento PESATO per le classi rare ===")
    print(f"  ({len(rare)} classi rare = train <= {rare_thresh:.0f} campioni)  chance={1/len(classes):.3f}")
    fit_eval(1.0, "balanced (alpha=1)")
    fit_eval(1.5, "pesato alpha=1.5")
    rf, proba, pred, acc, bacc, top3 = fit_eval(2.0, "pesato alpha=2 (scelto)")
    mf1 = f1_score(yte, pred, average="macro", zero_division=0)
    print(f"  -> scelto alpha=2: accuracy={acc:.3f}  ({acc/(1/len(classes)):.1f}x il caso)")

    is_dis_true = (yte != "Healthy").astype(int); is_dis_pred = (pred != "Healthy").astype(int)
    print(f"  'SE malato' (malato-vs-Healthy): sensibilita'={recall_score(is_dis_true, is_dis_pred):.3f}  "
          f"precisione={precision_score(is_dis_true, is_dis_pred, zero_division=0):.3f}")

    rep = pd.DataFrame({"classe": rf.classes_,
                        "recall": recall_score(yte, pred, average=None, labels=rf.classes_, zero_division=0),
                        "n_test": [int((yte == c).sum()) for c in rf.classes_]}).sort_values("recall", ascending=False)
    print("\n  Migliori 6 classi (recall):"); print(rep.head(6).to_string(index=False))
    print("  Peggiori 4 classi:"); print(rep.tail(4).to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([{"metrica": "accuracy", "valore": round(acc, 3)},
                  {"metrica": "balanced_accuracy", "valore": round(bacc, 3)},
                  {"metrica": "macro_f1", "valore": round(mf1, 3)},
                  {"metrica": "top3", "valore": round(top3, 3)},
                  {"metrica": "x_chance", "valore": round(acc/(1/len(classes)), 2)},
                  {"metrica": "n_classi", "valore": len(classes)}]).to_csv(
        os.path.join(RESULTS_DIR, "full_classifier_metrics.tsv"), sep="\t", index=False)
    rep.to_csv(os.path.join(RESULTS_DIR, "full_classifier_per_class.tsv"), sep="\t", index=False)

    print("\n=== (2) ATTESO SU UN LABORATORIO NUOVO (onesto) ===")
    print("  Le malattie del dataset sono quasi tutte in UN SOLO studio -> il transfer")
    print("  cross-studio crolla al caso (cross_source_transfer). Quindi il numero (1)")
    print("  vale su dati degli STESSI studi, NON su un paziente di un lab mai visto.")

    # --- pipeline d'uso ---
    def predict(seq):
        f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq)); f.update(new_descriptors(seq))
        x = (np.array([f[c] for c in ALLCOLS], np.float32) - m) / s
        p = rf.predict_proba([x])[0]; i = int(p.argmax())
        return rf.classes_[i], float(p[i])
    # demo su 3 sequenze di test
    print("\n=== demo predict() su 3 sequenze ===")
    for sid, seq in list(read_fasta_stream(FASTA, wanted_ids=set(df["id"].sample(3, random_state=1)))):
        lab, conf = predict(seq)
        print(f"  id={sid[:22]:22s} -> predetto: {lab:28s} (confidenza {conf:.2f})")

    print(f"\nSalvato in {RESULTS_DIR}/full_classifier_metrics.tsv e full_classifier_per_class.tsv")


if __name__ == "__main__":
    main()
