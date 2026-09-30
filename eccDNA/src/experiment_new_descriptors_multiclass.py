"""I descrittori NUOVI e ortogonali aiutano a IDENTIFICARE la malattia (9 classi)
in CROSS-STUDIO? -- il test che mancava.

I descrittori non-k-mer li avevamo gia' creati (experiment_richer_descriptors.py:
duplex_dg, compress_ratio, gc_heterogeneity, longest_homopolymer_frac) ma li avevamo
provati SOLO sul task cardiaco binario, mai sull'identificazione delle 9 malattie in
cross-studio. Qui li portiamo su quel task, aggiungendone altri 4 davvero ortogonali
ai 10 di base:
  - lingcomplex_k3/k4 : ricchezza del vocabolario (distinti/attesi), diversa dall'entropia
  - skew_heterogeneity: deviazione std del GC-skew lungo l'arco (mosaicismo)
  - palindrome_frac   : frazione di 6-mer palindromi rev-comp (proxy hairpin/inverted repeat)

Confronto ONESTO, length-matched, caso=1/9:
  set A) 10 descrittori (baseline)   B) 10 + 8 nuovi (18)   C) solo 8 nuovi
Modelli: RandomForest (migliore su tabellari) E siamese few-shot (3 loss), within E cross.
Domanda: il CROSS-STUDIO si muove? Se no, il muro non e' la ricchezza delle feature.
"""
import gzip
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from eccdna_utils import DESCRIPTOR_NAMES, read_fasta_stream, compute_sequence_descriptors
from train_multiclass import standardize, SEED
from full_classifier import FASTA, RESULTS_DIR
from classify_which import CROSS, COMPACT
from experiment_richer_descriptors import new_descriptors, NEW_COLS
from experiment_richer_kmers import length_match
from experiment_final_comparison import train_enc, eval_shot

CAP = 400
DBS = ["eccDNABase", "CircleBaseV2"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/new_descriptors_features.tsv")

COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s):
    return s.translate(COMP)[::-1]


def extra_descriptors(seq):
    """4 descrittori ortogonali ai 10 di base e ai 4 gia' esistenti."""
    L = len(seq)

    def lc(k):                                   # complessita' linguistica: distinti/attesi
        n = L - k + 1
        if n <= 0:
            return 0.0
        obs = len(set(seq[i:i + k] for i in range(n)))
        return obs / min(4 ** k, n)

    w = 20                                        # eterogeneita' del GC-skew su finestre
    sk = []
    for i in range(0, L - w + 1, w):
        win = seq[i:i + w]; g = win.count("G"); c = win.count("C")
        sk.append((g - c) / (g + c) if (g + c) else 0.0)
    skew_het = float(np.std(sk)) if sk else 0.0

    n6 = L - 6 + 1                                # frazione di 6-mer palindromi rev-comp
    pal = sum(1 for i in range(n6) if seq[i:i + 6] == revcomp(seq[i:i + 6])) / n6 if n6 > 0 else 0.0

    return {"lingcomplex_k3": lc(3), "lingcomplex_k4": lc(4),
            "skew_heterogeneity": skew_het, "palindrome_frac": pal}


EXTRA_COLS = ["lingcomplex_k3", "lingcomplex_k4", "skew_heterogeneity", "palindrome_frac"]
NEW_ALL = list(NEW_COLS) + EXTRA_COLS            # 8 nuovi descrittori ortogonali
COLS = list(DESCRIPTOR_NAMES) + NEW_ALL          # 18 in totale

SETS = {
    "A: 10 descrittori": list(DESCRIPTOR_NAMES),
    "B: 10 + 8 nuovi": COLS,
    "C: solo 8 nuovi": NEW_ALL,
}


def featurize(seq):
    d = compute_sequence_descriptors(seq)
    row = [d[c] for c in DESCRIPTOR_NAMES]
    nd = new_descriptors(seq); row += [nd[c] for c in NEW_COLS]
    ex = extra_descriptors(seq); row += [ex[c] for c in EXTRA_COLS]
    return row


def load():
    buf = {(d, s): [] for d in CROSS for s in DBS}
    for ch in pd.read_csv(COMPACT, sep="\t", usecols=["id", "label", "source_db", "length"], chunksize=400000, low_memory=False):
        ch = ch[ch["label"].isin(CROSS) & ch["source_db"].isin(DBS)]
        for (d, s), g in ch.groupby(["label", "source_db"]):
            b = buf[(d, s)]
            if sum(len(x) for x in b) < CAP:
                b.append(g.head(CAP)[["id", "label", "source_db", "length"]])
    meta = pd.concat([pd.concat(v).head(CAP) for v in buf.values() if v], ignore_index=True)
    meta["id"] = meta["id"].astype(str)
    ft = None
    if os.path.exists(CACHE):
        ft = pd.read_csv(CACHE, sep="\t", index_col=0); ft.index = ft.index.astype(str)
    ids = set(meta["id"]) - (set(ft.index) if ft is not None else set())
    if ids:
        rows = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
            if "N" not in seq and len(seq) >= 6:
                rows[sid] = featurize(seq)
        new = pd.DataFrame.from_dict(rows, orient="index", columns=COLS)
        ft = new if ft is None else pd.concat([ft, new]); ft = ft[~ft.index.duplicated()]; ft.to_csv(CACHE, sep="\t")
    ftr = ft.reset_index().rename(columns={ft.reset_index().columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    return meta.merge(ftr, on="id", how="inner").dropna(subset=COLS)


def main():
    df = length_match(load())
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    chance = 1 / nC
    y = df["label"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()
    a = db == DBS[0]; b = db == DBS[1]
    print(f"9 malattie, {len(df)} sequenze (length-matched), caso={chance:.3f}\n")

    # ================= RandomForest (usa tutti gli esempi) =================
    def rf_eval(Xtr, ytr, Xte, yte):
        rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(Xtr, ytr)
        pr = rf.predict(Xte)
        return accuracy_score(yte, pr), f1_score(yte, pr, average="macro", zero_division=0)

    print("=== RandomForest ===")
    rf_rows = []
    for name, cols in SETS.items():
        X = df[cols].to_numpy(np.float32)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        Xtr_s, Xte_s = standardize(Xtr, Xte)
        w_acc, w_f1 = rf_eval(Xtr_s, ytr, Xte_s, yte)
        XA, XB = standardize(X[a], X[b]); XB2, XA2 = standardize(X[b], X[a])
        c1 = rf_eval(XA, y[a], XB, y[b]); c2 = rf_eval(XB2, y[b], XA2, y[a])
        c_acc = (c1[0] + c2[0]) / 2; c_f1 = (c1[1] + c2[1]) / 2
        rf_rows.append({"set": name, "within_acc": round(w_acc, 3), "within_xcaso": round(w_acc / chance, 2),
                        "cross_acc": round(c_acc, 3), "cross_F1": round(c_f1, 3), "cross_xcaso": round(c_acc / chance, 2)})
        print(f"  {name:20s} within acc={w_acc:.3f} ({w_acc/chance:.1f}x) | cross acc={c_acc:.3f} F1={c_f1:.3f} ({c_acc/chance:.1f}x)")

    # ================= Siamese few-shot (K=5, 3 loss) su A vs B =================
    print("\n=== Siamese few-shot (K=5) ===")
    sm_rows = []
    for name in ["A: 10 descrittori", "B: 10 + 8 nuovi"]:
        cols = SETS[name]; X = df[cols].to_numpy(np.float32)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        Xtr_s, Xte_s = standardize(Xtr, Xte)
        XA, XB = standardize(X[a], X[b]); XB2, XA2 = standardize(X[b], X[a])
        for loss in ["euclideo", "coseno", "triplet"]:
            cos = (loss == "coseno")
            enc_w = train_enc(loss, Xtr_s, ytr, nC)
            encA = train_enc(loss, XA, y[a], nC); encB = train_enc(loss, XB2, y[b], nC)
            w = eval_shot(enc_w, Xtr_s, ytr, Xte_s, yte, 5, cos)
            c = (eval_shot(encA, XA, y[a], XB, y[b], 5, cos) + eval_shot(encB, XB2, y[b], XA2, y[a], 5, cos)) / 2
            sm_rows.append({"set": name, "loss": loss, "within_acc": round(w, 3), "within_xcaso": round(w / chance, 2),
                            "cross_acc": round(c, 3), "cross_xcaso": round(c / chance, 2)})
            print(f"  {name:20s} {loss:9s} within acc={w:.3f} ({w/chance:.1f}x) | cross acc={c:.3f} ({c/chance:.1f}x)")

    # ================= importanza: i nuovi contano? (within, set B) =================
    X = df[COLS].to_numpy(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(Xtr_s, ytr)
    imp = pd.Series(rf.feature_importances_, index=COLS).sort_values(ascending=False)
    print("\n--- Top 12 feature (set B), i nuovi marcati ---")
    for nome, v in imp.head(12).items():
        print(f"  {nome:24s} {v:.4f}{'  <-- NUOVO' if nome in NEW_ALL else ''}")
    print("Somma importanza 10 base :", round(imp[list(DESCRIPTOR_NAMES)].sum(), 3))
    print("Somma importanza 8 nuovi :", round(imp[NEW_ALL].sum(), 3))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rf_rows).to_csv(os.path.join(RESULTS_DIR, "new_descriptors_multiclass_rf.tsv"), sep="\t", index=False)
    pd.DataFrame(sm_rows).to_csv(os.path.join(RESULTS_DIR, "new_descriptors_multiclass_siamese.tsv"), sep="\t", index=False)
    print(f"\ncaso={chance:.3f}. Salvato in {RESULTS_DIR}/new_descriptors_multiclass_*.tsv")


if __name__ == "__main__":
    main()
