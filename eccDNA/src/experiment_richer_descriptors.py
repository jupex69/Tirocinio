"""Il limite e' la rappresentazione TABELLARE o il SEGNALE? Test sul solo task
valido (Cuore: Dilated Cardiomyopathy vs cuore sano, stesso tessuto/protocollo/DB).

Si confrontano, tutte con lunghezza controllata (caliper matching) e IC bootstrap:
 1. 74 feature tabellari (baseline, RandomForest) -> AUC ~0.578 gia' visto;
 2. 74 + NUOVI descrittori (non-k-mer): energia di duplex nearest-neighbor
    (SantaLucia), complessita' da compressione gzip (proxy di Kolmogorov),
    eterogeneita' locale del GC, frazione in run omopolimerici;
 3. CNN 1D sulla SEQUENZA GREZZA (one-hot, finestra a lunghezza fissa: la
    lunghezza non puo' essere sfruttata) -> rappresentazione NON tabellare che
    impara i motivi da se'.

Se (2) o (3) battono nettamente (1), la rappresentazione era il collo di
bottiglia. Se no, il tetto ~0.58 e' il segnale reale e piu' feature non aiutano.
"""

import gzip
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from eccdna_utils import read_fasta_stream
from train_siamese_multiclass import FEATURE_COLS
from experiment_within_tissue import load_task_ids, get_features
from cardiac_certify import caliper_match, strat_split, boot_ci
from train_multiclass import standardize, SEED

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTA = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection.body.fa")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# --- nuovi descrittori (non k-mer) ---
# energia libera nearest-neighbor ΔG°37 (SantaLucia 1998, kcal/mol) per i 16 dinucleotidi
NN_DG = {"AA": -1.00, "TT": -1.00, "AT": -0.88, "TA": -0.58, "CA": -1.45, "TG": -1.45,
         "GT": -1.44, "AC": -1.44, "CT": -1.28, "AG": -1.28, "GA": -1.30, "TC": -1.30,
         "CG": -2.17, "GC": -2.24, "GG": -1.84, "CC": -1.84}
NEW_COLS = ["duplex_dg", "compress_ratio", "gc_heterogeneity", "longest_homopolymer_frac"]


def new_descriptors(seq):
    L = len(seq)
    # ΔG medio per passo (indipendente dalla lunghezza)
    dgs = [NN_DG[seq[i:i+2]] for i in range(L-1) if seq[i:i+2] in NN_DG]
    dg = sum(dgs)/len(dgs) if dgs else 0.0
    # complessita' da compressione (ridondanza globale oltre i 3-mer)
    comp = len(gzip.compress(seq.encode(), 9)) / max(L, 1)
    # eterogeneita' locale del GC (deviazione std del GC su finestre da 20)
    w = 20
    gcs = [(seq[i:i+w].count("G")+seq[i:i+w].count("C"))/w for i in range(0, L-w+1, w)] if L >= w else []
    gc_het = float(np.std(gcs)) if gcs else 0.0
    # frazione nel piu' lungo run omopolimerico
    best = cur = 1 if L else 0
    for i in range(1, L):
        cur = cur+1 if seq[i] == seq[i-1] else 1
        best = max(best, cur)
    hp = best/L if L else 0.0
    return {"duplex_dg": dg, "compress_ratio": comp, "gc_heterogeneity": gc_het, "longest_homopolymer_frac": hp}


# --- CNN sulla sequenza grezza ---
MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot(seq, L):
    """Finestra centrale di L basi -> matrice (4, L). Lunghezza fissa: la CNN non
    puo' usare la lunghezza. Sequenze piu' corte di L: scartate a monte."""
    s = len(seq); start = max(0, (s - L)//2); sub = seq[start:start+L]
    x = np.zeros((4, L), np.float32)
    for i, ch in enumerate(sub):
        j = MAP.get(ch)
        if j is not None:
            x[j, i] = 1.0
    return x


class SeqCNN(nn.Module):
    def __init__(self, L, ch=32, k=8, drop=0.3):
        super().__init__()
        self.c1 = nn.Conv1d(4, ch, k, padding=k//2)
        self.c2 = nn.Conv1d(ch, ch, k, padding=k//2)
        self.fc = nn.Sequential(nn.Linear(ch, 32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32, 1))

    def forward(self, x):
        h = F.relu(self.c1(x)); h = F.relu(self.c2(h))
        h = h.max(dim=2).values  # global max pooling (invariante alla posizione)
        return self.fc(h).squeeze(-1)


def train_cnn(Xtr, ytr, Xva, yva, epochs=120, lr=1e-3, patience=15, seed=SEED):
    torch.manual_seed(seed)
    m = SeqCNN(Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.as_tensor(Xtr); yt = torch.as_tensor(ytr, dtype=torch.float32)
    Xv = torch.as_tensor(Xva)
    pw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(Xt); best, best_state, no = -1, None, 0
    for _ in range(epochs):
        m.train(); perm = torch.randperm(n)
        for s in range(0, n, 64):
            idx = perm[s:s+64]
            opt.zero_grad(); loss = crit(m(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            va = roc_auc_score(yva, torch.sigmoid(m(Xv)).numpy())
        if va > best:
            best, best_state, no = va, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            no += 1
            if no >= patience:
                break
    if best_state:
        m.load_state_dict(best_state)
    return m


def main():
    L = 180
    tasks = load_task_ids()
    ftab = get_features(tasks)
    ft = ftab.reset_index(); ft = ft.rename(columns={ft.columns[0]: "id"}); ft["id"] = ft["id"].astype(str)
    df = tasks[1].merge(ft, on="id", how="inner").dropna(subset=FEATURE_COLS)  # cuore
    ids = set(df["id"])

    print("Lettura sequenze grezze del cuore dal FASTA...")
    seqs = {}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
        if "N" not in seq and len(seq) >= L:
            seqs[sid] = seq
    df = df[df["id"].isin(seqs)].reset_index(drop=True)
    # nuovi descrittori
    nd = pd.DataFrame([new_descriptors(seqs[i]) for i in df["id"]])
    df = pd.concat([df, nd], axis=1)

    df = caliper_match(df); df = strat_split(df)
    tr, va, te = df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]
    print(f"coppie matchate (len>={L}): {int((df.y==1).sum())}+{int((df.y==0).sum())}  "
          f"(train={len(tr)} val={len(va)} test={len(te)})")
    print(f"len mediana test: malati={te[te.y==1]['length'].median():.0f} sani={te[te.y==0]['length'].median():.0f}\n")

    ytr, yva, yte = tr.y.to_numpy(), va.y.to_numpy(), te.y.to_numpy()

    def rf_auc(cols):
        Xtr, Xva, Xte = standardize(tr[cols].to_numpy(np.float32), va[cols].to_numpy(np.float32), te[cols].to_numpy(np.float32))
        rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        return boot_ci(yte, rf.predict_proba(Xte)[:, 1])

    # controllo lunghezza
    m0, l0, h0 = boot_ci(yte, RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
                         .fit(tr[["length"]], ytr).predict_proba(te[["length"]])[:, 1])
    print(f"CONTROLLO solo-lunghezza      : AUC={m0:.3f}  IC95%=[{l0:.3f},{h0:.3f}]")
    m1, l1, h1 = rf_auc(FEATURE_COLS)
    print(f"(1) 74 tabellari              : AUC={m1:.3f}  IC95%=[{l1:.3f},{h1:.3f}]")
    m2, l2, h2 = rf_auc(FEATURE_COLS + NEW_COLS)
    print(f"(2) 74 + nuovi descrittori    : AUC={m2:.3f}  IC95%=[{l2:.3f},{h2:.3f}]")

    # (3) CNN sulla sequenza grezza
    Xtr = np.stack([one_hot(seqs[i], L) for i in tr["id"]])
    Xva = np.stack([one_hot(seqs[i], L) for i in va["id"]])
    Xte = np.stack([one_hot(seqs[i], L) for i in te["id"]])
    cnn = train_cnn(Xtr, ytr, Xva, yva)
    with torch.no_grad():
        pte = torch.sigmoid(cnn(torch.as_tensor(Xte))).numpy()
    m3, l3, h3 = boot_ci(yte, pte)
    print(f"(3) CNN sequenza grezza       : AUC={m3:.3f}  IC95%=[{l3:.3f},{h3:.3f}]")

    # importanza dei nuovi descrittori
    Xtr_s, Xte_s = standardize(tr[FEATURE_COLS+NEW_COLS].to_numpy(np.float32), te[FEATURE_COLS+NEW_COLS].to_numpy(np.float32))
    rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1).fit(Xtr_s, ytr)
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS+NEW_COLS).sort_values(ascending=False)
    print("\nTop-8 feature per importanza (con i nuovi in maiuscolo se presenti):")
    for name, v in imp.head(8).items():
        tag = " <-- NUOVO" if name in NEW_COLS else ""
        print(f"  {name:24s} {v:.4f}{tag}")
    print("Posizione dei nuovi descrittori:", {c: int(imp.index.get_loc(c))+1 for c in NEW_COLS})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([
        {"rappresentazione": "solo-lunghezza (controllo)", "AUC": round(m0, 3), "IC": f"[{l0:.3f},{h0:.3f}]"},
        {"rappresentazione": "74 tabellari", "AUC": round(m1, 3), "IC": f"[{l1:.3f},{h1:.3f}]"},
        {"rappresentazione": "74 + nuovi descrittori", "AUC": round(m2, 3), "IC": f"[{l2:.3f},{h2:.3f}]"},
        {"rappresentazione": "CNN sequenza grezza", "AUC": round(m3, 3), "IC": f"[{l3:.3f},{h3:.3f}]"},
    ]).to_csv(os.path.join(RESULTS_DIR, "richer_descriptors_cardiac.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/richer_descriptors_cardiac.tsv")


if __name__ == "__main__":
    main()
