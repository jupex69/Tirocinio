"""Tecnica dalla letteratura cfDNA: FRAGMENTOMICA (end-motif, composizione dei
breakpoint, microomologia di giunzione). Sono feature considerate piu' biologiche
e piu' stabili al batch. Le aggiungo dove abbiamo un segnale di malattia REALE e
controllato: colon benigno (adenoma) vs maligno (cancro), stesso tessuto/protocollo,
length-matched. Test decisivo: la fragmentomica migliora l'AUC oltre 0.596?

Feature fragmentomiche per sequenza eccDNA (il body ha un breakpoint start/end):
 - start_gc30/end_gc30: GC nelle prime/ultime 30 basi (regione di giunzione);
 - gc_end_diff: asimmetria GC tra le due estremita';
 - junction_microhomology: piu' lungo repeat diretto tra inizio e fine (segnatura
   di formazione dell'eccDNA per microomologia);
 - start_purine/end_purine: contenuto purinico alle estremita';
 - end_motif_dg: energia di duplex del 4-mer terminale (motivo di fine frammento).

Confronto RandomForest: 78 feature (74+4) vs 78+fragmentomica, IC 95% bootstrap.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from eccdna_utils import read_fasta_stream, compute_sequence_descriptors
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from experiment_richer_descriptors import new_descriptors, NEW_COLS
from experiment_disease_state import load_ids, FASTA
from full_classifier import RESULTS_DIR
from cardiac_certify import caliper_match, strat_split, boot_ci
from train_multiclass import standardize, SEED

NN_DG = {"AA": -1.00, "TT": -1.00, "AT": -0.88, "TA": -0.58, "CA": -1.45, "TG": -1.45,
         "GT": -1.44, "AC": -1.44, "CT": -1.28, "AG": -1.28, "GA": -1.30, "TC": -1.30,
         "CG": -2.17, "GC": -2.24, "GG": -1.84, "CC": -1.84}
FRAG_COLS = ["start_gc30", "end_gc30", "gc_end_diff", "junction_microhomology",
             "start_purine", "end_purine", "end_motif_dg"]


def fragmentomics(seq):
    L = len(seq); w = min(30, L)
    s, e = seq[:w], seq[-w:]
    sg = (s.count("G") + s.count("C")) / w
    eg = (e.count("G") + e.count("C")) / w
    mh = 0
    for k in range(1, min(20, L // 2) + 1):
        if seq[:k] == seq[-k:]:
            mh = k
    sp = sum(1 for c in s if c in "AG") / w
    ep = sum(1 for c in e if c in "AG") / w
    dgs = [NN_DG[seq[i:i+2]] for i in list(range(3)) + list(range(L-3, L-1)) if seq[i:i+2] in NN_DG]
    dg = sum(dgs) / len(dgs) if dgs else 0.0
    return {"start_gc30": sg, "end_gc30": eg, "gc_end_diff": abs(sg - eg),
            "junction_microhomology": mh / L if L else 0.0, "start_purine": sp,
            "end_purine": ep, "end_motif_dg": dg}


def main():
    df = load_ids()  # colon adenoma (0) vs cancer (1)
    print(f"Adenoma: {int((df.y==0).sum())}  Cancer: {int((df.y==1).sum())}  — lettura sequenze...")
    feats = {}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=set(df["id"])):
        if "N" not in seq and len(seq) >= 4:
            f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq))
            f.update(new_descriptors(seq)); f.update(fragmentomics(seq)); feats[sid] = f
    ft = pd.DataFrame.from_dict(feats, orient="index")
    ft.index = ft.index.astype(str)
    df = df.set_index("id").join(ft, how="inner").reset_index()
    df = df.rename(columns={df.columns[0]: "id"}).dropna(subset=FEATURE_COLS + NEW_COLS + FRAG_COLS)

    df = caliper_match(df); df = strat_split(df)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    ytr, yte = tr.y.to_numpy(), te.y.to_numpy()
    print(f"dopo matching: benigni={int((df.y==0).sum())} maligni={int((df.y==1).sum())}  test={len(te)}\n")

    def auc(cols, nome):
        Xtr, Xte = standardize(tr[cols].to_numpy(np.float32), te[cols].to_numpy(np.float32))
        rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        m, lo, hi = boot_ci(yte, rf.predict_proba(Xte)[:, 1])
        print(f"  {nome:32s} AUC={m:.3f}  IC95%=[{lo:.3f},{hi:.3f}]")
        return m, lo, hi, rf, cols

    print("=== Colon benigno vs maligno: la fragmentomica aggiunge segnale? ===")
    base = auc(FEATURE_COLS + NEW_COLS, "78 (74+4) [baseline]")
    frag = auc(FEATURE_COLS + NEW_COLS + FRAG_COLS, "78 + fragmentomica")
    only = auc(FRAG_COLS, "SOLO fragmentomica (7)")

    # importanza delle feature fragmentomiche nel modello completo
    rf = frag[3]; cols = frag[4]
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    print("\n  Ranking delle feature fragmentomiche (su", len(cols), "totali):")
    for c in FRAG_COLS:
        print(f"    {c:24s} importanza={imp[c]:.4f}  (posizione {int(imp.index.get_loc(c))+1})")

    delta = frag[0] - base[0]
    print(f"\n>>> Guadagno fragmentomica: {base[0]:.3f} -> {frag[0]:.3f}  (Δ={delta:+.3f})")
    print(">>> " + ("MIGLIORA (IC piu' alto)" if frag[1] > base[1] + 0.005 else
                    "nessun miglioramento reale: il segnale non e' nella fragmentomica"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([{"feature_set": "74+4 (baseline)", "AUC": round(base[0], 3), "IC_low": round(base[1], 3)},
                  {"feature_set": "74+4+fragmentomica", "AUC": round(frag[0], 3), "IC_low": round(frag[1], 3)},
                  {"feature_set": "solo fragmentomica", "AUC": round(only[0], 3), "IC_low": round(only[1], 3)}]
                 ).to_csv(os.path.join(RESULTS_DIR, "fragmentomics_colorectal.tsv"), sep="\t", index=False)
    print(f"\nSalvato in {RESULTS_DIR}/fragmentomics_colorectal.tsv")


if __name__ == "__main__":
    main()
