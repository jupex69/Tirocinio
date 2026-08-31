"""Test dello Stadio 1 (binario) per il sistema a due stadi: si puo' fare meglio
del 0.68 dentro lo strato, o conviene usare il binario piu' ampio della tesi?

(B) DENTRO LO STRATO (rigore massimo: metodo+studio+lunghezza controllati):
    prova 10 descrittori vs 74 feature, con GBM / RF / MLP profondo.
(A) BINARIO DELLA TESI applicato allo strato: si allena sul dataset ampio
    (17 malattie, method-matched, 10 descrittori) e si valuta sul test dello strato,
    escludendo gli id gia' visti in addestramento (anti-leakage).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from eccdna_utils import DESCRIPTOR_NAMES, read_fasta_stream, compute_sequence_descriptors
from train_siamese_multiclass import kmer_spectrum, KMER_COLS, FEATURE_COLS
from training_data import build_balanced_splits
from gold_standard_data import load_stratum_table, length_matched_binary, SEED
from models_pytorch import DeepMLP, train_binary_classifier, predict_proba

FASTA = "data/processed/eccdna_disease_detection.body.fa"


def auc_tree(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    return roc_auc_score(yte, model.predict_proba(Xte)[:, 1])


def auc_mlp(cols, tr, va, te, ytr, yva, yte):
    Xtr, Xva, Xte = tr[cols].to_numpy(np.float64), va[cols].to_numpy(np.float64), te[cols].to_numpy(np.float64)
    m, s = Xtr.mean(0), Xtr.std(0); s[s == 0] = 1.0
    Xtr, Xva, Xte = (Xtr - m) / s, (Xva - m) / s, (Xte - m) / s
    mlp = DeepMLP(n_features=len(cols))
    mlp, _ = train_binary_classifier(mlp, Xtr, ytr, Xva, yva, seed=SEED)
    return roc_auc_score(yte, predict_proba(mlp, Xte))


def main():
    print("--- Test dello Stadio 1 (binario) ---")
    base = load_stratum_table()
    bn = length_matched_binary(base)
    print(f"Lettura FASTA e feature per {len(set(bn['id']))} sequenze dello strato...")
    feats = {}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=set(bn["id"])):
        if "N" not in seq and len(seq) >= 4:
            f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq)); feats[sid] = f
    ft = pd.DataFrame.from_dict(feats, orient="index").reset_index().rename(columns={"index": "id"})
    ft["id"] = ft["id"].astype(str); bn["id"] = bn["id"].astype(str)
    bn = bn.merge(ft, on="id", how="inner").dropna(subset=FEATURE_COLS)
    tr, va, te = bn[bn.split == "train"], bn[bn.split == "val"], bn[bn.split == "test"]
    ytr, yva, yte = tr["is_disease"].to_numpy(), va["is_disease"].to_numpy(), te["is_disease"].to_numpy()

    def gbm(): return HistGradientBoostingClassifier(random_state=SEED)
    def rf(): return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)

    print("\n=== (B) DENTRO LO STRATO (rigore massimo) — ROC-AUC sul test dello strato ===")
    for cols, etich in [(DESCRIPTOR_NAMES, "10 descrittori"), (FEATURE_COLS, "74 feature (k-mer+desc)")]:
        Xtr, Xte = tr[cols].to_numpy(np.float64), te[cols].to_numpy(np.float64)
        a_gbm = auc_tree(gbm(), Xtr, ytr, Xte, yte)
        a_rf = auc_tree(rf(), Xtr, ytr, Xte, yte)
        a_mlp = auc_mlp(cols, tr, va, te, ytr, yva, yte)
        print(f"  {etich:26s}  GBM={a_gbm:.3f}  RF={a_rf:.3f}  MLP={a_mlp:.3f}")

    print("\n=== (A) BINARIO DELLA TESI applicato allo strato (10 descrittori) ===")
    tr17, va17, te17 = build_balanced_splits()
    seen = set(tr17["id"].astype(str)) | set(va17["id"].astype(str))
    te_clean = te[~te["id"].astype(str).isin(seen)]  # anti-leakage
    print(f"  test dello strato dopo rimozione id gia' visti: {len(te_clean)}/{len(te)}")
    Xtr17 = tr17[DESCRIPTOR_NAMES].to_numpy(np.float64); ytr17 = tr17["y"].to_numpy()
    model = gbm().fit(Xtr17, ytr17)
    auc_own = roc_auc_score(te17["y"].to_numpy(), model.predict_proba(te17[DESCRIPTOR_NAMES].to_numpy(np.float64))[:, 1])
    auc_strato = roc_auc_score(te_clean["is_disease"].to_numpy(),
                               model.predict_proba(te_clean[DESCRIPTOR_NAMES].to_numpy(np.float64))[:, 1])
    print(f"  AUC sul suo test (17 malattie, riferimento): {auc_own:.3f}")
    print(f"  AUC applicato al test dello STRATO:          {auc_strato:.3f}")
    print("\n  => confronta con il 0.68 del cascade: scegliamo lo Stadio 1 migliore con i numeri veri.")


if __name__ == "__main__":
    main()
