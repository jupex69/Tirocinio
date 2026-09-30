"""SOLUZIONE al confondente 'metodo': multiclasse a METODO COSTANTE.

Il problema (vedi results/multiclass_confounder_profile.tsv): sul task a 17
malattie quasi ogni malattia e' ~100% un solo metodo di sequenziamento e UN SOLO
studio -> il modello puo' riconoscere il protocollo invece della biologia.

Il de-biasing adversariale NON e' applicabile: metodo e malattia sono
perfettamente confusi (rimuovere l'uno rimuove l'altro). L'unica soluzione
corretta e' TENERE IL METODO COSTANTE. Il gold-standard lo faceva ma, per una
whitelist di tessuti e una soglia a 2000 campioni, restava a 5 tessuti.

Qui si tiene lo stesso strato omogeneo (source_db=CircleBaseV2, method=Circle_seq:
metodo E database COSTANTI per costruzione) ma si conservano TUTTE le malattie
con >= MIN_N campioni: ne sopravvivono ~8-9 invece di 5. Le lunghezze restano
matchate per quantili tra le classi. Cosi' il metodo non e' piu' una scorciatoia
(e' identico per tutte) e si testa piu' diversita' di malattie sul segnale pulito.

CONFRONTO CHIAVE: quanto vale l'accuratezza in multipli del caso, rispetto al
task a 17 classi confuso (~4.4x il caso)? Se scende verso ~2x, la parte in
eccesso del 17-classi era metodo/studio.

LIMITE RESIDUO DICHIARATO: anche a metodo+DB costanti, ogni malattia resta un
singolo studio -> un confondente di batch a livello di studio NON e' eliminabile
con questi dati (servirebbero piu' studi indipendenti per malattia sotto lo
stesso protocollo). Questo e' il massimo controllo ottenibile qui.

Riusa training/valutazione dai due esperimenti gemelli. Risultati in results/
con prefisso 'single_method_'.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score, f1_score

from eccdna_utils import read_fasta_stream, compute_sequence_descriptors
from gold_standard_data import length_matched_multiclass
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from train_multiclass import (
    standardize, train_prototypical, predict_prototypical,
    train_softmax, predict_softmax, evaluate, SEED, DEVICE,
)
from experiment_losses_fewshot import train_metric_encoder, few_shot_eval, k_shot_full

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection_metadata.tsv")
FASTA = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection.body.fa")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CACHE = os.path.join(SCRIPT_DIR, "data/processed/single_method_features.tsv")

CHUNK = 400000
COLLECT_CAP = 8000     # tetto di raccolta per malattia (limita il calcolo feature)
MIN_N = 500            # malattie con almeno tanti campioni nello strato
N_BINS = 8             # bin di quantile per il length-matching

# igiene etichette: fondi i sinonimi ovvii dello stomaco (come canon_tissue)
SYN = {"gastric cancer": "Stomach", "stomach": "Stomach"}


def _canon(d):
    return SYN.get(str(d).lower(), str(d))


def load_single_method_base():
    """Strato CircleBaseV2 (metodo Circle_seq costante), etichette = malattia
    grezza canonicalizzata, con tetto di raccolta per classe."""
    parts = []
    for ch in pd.read_csv(META, sep="\t",
                          usecols=["id", "disease", "disease_binary_name", "source_db",
                                   "method", "length", "split_cluster"],
                          chunksize=CHUNK, low_memory=False):
        ch = ch[ch["source_db"] == "CircleBaseV2"]
        ch = ch[ch["disease_binary_name"].astype(str).str.lower() != "healthy"]
        ch = ch.assign(gruppo=ch["disease"].map(_canon), is_disease=1)
        parts.append(ch[["id", "gruppo", "is_disease", "length", "split_cluster"]]
                     .rename(columns={"split_cluster": "split"}))
    df = pd.concat(parts, ignore_index=True)
    df["id"] = df["id"].astype(str)
    df["split"] = df["split"].astype(str)
    return df.groupby("gruppo", group_keys=False).head(COLLECT_CAP).reset_index(drop=True)


def get_rich(force=False):
    base = load_single_method_base()
    mc = length_matched_multiclass(base, min_n=MIN_N, n_bins=N_BINS, seed=SEED)
    mc["id"] = mc["id"].astype(str)
    if os.path.exists(CACHE) and not force:
        ftab = pd.read_csv(CACHE, sep="\t", index_col=0)
        ftab.index = ftab.index.astype(str)
    else:
        ids = set(mc["id"])
        feats = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
            if "N" not in seq and len(seq) >= 4:
                f = kmer_spectrum(seq)
                f.update(compute_sequence_descriptors(seq))
                feats[sid] = f
        ftab = pd.DataFrame.from_dict(feats, orient="index")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        ftab.to_csv(CACHE, sep="\t")
    mc = mc.set_index("id").join(ftab, how="inner").reset_index().dropna(subset=FEATURE_COLS)
    return mc


def per_class_report(y_true, proba, classes):
    y_pred = proba.argmax(1)
    rec = recall_score(y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    return pd.DataFrame({"malattia": classes, "recall": np.round(rec, 3), "f1": np.round(f1, 3)})


def main(quick=False, force=False):
    ep = dict(epochs=8, batches=8, episodes=8, patience=4) if quick else dict(epochs=200, batches=40, episodes=40, patience=20)
    n_episodes = 60 if quick else 400
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("--- Multiclasse a METODO COSTANTE (CircleBaseV2 / Circle_seq, length-matched) ---")
    mc = get_rich(force=force)
    classes = sorted(mc["gruppo"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    print(f"Malattie ({n_classes}): {classes}")
    print(pd.crosstab(mc["gruppo"], mc["split"]).to_string())

    tr = mc[mc.split == "train"]; va = mc[mc.split == "val"]; te = mc[mc.split == "test"]
    Xtr = tr[FEATURE_COLS].to_numpy(np.float32); Xva = va[FEATURE_COLS].to_numpy(np.float32); Xte = te[FEATURE_COLS].to_numpy(np.float32)
    ytr = tr["gruppo"].map(c2i).to_numpy(); yva = va["gruppo"].map(c2i).to_numpy(); yte = te["gruppo"].map(c2i).to_numpy()
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)
    chance = 1 / n_classes
    print(f"\ntrain={len(Xtr)} val={len(Xva)} test={len(Xte)}  chance={chance:.3f}\n")

    print("=== Addestramento varianti (ribilanciate, metodo costante) ===")
    encoders = {}
    enc, v = train_prototypical(Xtr, ytr, Xva, yva, n_classes, cosine=True, seed=SEED,
                                select_metric="balanced", epochs=ep["epochs"], episodes=ep["episodes"], patience=ep["patience"])
    encoders["Prototipico coseno"] = (enc, True); print(f"  Prototipico coseno   val_bal_acc={v:.3f}")
    enc, v = train_prototypical(Xtr, ytr, Xva, yva, n_classes, cosine=False, seed=SEED,
                                select_metric="balanced", epochs=ep["epochs"], episodes=ep["episodes"], patience=ep["patience"])
    encoders["Prototipico euclideo"] = (enc, False); print(f"  Prototipico euclideo val_bal_acc={v:.3f}")
    enc, v = train_metric_encoder(Xtr, ytr, Xva, yva, n_classes, "triplet", margin=0.3,
                                  epochs=ep["epochs"], batches=ep["batches"], patience=ep["patience"])
    encoders["Triplet (batch-hard)"] = (enc, False); print(f"  Triplet              val_bal_acc={v:.3f}")
    enc, v = train_metric_encoder(Xtr, ytr, Xva, yva, n_classes, "contrastive", margin=1.0,
                                  epochs=ep["epochs"], batches=ep["batches"], patience=ep["patience"])
    encoders["Contrastive (coppie)"] = (enc, False); print(f"  Contrastive          val_bal_acc={v:.3f}")
    smax = train_softmax(Xtr, ytr, Xva, yva, n_classes, seed=SEED, balanced=True,
                         epochs=ep["epochs"] * (25 if quick else 1), patience=ep["patience"])

    # ---- (1) multiclasse standard ----
    print(f"\n=== (1) Multiclasse a metodo costante su test ({n_classes} malattie) ===")
    rows, per_class = [], {}
    for name, (enc, cos) in encoders.items():
        proba = predict_prototypical(enc, Xtr, ytr, Xte, n_classes, cosine=cos)
        r = evaluate(name, yte, proba, n_classes); r["x_chance"] = round(r["accuracy"] / chance, 2)
        rows.append(r); per_class[name] = per_class_report(yte, proba, classes)
    proba = predict_softmax(smax, Xte)
    r = evaluate("Softmax bilanciato", yte, proba, n_classes); r["x_chance"] = round(r["accuracy"] / chance, 2)
    rows.append(r); per_class["Softmax bilanciato"] = per_class_report(yte, proba, classes)
    res = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    print("\n" + res.round(3).to_string(index=False))
    res.to_csv(os.path.join(RESULTS_DIR, "single_method_loss_comparison.tsv"), sep="\t", index=False)

    # per-classe del miglior modello
    best = res.iloc[0]["model"]
    print(f"\n--- Per classe (miglior modello: {best}) ---")
    print(per_class[best].sort_values("f1", ascending=False).to_string(index=False))
    per_class[best].to_csv(os.path.join(RESULTS_DIR, "single_method_per_class.tsv"), sep="\t", index=False)

    # ---- (2) few-shot episodico ----
    print(f"\n=== (2) Few-shot episodico su test ===")
    fs_rows = []
    ways = sorted({min(3, n_classes), min(5, n_classes), n_classes})
    for name, (enc, cos) in encoders.items():
        for n_way in ways:
            for k_shot in (1, 5):
                m, sd, nw = few_shot_eval(enc, Xte, yte, cos, n_way, k_shot, n_episodes=n_episodes)
                fs_rows.append({"modello": name, "n_way": nw, "k_shot": k_shot,
                                "tipo": "one-shot" if k_shot == 1 else f"{k_shot}-shot",
                                "acc_media": round(m, 3), "acc_std": round(sd, 3), "chance": round(1 / nw, 3)})
    fs = pd.DataFrame(fs_rows).drop_duplicates(["modello", "n_way", "k_shot"])
    print("\n" + fs.to_string(index=False))
    fs.to_csv(os.path.join(RESULTS_DIR, "single_method_fewshot.tsv"), sep="\t", index=False)

    print(f"\nRisultati salvati in {RESULTS_DIR}/: single_method_loss_comparison.tsv, "
          f"single_method_per_class.tsv, single_method_fewshot.tsv")
    print(f"\nCONFRONTO: 17-classi confuso ~4.4x il caso  vs  metodo-costante {res.iloc[0]['x_chance']}x il caso")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
