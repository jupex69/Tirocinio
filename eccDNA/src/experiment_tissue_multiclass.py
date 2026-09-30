"""Multiclasse al livello di SISTEMA/TESSUTO MALATO (tra i malati), cross-studio.

Idea: le malattie che condividono un tessuto si confondono tra loro -> le UNIAMO in
un'unica classe "sistema" (es. stomaco+colon+esofago = Gastrointestinale). Cosi'
identifichiamo il TESSUTO MALATO invece del singolo tumore, sperando di coprire piu'
di 9 classi. Valutazione cross-studio (leave-one-database-out), entrambe le direzioni.

Metodi: RandomForest (candidato migliore) + le 3 loss siamesi in FEW-SHOT (K=5):
prototipico euclideo, prototipico coseno, triplet batch-hard. Metrica: recall per
sistema (quali trasferiscono) + accuratezza vs caso. Feature: 10 descrittori.
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

from eccdna_utils import DESCRIPTOR_NAMES, read_fasta_stream, compute_sequence_descriptors
from train_multiclass import (standardize, train_prototypical, _class_prototypes, _proto_logits, SEED, DEVICE)
from experiment_losses_fewshot import train_metric_encoder
from models_pytorch import _to_tensor
from experiment_oneshot import _episode, COMPACT
from full_classifier import FASTA, RESULTS_DIR

SYSTEM = {
    "Stomach Cancer": "Gastrointestinale", "Colorectal Cancer": "Gastrointestinale",
    "Esophageal Cancer": "Gastrointestinale", "Colorectal Adenoma": "Gastrointestinale",
    "Pancreatic Cancer": "Gastrointestinale", "Liver Cancer": "Gastrointestinale",
    "Ovarian Cancer": "Ginecologico", "Cervical Adenocarcinoma": "Ginecologico",
    "Endometrial Cancer": "Ginecologico", "Cervical Cancer": "Ginecologico",
    "Breast Cancer": "Mammella",
    "Prostate Cancer": "Urogenitale", "Urinary Bladder Cancer": "Urogenitale",
    "Kidney Cancer": "Urogenitale", "Clear Cell Renal Cell Carcinoma": "Urogenitale",
    "Lymphoma": "Ematologico", "Leukemia": "Ematologico", "Multiple Myeloma": "Ematologico",
    "Hematopoietic Cancer": "Ematologico", "B-Cell Lymphoma": "Ematologico",
    "Primary Pulmonary Hypertension": "Cardiovascolare", "Dilated Cardiomyopathy": "Cardiovascolare",
    "Coronary Artery Disease": "Cardiovascolare", "Cerebrovascular Disease": "Cardiovascolare",
    "Glioblastoma": "Sistema nervoso", "Neuroblastoma": "Sistema nervoso",
    "Medulloblastoma": "Sistema nervoso", "Lower Grade Glioma Cancer": "Sistema nervoso",
    "Lung Cancer": "Polmonare", "Lung Adenocarcinoma": "Polmonare", "Lung Squamous Cell Cancer": "Polmonare",
    "Hypopharynx Cancer": "Testa-collo", "Head And Neck Cancer": "Testa-collo",
    "Oral Cavity Cancer": "Testa-collo", "Thyroid Cancer": "Testa-collo",
    "Melanoma": "Cute", "Skin Cancer": "Cute",
    "Sarcoma": "Connettivo", "Fibrosarcoma": "Connettivo", "Osteosarcoma": "Connettivo",
    "Ewing Sarcoma": "Connettivo", "Smarca4-Deficient Sarcoma Of Thorax": "Connettivo",
}
DBS = ["eccDNABase", "CircleBaseV2"]
CACHES = ["data/processed/alldisease_oneshot_features.tsv", "data/processed/oneshot_desc_features.tsv",
          "data/processed/fewshot_expanded_features.tsv"]
EXP_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/tissue_multiclass_features.tsv")
CAP = 500
MINW = 60   # min per DB per includere il sistema (cross-studio)


def load():
    cnt, buf = {}, {}
    for ch in pd.read_csv(COMPACT, sep="\t", usecols=["id", "label", "is_disease", "source_db"],
                          chunksize=400000, low_memory=False):
        ch = ch[ch["is_disease"] == 1].copy()
        ch["sys"] = ch["label"].map(SYSTEM)
        ch = ch.dropna(subset=["sys"])
        for (t, s), g in ch.groupby(["sys", "source_db"]):
            cnt[(t, s)] = cnt.get((t, s), 0) + len(g)
            b = buf.setdefault((t, s), [])
            if sum(len(x) for x in b) < CAP:
                b.append(g.head(CAP)[["id", "sys", "source_db"]].rename(columns={"sys": "tessuto"}))
    per = {}
    for (t, s), n in cnt.items():
        per.setdefault(t, {})[s] = n
    keep = [t for t, d in per.items() if len(d) == 2 and min(d.values()) >= MINW]
    print("Sistemi (tessuti malati) e campioni per DB:")
    for t in sorted(per, key=lambda t: -min(per[t].values()) if len(per[t]) == 2 else 0):
        d = per[t]; ok = "OK" if (len(d) == 2 and min(d.values()) >= MINW) else "--"
        print(f"  [{ok}] {t:<20} eccDNABase={d.get('eccDNABase',0):>7}  CircleBaseV2={d.get('CircleBaseV2',0):>7}")
    print(f"\nSistemi cross-testabili (>= {MINW}/DB): {len(keep)}  -> {sorted(keep)}\n")
    meta = pd.concat([pd.concat(buf[(t, s)]).head(CAP) for t in keep for s in DBS if (t, s) in buf], ignore_index=True)
    meta["id"] = meta["id"].astype(str)
    ft = None
    for c in CACHES:
        if os.path.exists(c):
            t = pd.read_csv(c, sep="\t", index_col=0); t.index = t.index.astype(str)
            ft = t if ft is None else pd.concat([ft, t])
    if ft is not None:
        ft = ft[~ft.index.duplicated()]
    ids = set(meta["id"]) - (set(ft.index) if ft is not None else set())
    if ids:
        feats = {}
        for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
            if "N" not in seq and len(seq) >= 4:
                feats[sid] = compute_sequence_descriptors(seq)
        new = pd.DataFrame.from_dict(feats, orient="index")
        ft = new if ft is None else pd.concat([ft, new]); ft = ft[~ft.index.duplicated()]; new.to_csv(EXP_CACHE, sep="\t")
    ftr = ft.reset_index().rename(columns={ft.reset_index().columns[0]: "id"}); ftr["id"] = ftr["id"].astype(str)
    return meta.merge(ftr[["id"] + DESCRIPTOR_NAMES], on="id", how="inner").dropna(subset=DESCRIPTOR_NAMES)


@torch.no_grad()
def episodic_eval(enc, Xs, ys, Xq, yq, nC, cosine, k=5, n_ep=400, seed=SEED):
    rng = np.random.default_rng(seed)
    Es = enc(_to_tensor(Xs, DEVICE)) if enc else _to_tensor(Xs, DEVICE)
    Eq = enc(_to_tensor(Xq, DEVICE)) if enc else _to_tensor(Xq, DEVICE)
    classes = np.array([c for c in range(nC) if (ys == c).sum() >= 1 and (yq == c).sum() >= 1])
    accs = []; rec = np.full((n_ep, nC), np.nan)
    for e in range(n_ep):
        si, sl, qi, ql, chosen = _episode(rng, ys, classes, len(classes), k, 10, y2=yq)
        C = _class_prototypes(Es[si], torch.as_tensor(sl, device=DEVICE), len(classes), cosine)
        pred = _proto_logits(Eq[qi], C, cosine).argmax(1).cpu().numpy()
        accs.append((pred == ql).mean())
        for j, c in enumerate(chosen):
            m = ql == j
            if m.any():
                rec[e, c] = (pred[m] == j).mean()
    return float(np.mean(accs)), np.nanmean(rec, axis=0)


def main():
    df = load()
    classes = sorted(df["tessuto"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    chance = 1 / nC
    X = df[DESCRIPTOR_NAMES].to_numpy(np.float32); y = df["tessuto"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()
    print(f"{nC} sistemi cross-testabili, {len(df)} sequenze, caso={chance:.3f}\n")
    a = db == DBS[0]; b = db == DBS[1]

    rec_rf, rec_few = {}, {loss: [] for loss in ["euclideo", "coseno", "triplet"]}
    rf_acc, few_acc = [], {loss: [] for loss in ["euclideo", "coseno", "triplet"]}
    per_rf = []
    for S_mask, Q_mask, tag in [(a, b, "ecc->CB2"), (b, a, "CB2->ecc")]:
        Xtr, Xte = standardize(X[S_mask], X[Q_mask]); ytr, yte = y[S_mask], y[Q_mask]
        # RandomForest (candidato migliore)
        rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced").fit(Xtr, ytr)
        pr = rf.predict(Xte)
        rf_acc.append(accuracy_score(yte, pr))
        per_rf.append(recall_score(yte, pr, average=None, labels=range(nC), zero_division=0))
        print(f"[{tag}] RF: acc={accuracy_score(yte,pr):.3f} macro-F1={f1_score(yte,pr,average='macro',zero_division=0):.3f} ({accuracy_score(yte,pr)/chance:.1f}x)")
        # siamese few-shot, 3 loss  (shuffle: i dati arrivano raggruppati per classe)
        perm = np.random.default_rng(SEED).permutation(len(Xtr))
        Xsh, ysh = Xtr[perm], ytr[perm]; va = int(0.85 * len(Xsh))
        for loss in ["euclideo", "coseno", "triplet"]:
            cos = loss == "coseno"
            if loss in ("euclideo", "coseno"):
                enc, _ = train_prototypical(Xsh[:va], ysh[:va], Xsh[va:], ysh[va:], nC, cosine=cos, seed=SEED,
                                            select_metric="balanced", n_support=5, n_query=5, epochs=70, episodes=30, patience=10)
            else:
                enc, _ = train_metric_encoder(Xsh[:va], ysh[:va], Xsh[va:], ysh[va:], nC, "triplet", margin=0.3,
                                              epochs=70, batches=30, patience=10)
            acc, rc = episodic_eval(enc, Xtr, ytr, Xte, yte, nC, cosine=(cos if loss != "triplet" else False), k=5)
            few_acc[loss].append(acc); rec_few[loss].append(rc)
            print(f"       siamese {loss:9s} few-shot(K=5): acc={acc:.3f} ({acc/chance:.1f}x)")

    # tabelle riassuntive
    print("\n=== ACCURATEZZA CROSS-STUDIO (media 2 direzioni), caso {:.3f} ===".format(chance))
    print(f"  RandomForest      {np.mean(rf_acc):.3f} ({np.mean(rf_acc)/chance:.1f}x)")
    for loss in ["euclideo", "coseno", "triplet"]:
        print(f"  siamese {loss:9s} {np.mean(few_acc[loss]):.3f} ({np.mean(few_acc[loss])/chance:.1f}x)")

    rf_rec = np.nanmean(per_rf, axis=0)
    tab = pd.DataFrame({"sistema": classes, "recall_RF": np.round(rf_rec, 3)})
    for loss in ["euclideo", "coseno", "triplet"]:
        tab[f"recall_siam_{loss}"] = np.round(np.nanmean(rec_few[loss], axis=0), 3)
    tab = tab.sort_values("recall_RF", ascending=False)
    print(f"\n=== RECALL PER SISTEMA (cross-studio, caso {chance:.3f}) ===")
    print(tab.to_string(index=False))
    n_id = int((tab["recall_RF"] > chance * 1.5).sum())
    print(f"\nSistemi con recall RF > 1.5x il caso ({chance*1.5:.3f}): {n_id}/{nC}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tab.to_csv(os.path.join(RESULTS_DIR, "tissue_multiclass_perclass.tsv"), sep="\t", index=False)
    print(f"Salvato in {RESULTS_DIR}/tissue_multiclass_perclass.tsv")


if __name__ == "__main__":
    main()
