"""Dataset multiclasse per la classificazione della malattia (17 classi).

A differenza del task binario sano/malato, qui si usano SOLO i campioni malati
(role=malato nel file di abbinamento), etichettati con la malattia specifica, e
si predice quale delle 17 malattie. Feature: i 10 descrittori di sequenza.
Split per cluster genomico (come nel binario). Bilanciamento con un tetto per
classe (--cap) per attenuare lo sbilanciamento (Gastric ~17k vs Liver ~130).

LIMITE DICHIARATO: il metodo di sequenziamento e' quasi un proxy della malattia
(alcune malattie sono ~100% un solo protocollo). Parte del segnale multiclasse
puo' quindi derivare dal protocollo e non dalla biologia. Questo e' un primo
esperimento esplorativo; il limite va tenuto presente nell'interpretazione.
"""

import numpy as np
import pandas as pd

from eccdna_utils import DESCRIPTOR_NAMES
from training_data import (
    DEFAULT_FEATURES_PATH, DEFAULT_METADATA_PATH, DEFAULT_PAIRING_PATH,
    _load_features, _load_metadata_for_ids,
)

CAP_PER_DISEASE = 3000
SEED = 42

# Classi con meno campioni di questa soglia ricevono uno split stratificato a
# livello di campione (70/15/15) invece dello split per cluster genomico: con
# ~130-800 campioni lo split per cluster produce partizioni degeneri (es.
# Systemic lupus con val=0, Liver con test=3), inutilizzabili per selezione e
# valutazione. Il compromesso e' un lieve rischio di leakage intra-cluster su
# poche decine di sequenze; accettabile e dichiarato, dato che l'alternativa e'
# non poter valutare affatto le malattie rare (l'obiettivo dello studio).
RARE_TOTAL_THRESHOLD = 1000


def _cap_per_class(df, cap, seed):
    parti = [g.sample(n=min(len(g), cap), random_state=seed) for _, g in df.groupby("disease")]
    return pd.concat(parti, ignore_index=True)


def _stratified_rare_split(ids, seed, val_frac=0.15, test_frac=0.15):
    """Assegna train/val/test a livello di campione garantendo almeno 1 elemento
    in val e in test (idealmente ~15% ciascuno). Ritorna dict id -> split."""
    rng = np.random.default_rng(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    assign = {}
    for i, sid in enumerate(ids):
        if i < n_test:
            assign[sid] = "test"
        elif i < n_test + n_val:
            assign[sid] = "val"
        else:
            assign[sid] = "train"
    return assign


def _repair_rare_splits(malato, threshold, seed):
    """Sostituisce lo split per cluster con uno stratificato per le classi rare."""
    counts = malato["disease"].value_counts()
    rare = counts[counts < threshold].index
    for j, disease in enumerate(sorted(rare)):
        ids = malato.loc[malato["disease"] == disease, "id"]
        assign = _stratified_rare_split(ids, seed + j)
        malato.loc[ids.index, "split"] = ids.map(assign).values
    return malato, list(rare)


def build_multiclass_splits(cap=CAP_PER_DISEASE, seed=SEED):
    """Ritorna (train_df, val_df, test_df) per la classificazione multiclasse.
    Ogni riga: id, disease, i 10 descrittori, split. Solo campioni malati.
    Il tetto per classe si applica a tutti gli split."""
    feats = _load_features(DEFAULT_FEATURES_PATH).set_index("id")

    pairing = pd.read_csv(DEFAULT_PAIRING_PATH, sep="\t")
    pairing["id"] = pairing["id"].astype(str)
    malato = pairing[pairing["role"] == "malato"][["disease", "id"]].drop_duplicates("id")
    malato = malato[malato["id"].isin(feats.index)]

    split_series = _load_metadata_for_ids(DEFAULT_METADATA_PATH, set(malato["id"])).set_index("id")["split_cluster"]
    malato["split"] = malato["id"].map(split_series)

    # classi rare: split stratificato per campione (evita val/test vuoti)
    malato = malato.reset_index(drop=True)
    malato, rare = _repair_rare_splits(malato, RARE_TOTAL_THRESHOLD, seed)

    # unisci i descrittori
    desc = feats.loc[malato["id"], DESCRIPTOR_NAMES].reset_index(drop=True)
    malato = pd.concat([malato.reset_index(drop=True), desc], axis=1).dropna(subset=DESCRIPTOR_NAMES)

    out = []
    for nome in ("train", "val", "test"):
        part = malato[malato["split"] == nome]
        part = _cap_per_class(part, cap, seed).reset_index(drop=True)
        out.append(part)
    return out


def summarize(train_df, val_df, test_df):
    diseases = sorted(train_df["disease"].unique())
    print(f"Classi (malattie): {len(diseases)}")
    tab = pd.DataFrame({
        "train": train_df["disease"].value_counts(),
        "val": val_df["disease"].value_counts(),
        "test": test_df["disease"].value_counts(),
    }).fillna(0).astype(int).sort_values("train", ascending=False)
    print(tab.to_string())
    print(f"\nTotale: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == "__main__":
    tr, va, te = build_multiclass_splits()
    summarize(tr, va, te)
