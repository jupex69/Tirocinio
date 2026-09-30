"""Assemblaggio del dataset di training per il classificatore binario sano/malato.

Unisce i descrittori (eccdna_descriptor_features.tsv, prodotto da
descriptor_extractor.py) con l'etichetta binaria e la malattia specifica.

SPLIT: si usa 'split_cluster' (train/val/test), non uno split casuale.
Motivo (vedi README_disease_detection.md del tutor): 'split_cluster' raggruppa
per cluster genomico, riducendo il rischio che regioni molto simili finiscano
sia in train sia in test - split_random darebbe risultati troppo ottimistici.

BILANCIAMENTO (build_balanced_splits): il dataset grezzo NON e' bilanciato.
Il file dei descrittori deduplica i sani (una stessa sequenza sana e' abbinata
a piu' malattie ma compare una volta sola), quindi un semplice "tutti i malati
vs tutti i sani" da' ~72% malato / 28% sano; inoltre 4 tumori gastrointestinali
sono ~70% dei malati. Due sbilanciamenti che spingerebbero i modelli a imparare
"tumore GI" invece di "malattia".

La soluzione rispetta il vincolo centrale del progetto (i sani sono
method-matched a CIASCUNA malattia, vedi eccdna_disease_pairing.tsv): si
ricostruisce il dataset dal file di abbinamento, prendendo per ogni malattia
un numero UGUALE di malati e dei suoi sani method-matched, con un tetto per
malattia (--cap) per attenuare il dominio dei GI. Cosi' si ottiene 50/50
sano/malato E method-matching preservato dentro ogni blocco-malattia. Bilanciare
"a caso" (es. tagliando i GI senza guardare il metodo) reintrodurrebbe il
confondente, perche' tipo-di-malattia e metodo sono correlati (GI=Circle_seq,
altre=ATAC-seq): buttare via i GI sposterebbe il mix di metodi dei malati senza
aggiornare i sani.

NOTA sui sani ripetuti: un sano abbinato a piu' malattie puo' comparire piu'
volte nel set bilanciato (una per blocco-malattia). E' voluto: ogni malattia ha
diritto al proprio controllo sano method-matched. La ripetizione e' comunque
CONFINATA a uno stesso split (ogni id sta in un solo split_cluster), quindi non
crea leakage train/test.

La colonna 'disease' non e' mai una feature del modello (va esclusa da X):
serve per il campionamento anti-bias e per il breakdown per-malattia a valle.
"""

import os

import numpy as np
import pandas as pd

from eccdna_utils import DESCRIPTOR_NAMES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FEATURES_PATH = os.path.join(SCRIPT_DIR, "data/processed/eccdna_descriptor_features.tsv")
DEFAULT_METADATA_PATH = os.path.join(SCRIPT_DIR, "data/processed/eccdna_metadata_CLEAN.tsv")
DEFAULT_PAIRING_PATH = os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_pairing.tsv")
METADATA_CHUNK_SIZE = 250_000
DEFAULT_CAP_PER_DISEASE = 3000  # tetto di malati (e relativi sani) per malattia in train, per attenuare il dominio GI


def _load_metadata_for_ids(metadata_path, wanted_ids, chunk_size=METADATA_CHUNK_SIZE):
    """Legge id/split_cluster a blocchi (il file e' 385MB), tenendo solo le
    righe i cui id servono - stesso pattern RAM-safe usato altrove (es.
    load_method_series_full)."""
    colonne = ["id", "disease_binary_label", "disease", "split_cluster"]
    parti = []
    for chunk in pd.read_csv(metadata_path, sep="\t", usecols=colonne, chunksize=chunk_size, low_memory=False):
        chunk["id"] = chunk["id"].astype(str)
        parti.append(chunk[chunk["id"].isin(wanted_ids)])
    return pd.concat(parti, ignore_index=True)


def _load_features(features_path):
    df_features = pd.read_csv(features_path, sep="\t")
    df_features["id"] = df_features["id"].astype(str)
    return df_features.dropna(subset=DESCRIPTOR_NAMES)


def load_dataset(features_path=DEFAULT_FEATURES_PATH, metadata_path=DEFAULT_METADATA_PATH):
    """Dataset GREZZO (non bilanciato), una riga per id: id, descrittori,
    disease, y (0/1), split_cluster. Utile per ispezione/confronto; per il
    training usare build_balanced_splits."""
    df_features = _load_features(features_path)
    df_meta = _load_metadata_for_ids(metadata_path, set(df_features["id"]))
    df = df_features.merge(df_meta, on="id", how="inner")
    df = df.rename(columns={"disease_binary_label": "y"})
    df["y"] = df["y"].astype(int)
    return df


def get_splits(df):
    """Divide per split_cluster (train/val/test), come raccomandato dal tutor."""
    train_df = df[df["split_cluster"] == "train"].reset_index(drop=True)
    val_df = df[df["split_cluster"] == "val"].reset_index(drop=True)
    test_df = df[df["split_cluster"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def _balance_one_split(pairing_split, descriptors_by_id, cap, seed):
    """Per un singolo split: per ciascuna malattia prende min(n_malato,
    n_sano_unici, cap) malati e altrettanti dei suoi sani method-matched,
    campionati. Ritorna il DataFrame bilanciato (descrittori + disease + y).
    """
    rng = np.random.RandomState(seed)
    righe_malato, righe_sano = [], []

    for disease, gruppo in pairing_split.groupby("disease"):
        malato_ids = gruppo.loc[gruppo["role"] == "malato", "id"]
        sano_ids = gruppo.loc[gruppo["role"] == "sano", "id"].unique()
        # solo id che hanno effettivamente i descrittori calcolati
        malato_ids = [i for i in malato_ids if i in descriptors_by_id.index]
        sano_ids = [i for i in sano_ids if i in descriptors_by_id.index]

        n = min(len(malato_ids), len(sano_ids), cap)
        if n == 0:
            continue
        scelti_malato = rng.choice(malato_ids, size=n, replace=False)
        scelti_sano = rng.choice(sano_ids, size=n, replace=False)

        for _id in scelti_malato:
            righe_malato.append((_id, disease, 1))
        for _id in scelti_sano:
            righe_sano.append((_id, disease, 0))  # 'disease' = malattia a cui il sano e' abbinato (contesto, non label)

    righe = righe_malato + righe_sano
    meta = pd.DataFrame(righe, columns=["id", "disease", "y"])
    feats = descriptors_by_id.loc[meta["id"]].reset_index(drop=True)
    out = pd.concat([meta.reset_index(drop=True), feats], axis=1)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def build_balanced_splits(features_path=DEFAULT_FEATURES_PATH, metadata_path=DEFAULT_METADATA_PATH,
                          pairing_path=DEFAULT_PAIRING_PATH, cap=DEFAULT_CAP_PER_DISEASE, seed=42):
    """Costruisce train/val/test BILANCIATI (50/50 sano/malato, method-matched
    per malattia, dominio GI attenuato dal tetto per malattia) usando il file
    di abbinamento. Vedi docstring del modulo per la logica completa.

    Il tetto 'cap' si applica a tutti gli split allo stesso modo; val e test
    hanno naturalmente meno sequenze per malattia, quindi il tetto morde
    soprattutto sul train (dove i GI hanno ~17k malati ciascuno).

    Ritorna (train_df, val_df, test_df).
    """
    df_features = _load_features(features_path).set_index("id")

    pairing = pd.read_csv(pairing_path, sep="\t")
    pairing["id"] = pairing["id"].astype(str)

    split_series = _load_metadata_for_ids(metadata_path, set(pairing["id"])).set_index("id")["split_cluster"]
    pairing["split"] = pairing["id"].map(split_series)

    splits = {}
    for nome in ("train", "val", "test"):
        pairing_split = pairing[pairing["split"] == nome]
        splits[nome] = _balance_one_split(pairing_split, df_features, cap, seed)
    return splits["train"], splits["val"], splits["test"]


def summarize(df, name="dataset"):
    n_malato = int((df["y"] == 1).sum())
    n_sano = int((df["y"] == 0).sum())
    frazione = n_malato / len(df) if len(df) else 0
    print(f"{name}: {len(df)} righe ({n_malato} malato, {n_sano} sano; malato={frazione:.1%})")
    if n_malato > 0:
        print(df.loc[df["y"] == 1, "disease"].value_counts().to_string())


if __name__ == "__main__":
    print(f"=== DATASET BILANCIATO (cap={DEFAULT_CAP_PER_DISEASE}/malattia) ===")
    train_df, val_df, test_df = build_balanced_splits()
    summarize(train_df, "train")
    summarize(val_df, "val")
    summarize(test_df, "test")
