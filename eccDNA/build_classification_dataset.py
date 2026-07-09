"""Costruisce un dataset bilanciato per il classificatore diretto sano/malato.

A differenza di triplet_generator.py (che genera ancore/positivi/negativi
per un futuro approccio a triplet loss), questo script produce semplicemente
un elenco di id con relativa label, bilanciato 50/50 sano/malato e, tra i
malati, il piu' possibile uniforme tra i diversi tipi di malattia (stessa
idea anti-bias-Cancro-Gastrico, si veda eccdna_utils.sample_balanced_ids).

Il campionamento avviene DENTRO ciascuno split_cluster (train/val/test),
mai a cavallo tra split, per evitare fughe di dati.

Le dimensioni di default sono pensate per un training CPU-only in tempi
ragionevoli (poche decine di migliaia di sequenze, non l'intero dataset
da 3,7M record).

Uso:
    python build_classification_dataset.py
    python build_classification_dataset.py --n-train 50000 --n-val 8000 --n-test 8000
"""

import argparse
import os

import pandas as pd

from eccdna_utils import sample_balanced_ids

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-clean-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_metadata_CLEAN.tsv"),
    )
    parser.add_argument("--n-train", type=int, default=30000)
    parser.add_argument("--n-val", type=int, default=6000)
    parser.add_argument("--n-test", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "data/classification"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("--- FASE 1: CARICAMENTO METADATI PULITI ---")
    df_meta = pd.read_csv(args.metadata_clean_path, sep="\t", low_memory=False)
    print(f"Righe totali disponibili: {len(df_meta)}")

    splits = {
        "train": (df_meta[df_meta["split_cluster"] == "train"], args.n_train),
        "val": (df_meta[df_meta["split_cluster"] == "val"], args.n_val),
        "test": (df_meta[df_meta["split_cluster"] == "test"], args.n_test),
    }

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n--- FASE 2: CAMPIONAMENTO BILANCIATO PER SPLIT ---")
    for split_name, (df_split, n) in splits.items():
        result = sample_balanced_ids(df_split, n, seed=args.seed)
        n_healthy = (result["label"] == 0).sum()
        n_disease = (result["label"] == 1).sum()
        n_disease_types = result.loc[result["label"] == 1, "disease"].nunique()
        print(
            f"{split_name}: {len(result)} righe "
            f"({n_healthy} sane, {n_disease} malate su {n_disease_types} tipi di malattia)"
        )

        out_path = os.path.join(args.output_dir, f"{split_name}_ids.csv")
        result.to_csv(out_path, index=False)
        print(f"  -> salvato in {out_path}")

    print("\n--- FASE 3: COMPLETAMENTO ---")
    print("Dataset di classificazione bilanciato pronto in", args.output_dir)


if __name__ == "__main__":
    main()
