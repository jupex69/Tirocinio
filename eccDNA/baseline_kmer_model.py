"""Baseline veloce sano/malato basata sulle frequenze di k-meri.

Riusa il file eccdna_kmer_3_features.tsv gia' completamente generato (non
viene rigenerato: l'estrazione su 3,7M sequenze richiederebbe ore) e lo
allinea agli stessi id bilanciati prodotti da build_classification_dataset.py,
cosi' il confronto con train_classifier.py (CNN) e' su esattamente lo stesso
train/val/test.

Allena Logistic Regression e Random Forest (scikit-learn) sulle frequenze
di k-meri - lo stesso spirito della baseline TF-IDF + Logistic Regression
del vecchio main.py (cancellato), ma senza il leakage di allora e allineata
al dataset bilanciato per tipo di malattia.

Uso:
    python baseline_kmer_model.py
    python baseline_kmer_model.py --kmer-features-path data/processed/eccdna_kmer_4_features.tsv
"""

import argparse
import json
import os
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNK_SIZE = 250000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(SCRIPT_DIR, "data/classification"))
    parser.add_argument(
        "--kmer-features-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_kmer_3_features.tsv"),
    )
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "data/models_baseline"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_matching_kmer_rows(kmer_features_path, wanted_ids):
    print(f"Scansione a blocchi di {kmer_features_path} (filtro su {len(wanted_ids)} id)...")
    t0 = time.time()
    matched_chunks = []
    rows_seen = 0
    for chunk in pd.read_csv(kmer_features_path, sep="\t", chunksize=CHUNK_SIZE):
        rows_seen += len(chunk)
        matched = chunk[chunk["id"].astype(str).isin(wanted_ids)]
        if len(matched):
            matched_chunks.append(matched)
        if rows_seen % (CHUNK_SIZE * 4) == 0:
            print(f"  ...{rows_seen} righe scansionate, {sum(len(m) for m in matched_chunks)} trovate ({time.time() - t0:.0f}s)")

    df = pd.concat(matched_chunks, ignore_index=True) if matched_chunks else pd.DataFrame()
    print(f"Trovate {len(df)}/{len(wanted_ids)} righe in {time.time() - t0:.0f}s")
    return df


def evaluate_model(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
    }, confusion_matrix(y, preds).tolist()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- FASE 1: CARICAMENTO ID BILANCIATI ---")
    splits = {}
    for name in ["train", "val", "test"]:
        splits[name] = pd.read_csv(os.path.join(args.data_dir, f"{name}_ids.csv"))
        print(f"{name}: {len(splits[name])} righe")

    all_ids = set()
    for df in splits.values():
        all_ids.update(df["id"].astype(str))

    print("\n--- FASE 2: CARICAMENTO FEATURE K-MER ALLINEATE AGLI ID BILANCIATI ---")
    kmer_df = load_matching_kmer_rows(args.kmer_features_path, all_ids)
    kmer_cols = [c for c in kmer_df.columns if c != "id"]

    def build_xy(split_name):
        ids_df = splits[split_name][["id", "label"]].copy()
        ids_df["id"] = ids_df["id"].astype(str)
        merged = ids_df.merge(kmer_df, on="id", how="inner")
        dropped = len(ids_df) - len(merged)
        if dropped:
            print(f"  {split_name}: {dropped} id senza feature k-mer, scartati.")
        return merged[kmer_cols].values, merged["label"].values

    X_train, y_train = build_xy("train")
    X_val, y_val = build_xy("val")
    X_test, y_test = build_xy("test")
    print(f"train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

    print("\n--- FASE 3: TRAINING ---")
    results = {}

    print("Logistic Regression...")
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    logreg.fit(X_train, y_train)
    val_metrics, _ = evaluate_model(logreg, X_val, y_val)
    test_metrics, cm = evaluate_model(logreg, X_test, y_test)
    print("  val: ", val_metrics)
    print("  test:", test_metrics)
    results["logistic_regression"] = {"val_metrics": val_metrics, "test_metrics": test_metrics, "confusion_matrix": cm}

    print("\nRandom Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=args.seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    val_metrics, _ = evaluate_model(rf, X_val, y_val)
    test_metrics, cm = evaluate_model(rf, X_test, y_test)
    print("  val: ", val_metrics)
    print("  test:", test_metrics)
    results["random_forest"] = {"val_metrics": val_metrics, "test_metrics": test_metrics, "confusion_matrix": cm}

    results["n_train"] = len(X_train)
    results["n_val"] = len(X_val)
    results["n_test"] = len(X_test)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- FASE 4: COMPLETAMENTO ---")
    print("Metriche salvate in", os.path.join(args.output_dir, "metrics.json"))


if __name__ == "__main__":
    main()
