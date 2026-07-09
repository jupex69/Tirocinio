"""Training del classificatore CNN diretto sano/malato per eccDNA.

Pipeline:
  1. carica gli id bilanciati prodotti da build_classification_dataset.py
     (data/classification/{train,val,test}_ids.csv);
  2. legge in streaming dal FASTA SOLO le sequenze richieste (poche decine
     di migliaia, non i 3,7M dell'intero dataset -> sostenibile in RAM anche
     su CPU);
  3. le codifica una sola volta con one_hot_encode_circular (augmentation
     circolare a finestra fissa, si veda eccdna_utils.py);
  4. allena il CNN di model.py con early stopping sull'AUC di validazione;
  5. valuta sul test set con le stesse metriche usate nei paper di
     riferimento (accuracy, precision, recall, F1, ROC-AUC, matrice di
     confusione) per confrontabilita' diretta.

Uso:
    python train_classifier.py
    python train_classifier.py --epochs 15 --window-size 1024
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from eccdna_utils import one_hot_encode_circular, read_fasta_stream
from model import EccDNACNN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(SCRIPT_DIR, "data/classification"))
    parser.add_argument(
        "--fasta-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection.body.fa"),
    )
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--wrap-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="epoche senza miglioramento prima dell'early stopping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "data/models"))
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def use_all_cpu_threads():
    # Su questa macchina non c'e' GPU: usiamo tutti i core CPU disponibili
    # per il training (di default PyTorch ne usa solo una parte).
    n_cores = os.cpu_count() or 1
    torch.set_num_threads(n_cores)
    print(f"PyTorch impostato su {n_cores} thread CPU.")


def load_split_ids(data_dir):
    splits = {}
    for name in ["train", "val", "test"]:
        splits[name] = pd.read_csv(os.path.join(data_dir, f"{name}_ids.csv"))
    return splits


def load_needed_sequences(fasta_path, wanted_ids):
    wanted_ids = set(wanted_ids)
    sequences = {}
    processed = 0
    t0 = time.time()
    for seq_id, seq in read_fasta_stream(fasta_path, wanted_ids=wanted_ids):
        sequences[seq_id] = seq
        processed += 1
        if processed % 10000 == 0:
            print(f"  ...{processed}/{len(wanted_ids)} sequenze richieste trovate ({time.time() - t0:.0f}s)")
        if processed == len(wanted_ids):
            break
    print(f"Sequenze caricate: {len(sequences)}/{len(wanted_ids)} in {time.time() - t0:.0f}s")
    return sequences


def encode_split(df_ids, sequences, window_size, wrap_len):
    encoded = []
    labels = []
    missing = 0
    for _, row in df_ids.iterrows():
        seq = sequences.get(str(row["id"]))
        if seq is None or len(seq) == 0:
            missing += 1
            continue
        encoded.append(one_hot_encode_circular(seq, window_size=window_size, wrap_len=wrap_len))
        labels.append(row["label"])
    if missing:
        print(f"  ATTENZIONE: {missing} id senza sequenza valida, scartati.")
    X = torch.from_numpy(np.stack(encoded).astype(np.float32))
    y = torch.tensor(labels, dtype=torch.float32)
    return X, y


@torch.no_grad()
def evaluate(model, X, y, criterion, batch_size=256):
    model.eval()
    logits_all = []
    losses = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item() * len(xb))
        logits_all.append(logits)
    logits_all = torch.cat(logits_all)
    probs = torch.sigmoid(logits_all).numpy()
    preds = (probs >= 0.5).astype(int)
    y_np = y.numpy().astype(int)

    metrics = {
        "loss": float(sum(losses) / len(X)),
        "accuracy": float(accuracy_score(y_np, preds)),
        "precision": float(precision_score(y_np, preds, zero_division=0)),
        "recall": float(recall_score(y_np, preds, zero_division=0)),
        "f1": float(f1_score(y_np, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_np, probs)),
    }
    return metrics, preds, probs, y_np


def main():
    args = parse_args()
    set_seed(args.seed)
    use_all_cpu_threads()
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- FASE 1: CARICAMENTO ID BILANCIATI ---")
    splits = load_split_ids(args.data_dir)
    for name, df in splits.items():
        print(f"{name}: {len(df)} righe")

    all_ids = set()
    for df in splits.values():
        all_ids.update(df["id"].astype(str))

    print("\n--- FASE 2: CARICAMENTO SEQUENZE NECESSARIE DAL FASTA ---")
    sequences = load_needed_sequences(args.fasta_path, all_ids)

    print("\n--- FASE 3: CODIFICA ONE-HOT CIRCOLARE ---")
    print(f"window_size={args.window_size}, wrap_len={args.wrap_len}")
    X_train, y_train = encode_split(splits["train"], sequences, args.window_size, args.wrap_len)
    X_val, y_val = encode_split(splits["val"], sequences, args.window_size, args.wrap_len)
    X_test, y_test = encode_split(splits["test"], sequences, args.window_size, args.wrap_len)
    print(f"train: {tuple(X_train.shape)}  val: {tuple(X_val.shape)}  test: {tuple(X_test.shape)}")

    print("\n--- FASE 4: TRAINING ---")
    model = EccDNACNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modello EccDNACNN: {n_params} parametri")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True
    )

    best_val_auc = -1.0
    epochs_without_improvement = 0
    best_path = os.path.join(args.output_dir, "best_model.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        train_loss = running_loss / len(X_train)

        val_metrics, _, _, _ = evaluate(model, X_val, y_val, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

        improved = val_metrics["roc_auc"] > best_val_auc
        marker = " <- nuovo migliore" if improved else ""
        print(
            f"Epoca {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} | {time.time() - t0:.1f}s{marker}"
        )

        if improved:
            best_val_auc = val_metrics["roc_auc"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: nessun miglioramento dell'AUC di validazione da {args.patience} epoche.")
                break

    print("\n--- FASE 5: VALUTAZIONE SUL TEST SET (checkpoint migliore) ---")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_metrics, preds, probs, y_np = evaluate(model, X_test, y_test, criterion)
    cm = confusion_matrix(y_np, preds).tolist()

    print("Metriche sul test set:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Matrice di confusione [[TN, FP], [FN, TP]]:", cm)

    results = {
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "window_size": args.window_size,
        "wrap_len": args.wrap_len,
        "n_params": n_params,
        "best_val_auc": best_val_auc,
        "epochs_run": len(history),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist_df = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
        axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
        axes[0].set_xlabel("epoca")
        axes[0].legend()
        axes[0].set_title("Loss")

        axes[1].plot(hist_df["epoch"], hist_df["val_roc_auc"], label="val_auc")
        axes[1].plot(hist_df["epoch"], hist_df["val_accuracy"], label="val_accuracy")
        axes[1].set_xlabel("epoca")
        axes[1].legend()
        axes[1].set_title("Validation metrics")

        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=120)
        print("Grafico salvato in", os.path.join(args.output_dir, "training_curves.png"))
    except Exception as e:
        print("Impossibile generare il grafico:", e)

    print("\n--- FASE 6: COMPLETAMENTO ---")
    print("Checkpoint migliore:", best_path)
    print("Metriche:", os.path.join(args.output_dir, "metrics.json"))


if __name__ == "__main__":
    main()
