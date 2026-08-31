"""Classificazione MULTICLASSE della malattia (17 classi) con reti siamesi e
schema prototypical network (softmax sulle distanze ai prototipi di classe).

Si confrontano quattro configurazioni, tutte con output softmax multiclasse:
- Prototipico EUCLIDEO : softmax su -||x - c_k||^2 (distanza euclidea ai centroidi)
- Prototipico COSENO   : softmax sulla similarita' coseno (x . c_k) * scala
- TRIPLET + prototipo  : embedding modellato con triplet loss batch-hard,
                         classificazione finale per prototipo (euclideo)
- Baseline SOFTMAX      : MLP con testa lineare a 17 uscite + cross-entropy
                         (senza metric learning, riferimento)

I prototipi in addestramento si calcolano per episodio (support/query, stile
Snell et al. 2017); in valutazione dai centroidi di TUTTO il training.

Metriche multiclasse: accuracy, accuracy bilanciata, macro-F1, top-3 accuracy.
Il caso casuale su 17 classi e' ~0.059 di accuracy.

LIMITE DICHIARATO: il metodo di sequenziamento e' quasi un proxy della malattia;
parte del segnale puo' derivare dal protocollo, non dalla biologia (vedi
multiclass_data.py). Primo esperimento esplorativo.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score,
)

from eccdna_utils import DESCRIPTOR_NAMES
from multiclass_data import build_multiclass_splits, summarize
from models_pytorch import _to_tensor, _pk_sample, _batch_hard_triplet_loss

SEED = 42
DEVICE = "cpu"
COSINE_SCALE = 10.0


# ------------------------------- modelli -------------------------------
class Encoder(nn.Module):
    """MLP che mappa i descrittori in un embedding grezzo (non normalizzato)."""
    def __init__(self, n_features, hidden=(64, 32), embedding_dim=32, dropout=0.3):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SoftmaxMLP(nn.Module):
    """Baseline: encoder + testa lineare a n_classes uscite (cross-entropy)."""
    def __init__(self, n_features, n_classes, hidden=(64, 32), embedding_dim=32, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(n_features, hidden, embedding_dim, dropout)
        self.head = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


# --------------------------- prototipi & logits ---------------------------
def _class_prototypes(emb, labels, n_classes, cosine):
    protos = []
    for k in range(n_classes):
        mask = labels == k
        protos.append(emb[mask].mean(0) if mask.any() else torch.zeros(emb.shape[1], device=emb.device))
    C = torch.stack(protos)
    return F.normalize(C, dim=-1) if cosine else C


def _proto_logits(q, C, cosine, scale=COSINE_SCALE):
    if cosine:
        return F.normalize(q, dim=-1) @ C.t() * scale
    return -torch.cdist(q, C) ** 2


def _sample_episode(y, n_classes, n_support, n_query, rng):
    s_idx, s_lab, q_idx, q_lab = [], [], [], []
    for k in range(n_classes):
        pool = np.where(y == k)[0]
        pick = rng.choice(pool, n_support + n_query, replace=len(pool) < n_support + n_query)
        s_idx += list(pick[:n_support]); s_lab += [k] * n_support
        q_idx += list(pick[n_support:]); q_lab += [k] * n_query
    return (np.array(s_idx), np.array(s_lab), np.array(q_idx), np.array(q_lab))


# ------------------------------- training -------------------------------
def train_prototypical(Xtr, ytr, Xva, yva, n_classes, cosine, epochs=200,
                       episodes=40, n_support=10, n_query=10, lr=1e-3, patience=20, seed=SEED,
                       hidden=(64, 32), embedding_dim=32, dropout=0.3, cosine_scale=COSINE_SCALE,
                       select_metric="accuracy"):
    """select_metric: 'accuracy' o 'balanced' — metrica per early-stopping/scelta
    del miglior stato (usa 'balanced' per un confronto equo col softmax bilanciato)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = Encoder(Xtr.shape[1], hidden, embedding_dim, dropout).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    Xt = _to_tensor(Xtr, DEVICE)
    score = balanced_accuracy_score if select_metric == "balanced" else accuracy_score
    best_acc, best_state, no_improve = -1.0, None, 0

    for _ in range(epochs):
        enc.train()
        for _ in range(episodes):
            s_idx, s_lab, q_idx, q_lab = _sample_episode(ytr, n_classes, n_support, n_query, rng)
            opt.zero_grad()
            emb_s = enc(Xt[s_idx])
            emb_q = enc(Xt[q_idx])
            C = _class_prototypes(emb_s, torch.as_tensor(s_lab, device=DEVICE), n_classes, cosine)
            logits = _proto_logits(emb_q, C, cosine, cosine_scale)
            loss = F.cross_entropy(logits, torch.as_tensor(q_lab, device=DEVICE))
            loss.backward()
            opt.step()

        proba = predict_prototypical(enc, Xtr, ytr, Xva, n_classes, cosine)
        acc = score(yva, proba.argmax(1))
        if acc > best_acc:
            best_acc, best_state, no_improve = acc, {k: v.clone() for k, v in enc.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state:
        enc.load_state_dict(best_state)
    return enc, best_acc


@torch.no_grad()
def predict_prototypical(enc, Xtr, ytr, X, n_classes, cosine):
    enc.eval()
    emb_tr = enc(_to_tensor(Xtr, DEVICE))
    C = _class_prototypes(emb_tr, torch.as_tensor(ytr, device=DEVICE), n_classes, cosine)
    logits = _proto_logits(enc(_to_tensor(X, DEVICE)), C, cosine)
    return F.softmax(logits, dim=1).cpu().numpy()


def train_triplet(Xtr, ytr, Xva, yva, n_classes, epochs=200, batches=40,
                  n_per_class=16, lr=1e-3, margin=0.3, patience=20, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = Encoder(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    Xt = _to_tensor(Xtr, DEVICE)
    best_acc, best_state, no_improve = -1.0, None, 0

    for _ in range(epochs):
        enc.train()
        for _ in range(batches):
            idx = _pk_sample(ytr, rng, n_classes, n_per_class)
            opt.zero_grad()
            emb = enc(Xt[idx])
            loss = _batch_hard_triplet_loss(emb, ytr[idx], margin)
            loss.backward()
            opt.step()
        proba = predict_prototypical(enc, Xtr, ytr, Xva, n_classes, cosine=False)
        acc = accuracy_score(yva, proba.argmax(1))
        if acc > best_acc:
            best_acc, best_state, no_improve = acc, {k: v.clone() for k, v in enc.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state:
        enc.load_state_dict(best_state)
    return enc, best_acc


def train_softmax(Xtr, ytr, Xva, yva, n_classes, epochs=200, batch_size=256,
                  lr=1e-3, patience=20, seed=SEED, balanced=False):
    """balanced=True: cross-entropy pesata con pesi inversamente proporzionali
    alla frequenza di classe (controllo: isola l'effetto del bilanciamento da
    quello del metric learning). Selezione su balanced accuracy quando balanced."""
    torch.manual_seed(seed)
    model = SoftmaxMLP(Xtr.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = _to_tensor(Xtr, DEVICE), torch.as_tensor(ytr, device=DEVICE)
    Xv = _to_tensor(Xva, DEVICE)
    n = Xt.shape[0]
    weight = None
    if balanced:
        counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
        w = n / (n_classes * np.maximum(counts, 1))
        weight = torch.as_tensor(w, dtype=torch.float32, device=DEVICE)
    best_acc, best_state, no_improve = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[idx]), yt[idx], weight=weight)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_va = model(Xv).argmax(1).cpu().numpy()
            acc = balanced_accuracy_score(yva, pred_va) if balanced else accuracy_score(yva, pred_va)
        if acc > best_acc:
            best_acc, best_state, no_improve = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_softmax(model, X):
    model.eval()
    return F.softmax(model(_to_tensor(X, DEVICE)), dim=1).cpu().numpy()


# ------------------------------- valutazione -------------------------------
def evaluate(name, y_true, proba, n_classes):
    y_pred = proba.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    top3 = top_k_accuracy_score(y_true, proba, k=3, labels=list(range(n_classes)))
    print(f"{name:26s} acc={acc:.3f}  bal_acc={bacc:.3f}  macro-F1={mf1:.3f}  top3={top3:.3f}")
    return {"model": name, "accuracy": acc, "balanced_accuracy": bacc, "macro_f1": mf1, "top3": top3}


def standardize(train_X, *others):
    m, s = train_X.mean(0), train_X.std(0); s[s == 0] = 1.0
    return [(train_X - m) / s] + [(X - m) / s for X in others]


def main():
    print("--- Caricamento dataset multiclasse (17 malattie) ---")
    tr, va, te = build_multiclass_splits()
    summarize(tr, va, te)

    classes = sorted(tr["disease"].unique())
    cls2idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    Xtr = tr[DESCRIPTOR_NAMES].to_numpy(np.float32)
    Xva = va[DESCRIPTOR_NAMES].to_numpy(np.float32)
    Xte = te[DESCRIPTOR_NAMES].to_numpy(np.float32)
    ytr = tr["disease"].map(cls2idx).to_numpy()
    yva = va["disease"].map(cls2idx).to_numpy()
    yte = te["disease"].map(cls2idx).to_numpy()
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)

    print(f"\nCaso casuale (17 classi): accuracy ~= {1/n_classes:.3f}\n")
    print("--- CONFRONTO (test set) ---")
    results = []

    enc, vacc = train_prototypical(Xtr, ytr, Xva, yva, n_classes, cosine=False, seed=SEED)
    results.append(evaluate("Prototipico euclideo", yte, predict_prototypical(enc, Xtr, ytr, Xte, n_classes, False), n_classes))

    enc, vacc = train_prototypical(Xtr, ytr, Xva, yva, n_classes, cosine=True, seed=SEED)
    results.append(evaluate("Prototipico coseno", yte, predict_prototypical(enc, Xtr, ytr, Xte, n_classes, True), n_classes))

    enc, vacc = train_triplet(Xtr, ytr, Xva, yva, n_classes, seed=SEED)
    results.append(evaluate("Triplet + prototipo", yte, predict_prototypical(enc, Xtr, ytr, Xte, n_classes, False), n_classes))

    model = train_softmax(Xtr, ytr, Xva, yva, n_classes, seed=SEED)
    results.append(evaluate("Softmax baseline", yte, predict_softmax(model, Xte), n_classes))

    print("\n--- RIEPILOGO (ordinato per macro-F1) ---")
    df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    print(df.round(3).to_string(index=False))
    df.to_csv("data/processed/multiclass_results.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
