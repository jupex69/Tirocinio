"""Adattamento di dominio ADVERSARIALE (DANN, Ganin & Lempitsky 2015) per il
transfer cross-source: il metodo di riferimento per dati MULTI-COORTE futuri.

Idea: un encoder condiviso alimenta (a) una testa che predice il TESSUTO e (b)
una testa avversaria che prova a indovinare il DATABASE, collegata da un
'gradient reversal': l'encoder viene spinto a rendere l'embedding UTILE per il
tessuto ma INDISTINGUIBILE per database -> feature di dominio-invarianti ->
transfer migliore verso un laboratorio mai visto.

Perche' qui e' LEGITTIMO (mentre sul task a 17 classi no): sui 6 tessuti
cross-source il tessuto NON e' perfettamente confuso col database (ogni tessuto
esiste in entrambi), quindi si puo' togliere l'informazione di database SENZA
cancellare il tessuto. E' esattamente il caso d'uso per cui il DANN esiste.

Protocollo DANN: sorgente = un DB (con etichette tessuto) + bersaglio = altro DB
(SENZA etichette, usato solo per l'avversario di dominio). Si valuta il tessuto
sul bersaglio. Confronto con baseline non-adversariale (stesso encoder, senza
avversario). Entrambe le direzioni.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from experiment_cross_source import load_cross_base, get_features, RESULTS_DIR
from train_siamese_multiclass import FEATURE_COLS

SEED = 42
DEVICE = "cpu"


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grad_reverse(x, lamb):
    return GradReverse.apply(x, lamb)


class DANN(nn.Module):
    def __init__(self, n_feat, n_tissue, hidden=64, emb=32, dropout=0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_feat, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, emb), nn.ReLU())
        self.tissue = nn.Linear(emb, n_tissue)
        self.domain = nn.Sequential(nn.Linear(emb, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x, lamb=0.0):
        z = self.enc(x)
        return self.tissue(z), self.domain(grad_reverse(z, lamb))


def _std(train, *rest):
    m, s = train.mean(0), train.std(0); s[s == 0] = 1.0
    return [(train - m) / s] + [(r - m) / s for r in rest]


def train_dann(Xs, ys, Xt, n_tissue, adversarial, epochs=150, lr=1e-3, seed=SEED):
    """Xs/ys: sorgente con etichette tessuto. Xt: bersaglio SENZA etichette (solo
    dominio). adversarial=False -> baseline (lambda=0, niente avversario)."""
    torch.manual_seed(seed)
    m = DANN(Xs.shape[1], n_tissue).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    Xs_t = torch.as_tensor(Xs, dtype=torch.float32); ys_t = torch.as_tensor(ys)
    Xt_t = torch.as_tensor(Xt, dtype=torch.float32)
    ns, nt = len(Xs_t), len(Xt_t)
    for ep in range(epochs):
        m.train()
        lamb = (2.0 / (1.0 + np.exp(-10 * ep / epochs)) - 1.0) if adversarial else 0.0
        # batch pieno (dati piccoli)
        ts, ds = m(Xs_t, lamb)
        loss = F.cross_entropy(ts, ys_t)
        if adversarial:
            _, dt = m(Xt_t, lamb)
            dom_logits = torch.cat([ds, dt]); dom_y = torch.cat([torch.zeros(ns), torch.ones(nt)]).long()
            loss = loss + F.cross_entropy(dom_logits, dom_y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def predict(m, X):
    m.eval()
    t, _ = m(torch.as_tensor(X, dtype=torch.float32), 0.0)
    return F.softmax(t, 1).cpu().numpy()


def run(mc, tissues, A, B, adversarial):
    c2i = {c: i for i, c in enumerate(tissues)}
    a = mc[mc.source_db == A]; b = mc[mc.source_db == B]
    Xs = a[FEATURE_COLS].to_numpy(np.float32); ys = a["tessuto"].map(c2i).to_numpy()
    Xt = b[FEATURE_COLS].to_numpy(np.float32); yt = b["tessuto"].map(c2i).to_numpy()
    Xs, Xt = _std(Xs, Xt)
    m = train_dann(Xs, ys, Xt, len(tissues), adversarial)
    proba = predict(m, Xt); pred = proba.argmax(1)
    acc = accuracy_score(yt, pred); bacc = balanced_accuracy_score(yt, pred)
    # AUC one-vs-rest per tessuto
    aucs = {}
    for c, i in c2i.items():
        yb = (yt == i).astype(int)
        aucs[c] = roc_auc_score(yb, proba[:, i]) if 0 < yb.sum() < len(yb) else np.nan
    return acc, bacc, aucs


def main():
    mc = get_features(load_cross_base())
    tissues = sorted(mc["tessuto"].unique())
    print(f"Tessuti: {tissues}  caso={1/len(tissues):.3f}\n")
    rows, auc_rows = [], []
    for A, B in [("eccDNABase", "CircleBaseV2"), ("CircleBaseV2", "eccDNABase")]:
        for adv in (False, True):
            acc, bacc, aucs = run(mc, tissues, A, B, adv)
            tag = "DANN" if adv else "baseline"
            print(f"{A}->{B:14s} [{tag:8s}]  acc={acc:.3f}  bal_acc={bacc:.3f}  "
                  f"AUC medio={np.nanmean(list(aucs.values())):.3f}")
            rows.append({"direzione": f"{A}->{B}", "metodo": tag, "acc": round(acc, 3),
                         "bal_acc": round(bacc, 3), "AUC_medio": round(float(np.nanmean(list(aucs.values()))), 3)})
            for c, v in aucs.items():
                auc_rows.append({"direzione": f"{A}->{B}", "metodo": tag, "tessuto": c, "AUC": round(float(v), 3)})
    res = pd.DataFrame(rows)
    print("\n=== DANN vs baseline (transfer cross-source) ===")
    print(res.to_string(index=False))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "adversarial_transfer.tsv"), sep="\t", index=False)
    pd.DataFrame(auc_rows).to_csv(os.path.join(RESULTS_DIR, "adversarial_per_tissue.tsv"), sep="\t", index=False)

    # robustezza per tessuto: min AUC tra le due direzioni, baseline vs DANN
    at = pd.DataFrame(auc_rows)
    print("\n=== AUC robusta per tessuto (min tra direzioni): baseline vs DANN ===")
    for metodo in ("baseline", "DANN"):
        sub = at[at.metodo == metodo]
        rob = sub.groupby("tessuto")["AUC"].min().sort_values(ascending=False)
        n60 = int((rob >= 0.60).sum())
        print(f"  {metodo:8s}: rilevabili(>=0.60)={n60}  " + "  ".join(f"{t}:{v:.2f}" for t, v in rob.items()))
    print(f"\nSalvato in {RESULTS_DIR}/adversarial_transfer.tsv e adversarial_per_tissue.tsv")


if __name__ == "__main__":
    main()
