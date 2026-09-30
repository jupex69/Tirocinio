"""Pre-addestramento SELF-SUPERVISED (contrastivo, stile SimCLR/foundation-model)
dell'encoder eccDNA, poi few-shot/one-shot per l'identificazione della malattia.

La via indicata dalla letteratura: imparare una buona rappresentazione dell'eccDNA
SENZA etichette di malattia (su tanti dati non etichettati), poi specializzarla con
pochi esempi. Qui:

  1) PRE-TRAINING contrastivo (NT-Xent): due "viste" della stessa molecola circolare
     -- due archi casuali (sotto-sequenze con wrap) -- devono avere embedding vicini;
     molecole diverse lontane. Nessuna etichetta. Feature: 74 (10 descrittori + 64 3-mer).
  2) DOWNSTREAM: few-shot/one-shot (3 loss: euclideo, coseno, triplet) sulle 9 malattie,
     encoder inizializzato DA ZERO vs PRE-ADDESTRATO, poi fine-tuning episodico.
     Within-study e cross-studio, length-matched. Caso = 1/9.

Confronto chiave: il pre-training aiuta il few-shot? E soprattutto il CROSS-STUDIO?
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_multiclass import Encoder, _class_prototypes, _proto_logits, standardize, SEED, DEVICE
from models_pytorch import _to_tensor, _pk_sample, _batch_hard_triplet_loss
from eccdna_utils import compute_sequence_descriptors, read_fasta_stream
from train_siamese_multiclass import kmer_spectrum, FEATURE_COLS
from full_classifier import FASTA, RESULTS_DIR
from experiment_disease_multiclass import load as load_disease, length_match
from experiment_oneshot import _episode

PRE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/pretrain_views.npz")
N_PRE = 12000        # molecole non etichettate per il pre-training
V = 4                # viste (archi) per molecola
ARC = 0.7            # frazione di lunghezza di ogni arco
NF = len(FEATURE_COLS)
torch.manual_seed(SEED)


# ---------------- viste contrastive ----------------
def subarc(seq, frac, rng):
    L = len(seq); w = max(6, int(frac * L)); s = int(rng.integers(0, L))
    return seq[s:s + w] if s + w <= L else seq[s:] + seq[:s + w - L]   # circolare


def feat74(seq):
    f = kmer_spectrum(seq); f.update(compute_sequence_descriptors(seq))
    return [f[c] for c in FEATURE_COLS]


def build_pretrain_views():
    if os.path.exists(PRE_CACHE):
        d = np.load(PRE_CACHE); print(f"Viste di pre-training da cache: {d['views'].shape}")
        return d["views"]
    rng = np.random.default_rng(SEED)
    COMPACT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/processed/eccdna_compacted_metadata.tsv")
    ids = []
    for ch in pd.read_csv(COMPACT, sep="\t", usecols=["id"], chunksize=500000, low_memory=False):
        s = ch["id"].astype(str).to_numpy()
        take = s[rng.random(len(s)) < 0.006]
        ids.extend(take.tolist())
        if len(ids) >= N_PRE * 1.3:
            break
    ids = set(rng.choice(ids, min(N_PRE, len(ids)), replace=False))
    print(f"Molecole campionate per il pre-training: {len(ids)}")
    views = []
    seen = 0
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=ids):
        if "N" in seq or len(seq) < 12:
            continue
        vs = [feat74(subarc(seq, ARC, rng)) for _ in range(V)]
        views.append(vs); seen += 1
        if seen % 2000 == 0:
            print(f"  featurizzate {seen} molecole...")
    views = np.asarray(views, dtype=np.float32)     # (N, V, 74)
    np.savez_compressed(PRE_CACHE, views=views)
    print(f"Viste: {views.shape}  salvate.")
    return views


# ---------------- pre-training contrastivo ----------------
class ProjHead(nn.Module):
    def __init__(self, d=32):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
    def forward(self, x): return self.net(x)


def nt_xent(z1, z2, temp=0.5):
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)          # (2B, d)
    sim = z @ z.t() / temp
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, targets)


def pretrain(views, mean, std, epochs=80, batch=256, lr=1e-3):
    N = len(views); Z = (views - mean) / std
    enc = Encoder(NF).to(DEVICE); proj = ProjHead().to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(SEED); enc.train()
    for ep in range(epochs):
        perm = rng.permutation(N); tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            if len(idx) < 8:
                continue
            v1 = rng.integers(0, V, len(idx)); v2 = (v1 + 1 + rng.integers(0, V - 1, len(idx))) % V
            x1 = _to_tensor(Z[idx, v1], DEVICE); x2 = _to_tensor(Z[idx, v2], DEVICE)
            loss = nt_xent(proj(enc(x1)), proj(enc(x2)))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  pretrain ep {ep:3d}  loss={tot/max(1,N//batch):.3f}")
    return enc


# ---------------- fine-tuning episodico (3 loss) ----------------
def train_episodic(enc, X, y, nC, loss_kind, k=1, q=5, epochs=60, episodes=30, lr=1e-3, seed=SEED):
    rng = np.random.default_rng(seed); opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=1e-4)
    Xt = _to_tensor(X, DEVICE); yt = np.asarray(y)
    cos = loss_kind == "coseno"
    for ep in range(epochs):
        enc.train()
        for _ in range(episodes):
            if loss_kind in ("euclideo", "coseno"):
                classes = np.array([c for c in np.unique(yt) if (yt == c).sum() >= k + q])
                if len(classes) < 2: continue
                si, sl, qi, ql, _ = _episode(rng, yt, classes, len(classes), k, q)
                emb = enc(Xt)
                C = _class_prototypes(emb[si], torch.as_tensor(sl, device=DEVICE), len(classes), cos)
                logit = _proto_logits(emb[qi], C, cos)
                lossv = F.cross_entropy(logit, torch.as_tensor(ql, device=DEVICE))
            else:  # triplet batch-hard
                idx = _pk_sample(yt, rng, nC, max(2, 128 // nC))
                lossv = _batch_hard_triplet_loss(enc(Xt[idx]), yt[idx], margin=0.3)
            opt.zero_grad(); lossv.backward(); opt.step()
    return enc


@torch.no_grad()
def eval_shot(enc, Xs, ys, Xq, yq, k, cos, n_ep=500, seed=SEED):
    rng = np.random.default_rng(seed); enc.eval()
    Es = enc(_to_tensor(Xs, DEVICE)); Eq = enc(_to_tensor(Xq, DEVICE))
    classes = np.array([c for c in np.unique(ys) if (ys == c).sum() >= 1 and (yq == c).sum() >= 1])
    accs = []
    for _ in range(n_ep):
        si, sl, qi, ql, _ = _episode(rng, ys, classes, len(classes), k, 10, y2=yq)
        C = _class_prototypes(Es[si], torch.as_tensor(sl, device=DEVICE), len(classes), cos)
        pred = _proto_logits(Eq[qi], C, cos).argmax(1).cpu().numpy()
        accs.append((pred == ql).mean())
    return float(np.mean(accs))


def clone_enc(src):
    e = Encoder(NF).to(DEVICE)
    if src is not None:
        e.load_state_dict(src.state_dict())
    return e


def main():
    # feature per il pre-training + statistiche
    views = build_pretrain_views()
    flat = views.reshape(-1, NF)
    mean, std = flat.mean(0), flat.std(0); std[std == 0] = 1.0

    print("\n=== PRE-TRAINING CONTRASTIVO (self-supervised) ===")
    pre_enc = pretrain(views, mean, std)

    # dati 9 malattie (74 feature), length-matched
    df = length_match(load_disease())
    classes = sorted(df["label"].unique()); c2i = {c: i for i, c in enumerate(classes)}; nC = len(classes)
    chance = 1 / nC
    X = df[FEATURE_COLS].to_numpy(np.float32); y = df["label"].map(c2i).to_numpy(); db = df["source_db"].to_numpy()
    a = db == "eccDNABase"; b = db == "CircleBaseV2"
    print(f"\n9 malattie, {len(df)} seq (length-matched), 74 feature, caso={chance:.3f}\n")

    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    XA, XB = standardize(X[a], X[b]); XB2, XA2 = standardize(X[b], X[a])

    def pre_init():                         # encoder pre-addestrato, standardizzato come i suoi dati
        return clone_enc(pre_enc)

    rows = []
    for loss in ["euclideo", "coseno", "triplet"]:
        cos = loss == "coseno"
        for init_name, init_fn in [("da zero", lambda: clone_enc(None)), ("pre-addestrato", pre_init)]:
            enc_w = train_episodic(init_fn(), Xtr_s, ytr, nC, loss)
            encA = train_episodic(init_fn(), XA, y[a], nC, loss)
            encB = train_episodic(init_fn(), XB2, y[b], nC, loss)
            for k, reg in [(1, "one-shot"), (5, "few-shot")]:
                w = eval_shot(enc_w, Xtr_s, ytr, Xte_s, yte, k, cos)
                c = (eval_shot(encA, XA, y[a], XB, y[b], k, cos) + eval_shot(encB, XB2, y[b], XA2, y[a], k, cos)) / 2
                rows.append({"loss": loss, "init": init_name, "regime": reg,
                             "within": round(w, 3), "within_x": round(w / chance, 2),
                             "cross": round(c, 3), "cross_x": round(c / chance, 2)})
                print(f"  {loss:9s} {init_name:14s} {reg:9s} within={w:.3f} ({w/chance:.1f}x)  cross={c:.3f} ({c/chance:.1f}x)")

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "pretrain_fewshot.tsv"), sep="\t", index=False)
    print("\n" + res.to_string(index=False))
    print(f"\ncaso={chance:.3f}. Salvato in {RESULTS_DIR}/pretrain_fewshot.tsv")


if __name__ == "__main__":
    main()
