"""Confronto di FUNZIONI DI PERDITA per la rete siamese sul multiclasse e
valutazione FEW-SHOT / ONE-SHOT, con particolare attenzione alle CLASSI MINORI.

Obiettivo (richiesta esplicita): migliorare l'individuazione delle malattie rare
ribilanciando meglio e provando loss diverse dal solo coseno gia' testato.

RIBILANCIAMENTO (peso alle classi minori) — applicato a TUTTE le varianti:
 - training episodico / P-K: ogni classe entra con lo STESSO numero di campioni
   per episodio; le rare (pool piccolo) vengono ricampionate a parita' -> di
   fatto sovracampionate rispetto alla loro frequenza naturale;
 - selezione del modello (early stopping) su BALANCED ACCURACY, non accuracy:
   non premia il collasso sulle classi comuni;
 - softmax di riferimento con pesi di classe inversi alla frequenza.
Nota metodologica: dare peso PIU' CHE paritario alle rare tende a peggiorare il
totale; qui si tiene la parita' e si RIPORTA l'aggregato rare vs comuni, cosi'
il trade-off e' misurabile invece che assunto.

LOSS A CONFRONTO (encoder identico, cambia solo come si struttura l'embedding):
 - Prototipico COSENO   : softmax su similarita' coseno ai prototipi (baseline gia' vista)
 - Prototipico EUCLIDEO : softmax su -||x-c||^2
 - TRIPLET (batch-hard) : Hermans et al. 2017, prototipo euclideo in inferenza
 - CONTRASTIVE (a coppie): Hadsell et al. 2006, prototipo euclideo in inferenza
 - Softmax bilanciato   : MLP + cross-entropy pesata (riferimento non metrico)

FEW-SHOT / ONE-SHOT:
 - episodico N-way K-shot (K=1 one-shot, K=5) su test: quanto l'embedding
   riconosce una classe da 1 (o 5) soli esempi di supporto;
 - one-shot "full-way" per-classe: K esempi di supporto per ogni classe dal
   train, si classifica TUTTO il test -> recall per ciascuna classe (rara inclusa).

LIMITE DICHIARATO (fondamentale): sul multiclasse a 17 malattie il metodo di
sequenziamento e' quasi un proxy della malattia (vedi multiclass_data.py). Un
guadagno qui puo' riflettere il protocollo, non la biologia. Questo e' un
CONFRONTO METODOLOGICO tra loss, non un nuovo risultato biologico; il segnale
biologicamente pulito resta quello confounder-controlled (gold_standard_data.py).

I risultati di TUTTE le varianti sono salvati in results/ (tracciata da git).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from train_multiclass import (
    Encoder, standardize, train_prototypical, predict_prototypical,
    _class_prototypes, _proto_logits, train_softmax, predict_softmax,
    evaluate, SEED, DEVICE,
)
from train_siamese_multiclass import build_rich_splits, FEATURE_COLS, per_class_report, RARE_MAX_TRAIN
from models_pytorch import _to_tensor, _pk_sample, _batch_hard_triplet_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# =============================== LOSS ===============================
def _contrastive_loss(emb, labels, margin):
    """Contrastive loss a coppie (Hadsell et al. 2006): le coppie della STESSA
    classe vengono avvicinate (d^2), quelle di classe diversa allontanate oltre
    un margine (max(0, margine - d)^2). Si contano le coppie una sola volta
    (triangolo superiore). Con P-K sampling ogni classe pesa uguale."""
    dist = torch.cdist(emb, emb, p=2)
    labels = torch.as_tensor(labels, device=emb.device)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    triu = torch.triu(torch.ones_like(same), diagonal=1).bool()
    pos = same & triu
    neg = (~same) & triu
    zero = emb.sum() * 0.0
    loss_pos = (dist[pos] ** 2).mean() if pos.any() else zero
    loss_neg = (torch.relu(margin - dist[neg]) ** 2).mean() if neg.any() else zero
    return loss_pos + loss_neg


# ============================= TRAINING =============================
def train_metric_encoder(Xtr, ytr, Xva, yva, n_classes, loss_kind, margin,
                         epochs=200, batches=40, n_per_class=16, lr=1e-3,
                         patience=20, seed=SEED, select_metric="balanced"):
    """Addestra l'encoder con triplet (batch-hard) o contrastive su batch P-K
    (classi bilanciate). Inferenza/selezione per prototipo euclideo; early
    stopping su balanced accuracy (peso alle classi minori)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = Encoder(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    Xt = _to_tensor(Xtr, DEVICE)
    score = balanced_accuracy_score if select_metric == "balanced" else accuracy_score
    lossfn = _batch_hard_triplet_loss if loss_kind == "triplet" else _contrastive_loss
    best, best_state, no_imp = -1.0, None, 0

    for _ in range(epochs):
        enc.train()
        for _ in range(batches):
            idx = _pk_sample(ytr, rng, n_classes, n_per_class)
            opt.zero_grad()
            emb = enc(Xt[idx])
            loss = lossfn(emb, ytr[idx], margin)
            loss.backward()
            opt.step()
        proba = predict_prototypical(enc, Xtr, ytr, Xva, n_classes, cosine=False)
        acc = score(yva, proba.argmax(1))
        if acc > best:
            best, best_state, no_imp = acc, {k: v.clone() for k, v in enc.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state:
        enc.load_state_dict(best_state)
    return enc, best


# ========================= FEW-SHOT / ONE-SHOT =========================
def _few_shot_episode(y, classes_pool, n_way, k_shot, n_query, rng):
    chosen = rng.choice(classes_pool, size=n_way, replace=False)
    s_idx, s_lab, q_idx, q_lab = [], [], [], []
    for j, c in enumerate(chosen):
        pool = np.where(y == c)[0]
        need = k_shot + n_query
        pick = rng.choice(pool, need, replace=len(pool) < need)
        s_idx += list(pick[:k_shot]); s_lab += [j] * k_shot
        q_idx += list(pick[k_shot:]); q_lab += [j] * n_query
    return np.array(s_idx), np.array(s_lab), np.array(q_idx), np.array(q_lab)


@torch.no_grad()
def few_shot_eval(enc, X, y, cosine, n_way, k_shot, n_query=10, n_episodes=300, seed=SEED):
    """Accuratezza media su episodi N-way K-shot costruiti dal test: si stimano i
    prototipi dai K esempi di supporto e si classificano i query. K=1 -> one-shot.
    Misura quanto l'embedding riconosce una classe da pochissimi esempi."""
    enc.eval()
    rng = np.random.default_rng(seed)
    classes_pool = np.unique(y)
    n_way = min(n_way, len(classes_pool))
    emb_all = enc(_to_tensor(X, DEVICE))
    accs = []
    for _ in range(n_episodes):
        s_idx, s_lab, q_idx, q_lab = _few_shot_episode(y, classes_pool, n_way, k_shot, n_query, rng)
        C = _class_prototypes(emb_all[s_idx], torch.as_tensor(s_lab, device=DEVICE), n_way, cosine)
        pred = _proto_logits(emb_all[q_idx], C, cosine).argmax(1).cpu().numpy()
        accs.append((pred == q_lab).mean())
    return float(np.mean(accs)), float(np.std(accs)), n_way


@torch.no_grad()
def k_shot_full(enc, Xtr, ytr, Xte, yte, n_classes, cosine, k_shot, n_draws=50, seed=SEED):
    """One-shot/K-shot 'full-way': K esempi di supporto per OGNI classe (dal
    train), si classifica TUTTO il test per prototipo piu' vicino, ripetendo
    n_draws volte. Ritorna (accuratezze per estrazione, recall media per classe).
    Mostra quanto ogni classe RARA e' riconoscibile da pochi esempi."""
    enc.eval()
    rng = np.random.default_rng(seed)
    emb_tr = enc(_to_tensor(Xtr, DEVICE))
    emb_te = enc(_to_tensor(Xte, DEVICE))
    accs, recalls = [], np.full((n_draws, n_classes), np.nan)
    for d in range(n_draws):
        s_idx, s_lab = [], []
        for c in range(n_classes):
            pool = np.where(ytr == c)[0]
            pick = rng.choice(pool, k_shot, replace=len(pool) < k_shot)
            s_idx += list(pick); s_lab += [c] * k_shot
        C = _class_prototypes(emb_tr[s_idx], torch.as_tensor(s_lab, device=DEVICE), n_classes, cosine)
        pred = _proto_logits(emb_te, C, cosine).argmax(1).cpu().numpy()
        accs.append((pred == yte).mean())
        for c in range(n_classes):
            m = yte == c
            if m.any():
                recalls[d, c] = (pred[m] == c).mean()
    return accs, np.nanmean(recalls, axis=0)


# ============================== METRICHE ==============================
def full_metrics(name, yte, proba, n_classes, classes, support_train):
    """Metriche globali + aggregati rare vs comuni (soglia RARE_MAX_TRAIN)."""
    row = evaluate(name, yte, proba, n_classes)
    rep = per_class_report(yte, proba, classes, support_train)
    rare = rep[rep["train_n"] <= RARE_MAX_TRAIN]
    com = rep[rep["train_n"] > RARE_MAX_TRAIN]
    row.update({
        "rare_recall": round(float(rare["recall"].mean()), 3),
        "rare_f1": round(float(rare["f1"].mean()), 3),
        "common_recall": round(float(com["recall"].mean()), 3),
        "common_f1": round(float(com["f1"].mean()), 3),
        "n_rare": len(rare), "n_common": len(com),
    })
    return row, rep


# ================================ MAIN ================================
def main(quick=False):
    ep = dict(epochs=8, batches=8, episodes=8, patience=4) if quick else dict(epochs=200, batches=40, episodes=40, patience=20)
    n_draws = 15 if quick else 60
    n_episodes = 60 if quick else 400
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("--- Caricamento dataset multiclasse ricco (64 spettro 3-mer + 10 descrittori) ---")
    tr, va, te = build_rich_splits()
    classes = sorted(tr["disease"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    support_train = tr["disease"].value_counts().to_dict()

    Xtr = tr[FEATURE_COLS].to_numpy(np.float32)
    Xva = va[FEATURE_COLS].to_numpy(np.float32)
    Xte = te[FEATURE_COLS].to_numpy(np.float32)
    ytr = tr["disease"].map(c2i).to_numpy()
    yva = va["disease"].map(c2i).to_numpy()
    yte = te["disease"].map(c2i).to_numpy()
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)
    print(f"Classi={n_classes}  train={len(Xtr)} val={len(Xva)} test={len(Xte)}  "
          f"chance={1/n_classes:.3f}\n")

    # ---- addestramento delle varianti (encoder, metrica di predizione) ----
    print("=== Addestramento varianti (ribilanciate) ===")
    encoders = {}  # nome -> (encoder, cosine_bool)

    enc, v = train_prototypical(Xtr, ytr, Xva, yva, n_classes, cosine=True, seed=SEED,
                                select_metric="balanced", epochs=ep["epochs"],
                                episodes=ep["episodes"], patience=ep["patience"])
    encoders["Prototipico coseno"] = (enc, True); print(f"  Prototipico coseno   val_bal_acc={v:.3f}")

    enc, v = train_prototypical(Xtr, ytr, Xva, yva, n_classes, cosine=False, seed=SEED,
                                select_metric="balanced", epochs=ep["epochs"],
                                episodes=ep["episodes"], patience=ep["patience"])
    encoders["Prototipico euclideo"] = (enc, False); print(f"  Prototipico euclideo val_bal_acc={v:.3f}")

    enc, v = train_metric_encoder(Xtr, ytr, Xva, yva, n_classes, "triplet", margin=0.3,
                                  epochs=ep["epochs"], batches=ep["batches"], patience=ep["patience"])
    encoders["Triplet (batch-hard)"] = (enc, False); print(f"  Triplet              val_bal_acc={v:.3f}")

    enc, v = train_metric_encoder(Xtr, ytr, Xva, yva, n_classes, "contrastive", margin=1.0,
                                  epochs=ep["epochs"], batches=ep["batches"], patience=ep["patience"])
    encoders["Contrastive (coppie)"] = (enc, False); print(f"  Contrastive          val_bal_acc={v:.3f}")

    smax = train_softmax(Xtr, ytr, Xva, yva, n_classes, seed=SEED, balanced=True,
                         epochs=ep["epochs"] * (25 if quick else 1), patience=ep["patience"])

    # ---- (1) metriche multiclasse standard + rare vs comuni ----
    print("\n=== (1) Multiclasse standard su test (acc / bal_acc / macro-F1 / rare) ===")
    rows, per_class_tables = [], {}
    for name, (enc, cos) in encoders.items():
        proba = predict_prototypical(enc, Xtr, ytr, Xte, n_classes, cosine=cos)
        row, rep = full_metrics(name, yte, proba, n_classes, classes, support_train)
        rows.append(row); per_class_tables[name] = rep
    row, rep = full_metrics("Softmax bilanciato", yte, predict_softmax(smax, Xte),
                            n_classes, classes, support_train)
    rows.append(row); per_class_tables["Softmax bilanciato"] = rep

    res = pd.DataFrame(rows).sort_values("rare_f1", ascending=False)
    print("\n" + res.round(3).to_string(index=False))
    res.to_csv(os.path.join(RESULTS_DIR, "multiclass_loss_comparison.tsv"), sep="\t", index=False)

    # ---- (2) few-shot episodico N-way K-shot ----
    print(f"\n=== (2) Few-shot episodico su test (chance 5-way=0.200) ===")
    fs_rows = []
    for name, (enc, cos) in encoders.items():
        for n_way, k_shot in [(5, 1), (5, 5), (n_classes, 1), (n_classes, 5)]:
            m, sd, nw = few_shot_eval(enc, Xte, yte, cos, n_way, k_shot, n_episodes=n_episodes)
            tag = "one-shot" if k_shot == 1 else f"{k_shot}-shot"
            fs_rows.append({"modello": name, "n_way": nw, "k_shot": k_shot,
                            "tipo": tag, "acc_media": round(m, 3), "acc_std": round(sd, 3),
                            "chance": round(1 / nw, 3)})
    fs = pd.DataFrame(fs_rows)
    print("\n" + fs.to_string(index=False))
    fs.to_csv(os.path.join(RESULTS_DIR, "multiclass_fewshot.tsv"), sep="\t", index=False)

    # ---- (3) one-shot 'full-way' per-classe (miglior encoder per rare_f1) ----
    best_name = res[res["model"].isin(encoders)]["model"].iloc[0]
    best_enc, best_cos = encoders[best_name]
    print(f"\n=== (3) One-shot/5-shot full-way per-classe — encoder: {best_name} ===")
    per_class_rows = []
    for k_shot in (1, 5):
        accs, rec = k_shot_full(best_enc, Xtr, ytr, Xte, yte, n_classes, best_cos, k_shot, n_draws=n_draws)
        print(f"  {k_shot}-shot full-way: acc={np.mean(accs):.3f}±{np.std(accs):.3f} (chance {1/n_classes:.3f})")
        for c, r in zip(classes, rec):
            per_class_rows.append({"malattia": c, "train_n": support_train.get(c, 0),
                                   "k_shot": k_shot, "recall": round(float(r), 3)})
    pc = pd.DataFrame(per_class_rows).sort_values(["k_shot", "train_n"])
    pc.to_csv(os.path.join(RESULTS_DIR, "multiclass_oneshot_per_class.tsv"), sep="\t", index=False)
    print("\n  recall 1-shot dalla classe piu' rara:")
    print(pc[pc.k_shot == 1].head(8).to_string(index=False))

    print(f"\nRisultati salvati in {RESULTS_DIR}/:")
    print("  multiclass_loss_comparison.tsv, multiclass_fewshot.tsv, multiclass_oneshot_per_class.tsv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="run breve per test di correttezza")
    main(**vars(ap.parse_args()))
