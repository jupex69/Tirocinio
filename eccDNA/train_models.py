"""Confronto di 6 modelli sul classificatore binario sano/malato (0/1,
indipendente dalla malattia specifica), sui 10 descrittori biologici/statistici.

Modelli (2 di machine learning classico, 4 reti neurali PyTorch):
- Random Forest (sklearn): ensemble ad alberi parallelo (bagging).
- Gradient Boosting (sklearn HistGradientBoostingClassifier): ensemble ad
  alberi sequenziale (boosting).
- MLP profondo (PyTorch): classificatore feed-forward diretto (10->64->32->16->1).
- MLP con self-attention sui descrittori (PyTorch): come l'MLP ma con un gate
  appreso che pesa le 10 feature per campione prima della rete.
- Rete siamese, metric learning puro (PyTorch): embedding appreso con triplet
  loss batch-hard su 18 classi (17 malattie + sano) e classificazione per
  prototipo piu' vicino (centroide sano vs centroide di malattia).
- Rete siamese, loss combinata (PyTorch): encoder con testa di classificazione,
  loss = BCE + triplet batch-hard leggera (peso 0.05).

SPLIT: train/val/test da 'split_cluster' (vedi training_data.py e
README_disease_detection.md del tutor) - non uno split casuale, per non
sovrastimare le performance con regioni genomiche simili in train e test.

STANDARDIZZAZIONE: media/std calcolate SOLO sul train e riapplicate a
val/test (mai il contrario, altrimenti leakage delle statistiche di test).

'disease' non e' mai una feature (X): viene tenuta solo per il campionamento
anti-bias delle triplette e la classe della siamese, e per il breakdown finale
per malattia delle metriche.

============================ RISULTATI OTTENUTI ============================
Dataset bilanciato (build_balanced_splits, cap 3000/malattia): train 62.090,
val 21.712, test 10.696 - tutti 50/50 sano/malato, method-matched per malattia,
su 17 malattie con segnale robusto. Metrica principale: ROC-AUC (indipendente
dalla soglia). Test set (10.696 sequenze), ordinati per ROC-AUC:

    Modello                      ROC-AUC  Accuracy  Precision  Recall   F1
    MLP con attenzione            0.779     0.706     0.675    0.795   0.730
    MLP profondo                  0.778     0.704     0.669    0.809   0.732
    Siamese (loss combinata)      0.776     0.706     0.680    0.777   0.725
    Gradient Boosting             0.769     0.699     0.667    0.794   0.725
    Random Forest                 0.758     0.680     0.632    0.859   0.728
    Siamese (metric learning)     0.545     0.527     0.517    0.824   0.636

Letture principali:
- I 4 approcci discriminativi (2 MLP, GBM, RF) piu' la siamese a loss combinata
  si raggruppano entro ~2 punti di AUC (0.758-0.779): con poche feature tabellari
  la complessita' del modello conta poco, il tetto lo danno i descrittori.
- L'MLP con attenzione e' primo, ma per un margine minimo sull'MLP profondo.
- Random Forest e' il piu' "prudente" (recall alta, ma piu' falsi positivi).
- La siamese a METRIC LEARNING PURO fallisce (0.545, quasi caso): il paradigma
  contrastivo non si adatta a poche feature scalari con classi molto sovrapposte.
  La variante a loss combinata raggiunge ~0.78 solo perche', col peso della
  triplet a 0.05, e' di fatto un MLP (la componente contrastiva, se conta, peggiora).

SET DA 14 A 10 DESCRITTORI: un'ablation ha rimosso termodinamica
(nn_stability_mean/std) e periodicita' (periodicity_3bp/10bp) - nn_stability_mean
era correlato 0.99 con gc_content, e togliendo tutte e 4 l'AUC calava solo di
~0.004-0.011 (entro il rumore). Set finale piu' essenziale a parita' di prestazioni.

L'AUC ~0.78 e' piu' bassa di un ipotetico ~0.9 perche' e' onesta: sopravvive al
controllo dei confondenti lunghezza+metodo e allo split per cluster genomico.
Output completo salvato in data/processed/model_comparison_results.tsv e
model_comparison_per_disease.tsv. (Ambiente ecc_Dna_hotspot, PyTorch 2.13 CPU,
seed 42; le reti neurali hanno una variabilita' run-to-run di ~+/-0.005.)
===========================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix,
)

from eccdna_utils import DESCRIPTOR_NAMES
from training_data import build_balanced_splits, summarize
from models_pytorch import (
    DeepMLP, AttentionGatedMLP, SiameseEncoder, SiameseWithHead,
    train_binary_classifier, predict_proba,
    train_siamese_batch_hard, multi_prototype_score, train_siamese_combined,
)

SEED = 42


def standardize(train_X, *other_Xs):
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0
    scaled = [(train_X - mean) / std]
    scaled += [(X - mean) / std for X in other_Xs]
    return scaled


def evaluate(name, y_true, scores, threshold=0.5):
    y_pred = (scores >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"{name:28s} acc={acc:.3f}  precision={precision:.3f}  recall={recall:.3f}  "
          f"f1={f1:.3f}  ROC-AUC={auc:.3f}  (TP={tp} FP={fp} FN={fn} TN={tn})")
    return {"model": name, "accuracy": acc, "precision": precision, "recall": recall,
            "f1": f1, "roc_auc": auc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_threshold(y_val, val_scores):
    """Soglia che massimizza la balanced accuracy sulla validation. Serve per i
    modelli il cui punteggio NON e' una probabilita' calibrata (la siamese: il
    suo score e' una differenza di similarita'), per cui 0.5 fisso sarebbe una
    soglia arbitraria e ingiusta nel confronto."""
    ordine = np.argsort(val_scores)
    candidati = np.unique(val_scores[ordine])
    y_val = np.asarray(y_val)
    best_t, best_bacc = 0.0, -1.0
    for t in candidati:
        pred = (val_scores >= t).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum(); fn = ((pred == 0) & (y_val == 1)).sum()
        tn = ((pred == 0) & (y_val == 0)).sum(); fp = ((pred == 1) & (y_val == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        bacc = (sens + spec) / 2
        if bacc > best_bacc:
            best_bacc, best_t = bacc, t
    return best_t


def per_disease_recall(name, test_df, scores, threshold=0.5):
    """Sensibilita' (recall) del modello per ciascuna malattia specifica del
    test set - utile per capire QUALI malattie ciascun modello fatica a
    riconoscere, non solo l'aggregato binario."""
    df = test_df.copy()
    df["pred"] = (scores >= threshold).astype(int)
    malati = df[df["y"] == 1]
    righe = []
    for disease, gruppo in malati.groupby("disease"):
        recall = (gruppo["pred"] == 1).mean()
        righe.append({"model": name, "disease": disease, "n": len(gruppo), "recall": recall})
    return pd.DataFrame(righe)


def main():
    print("--- Caricamento dataset BILANCIATO (50/50 sano/malato, method-matched per malattia) ---")
    train_df, val_df, test_df = build_balanced_splits()
    summarize(train_df, "train")
    summarize(val_df, "val")
    summarize(test_df, "test")

    X_train_raw = train_df[DESCRIPTOR_NAMES].to_numpy(dtype=np.float32)
    X_val_raw = val_df[DESCRIPTOR_NAMES].to_numpy(dtype=np.float32)
    X_test_raw = test_df[DESCRIPTOR_NAMES].to_numpy(dtype=np.float32)
    y_train = train_df["y"].to_numpy()
    y_val = val_df["y"].to_numpy()
    y_test = test_df["y"].to_numpy()
    disease_train = train_df["disease"].fillna("Healthy").to_numpy()
    # etichetta a 18 classi per la siamese batch-hard: 'Healthy' per i sani (che
    # nel dataset bilanciato portano il nome della malattia a cui sono abbinati,
    # non NaN), il nome della malattia per i malati.
    class_train = np.where(y_train == 0, "Healthy", disease_train)

    X_train, X_val, X_test = standardize(X_train_raw, X_val_raw, X_test_raw)

    results = []
    per_disease_frames = []

    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    scores = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", y_test, scores))
    per_disease_frames.append(per_disease_recall("Random Forest", test_df, scores))

    print("\n--- Gradient Boosting ---")
    gbm = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
    gbm.fit(X_train, y_train)
    scores = gbm.predict_proba(X_test)[:, 1]
    results.append(evaluate("Gradient Boosting", y_test, scores))
    per_disease_frames.append(per_disease_recall("Gradient Boosting", test_df, scores))

    print("\n--- MLP profondo (PyTorch) ---")
    mlp = DeepMLP(n_features=len(DESCRIPTOR_NAMES))
    mlp, val_auc = train_binary_classifier(mlp, X_train, y_train, X_val, y_val, seed=SEED)
    print(f"  (early stopping: miglior AUC di validation = {val_auc:.3f})")
    scores = predict_proba(mlp, X_test)
    results.append(evaluate("MLP profondo", y_test, scores))
    per_disease_frames.append(per_disease_recall("MLP profondo", test_df, scores))

    print("\n--- MLP con attenzione (PyTorch) ---")
    amlp = AttentionGatedMLP(n_features=len(DESCRIPTOR_NAMES))
    amlp, val_auc = train_binary_classifier(amlp, X_train, y_train, X_val, y_val, seed=SEED)
    print(f"  (early stopping: miglior AUC di validation = {val_auc:.3f})")
    scores = predict_proba(amlp, X_test)
    results.append(evaluate("MLP con attenzione", y_test, scores))
    per_disease_frames.append(per_disease_recall("MLP con attenzione", test_df, scores))

    # SIAMESE - due varianti, per mostrare cosa la rende (non) competitiva su
    # questo problema. Genuino metric learning (embedding + prototipo) contro
    # embedding con testa di classificazione (loss combinata triplet+BCE).
    # La soglia si calibra sulla validation: il punteggio a prototipo non e'
    # una probabilita', 0.5 fisso sarebbe arbitrario (l'AUC e' comunque
    # indipendente dalla soglia).
    print("\n--- Rete siamese, metric learning puro (batch-hard + prototipo) ---")
    siamese_ml = SiameseEncoder(n_features=len(DESCRIPTOR_NAMES))
    siamese_ml, centroid_healthy_ml, disease_centroids, val_auc = train_siamese_batch_hard(
        siamese_ml, X_train, class_train, y_train, X_val, y_val, seed=SEED,
    )
    print(f"  (early stopping: miglior AUC di validation = {val_auc:.3f}; "
          f"{len(disease_centroids)} prototipi di malattia)")
    val_scores = multi_prototype_score(siamese_ml, X_val, centroid_healthy_ml, disease_centroids)
    t = best_threshold(y_val, val_scores)
    scores = multi_prototype_score(siamese_ml, X_test, centroid_healthy_ml, disease_centroids)
    results.append(evaluate("Siamese (metric learning)", y_test, scores, threshold=t))
    per_disease_frames.append(per_disease_recall("Siamese (metric learning)", test_df, scores, threshold=t))

    print("\n--- Rete siamese, loss combinata (embedding triplet + testa BCE) ---")
    siamese_c = SiameseWithHead(n_features=len(DESCRIPTOR_NAMES))
    siamese_c, val_auc = train_siamese_combined(
        siamese_c, X_train, y_train, X_val, y_val, seed=SEED, triplet_weight=0.05,
    )
    print(f"  (early stopping: miglior AUC di validation = {val_auc:.3f})")
    scores = predict_proba(siamese_c, X_test)
    results.append(evaluate("Siamese (loss combinata)", y_test, scores))
    per_disease_frames.append(per_disease_recall("Siamese (loss combinata)", test_df, scores))

    print("\n--- RIEPILOGO (ordinato per ROC-AUC) ---")
    df_results = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print(df_results.to_string(index=False))
    df_results.to_csv("data/processed/model_comparison_results.tsv", sep="\t", index=False)

    print("\n--- RECALL PER MALATTIA (test set), per modello ---")
    df_per_disease = pd.concat(per_disease_frames, ignore_index=True)
    pivot = df_per_disease.pivot(index="disease", columns="model", values="recall")
    print(pivot.to_string())
    df_per_disease.to_csv("data/processed/model_comparison_per_disease.tsv", sep="\t", index=False)

    print("\nRisultati salvati in data/processed/model_comparison_results.tsv "
          "e data/processed/model_comparison_per_disease.tsv")


if __name__ == "__main__":
    main()
