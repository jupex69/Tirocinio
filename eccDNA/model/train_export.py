"""Addestra ed esporta i due modelli finali (RandomForest sui 10 descrittori
interpretabili) in formato joblib, pronti per l'uso via model/predict.py.

  1) binary_presence_rf.joblib   -- sano vs malato (RQ1): l'unico che GENERALIZZA
                                    a un nuovo studio (ROC-AUC ~0,71 cross-studio).
  2) multiclass_disease_rf.joblib -- quale delle 9 malattie (RQ2): utile SOLO entro
                                    lo stesso studio (non generalizza; vedi tesi).

Riusa i loader e le cache del progetto (src/), quindi va eseguito con i dati in
src/data/ presenti. Non richiede GPU. RandomForest e' invariante alla scala, quindi
i modelli lavorano direttamente sui 10 descrittori grezzi.
"""
import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

from eccdna_utils import DESCRIPTOR_NAMES               # i 10 descrittori interpretabili
from train_multiclass import SEED
from experiment_binary_crossstudy import load as load_binary
from experiment_disease_multiclass import load as load_disease, length_match

FEATURES = list(DESCRIPTOR_NAMES)
N_TREES = 400


def _rf():
    # min_samples_leaf contiene la dimensione degli alberi (file piu' piccolo)
    # senza intaccare in pratica l'accuratezza; compress in joblib riduce ancora.
    return RandomForestClassifier(n_estimators=N_TREES, random_state=SEED,
                                  n_jobs=-1, class_weight="balanced", min_samples_leaf=3)


def export_binary():
    df = load_binary()
    X = df[FEATURES].to_numpy(np.float32)
    y = np.where(df["y"].to_numpy() == 1, "malato", "sano")
    rf = _rf().fit(X, y)
    out = os.path.join(HERE, "binary_presence_rf.joblib")
    joblib.dump({"model": rf, "features": FEATURES, "classes": list(rf.classes_),
                 "task": "presenza (sano vs malato)", "n_train": int(len(y))}, out, compress=3)
    print(f"[binario]   {len(y)} campioni, classi {list(rf.classes_)} -> {os.path.basename(out)}")


def export_multiclass():
    df = length_match(load_disease())
    X = df[FEATURES].to_numpy(np.float32)
    y = df["label"].to_numpy()
    rf = _rf().fit(X, y)
    out = os.path.join(HERE, "multiclass_disease_rf.joblib")
    joblib.dump({"model": rf, "features": FEATURES, "classes": list(rf.classes_),
                 "task": "identita' (quale malattia, 9 classi)", "n_train": int(len(y))}, out, compress=3)
    print(f"[multiclasse] {len(y)} campioni, {len(rf.classes_)} malattie -> {os.path.basename(out)}")


if __name__ == "__main__":
    print("Feature (10 descrittori):", FEATURES)
    export_binary()
    export_multiclass()
    print("Fatto.")
