"""Uso dei modelli esportati: da una sequenza di eccDNA ai 10 descrittori, poi la
predizione. Funziona sia da riga di comando sia importato come modulo.

Esempi:
    python predict.py --model binary --seq ACGTACGT...        # sano vs malato
    python predict.py --model multiclass --seq ACGT...        # quale malattia
    python predict.py --model binary --fasta campioni.fa      # una riga per sequenza

Nota onesta (dalla tesi): il modello BINARIO (sano/malato) generalizza a un
laboratorio nuovo; il MULTICLASSE identifica la malattia solo entro lo stesso studio
e non va usato come diagnosi. Vedi model/README.md.
"""
import os
import sys
import argparse
import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from eccdna_utils import compute_sequence_descriptors, read_fasta_stream

MODELS = {
    "binary": os.path.join(HERE, "binary_presence_rf.joblib"),
    "multiclass": os.path.join(HERE, "multiclass_disease_rf.joblib"),
}


def load(kind):
    if kind not in MODELS:
        raise ValueError(f"modello sconosciuto: {kind} (usa 'binary' o 'multiclass')")
    return joblib.load(MODELS[kind])


def predict(seq, kind="binary"):
    """Ritorna (etichetta_predetta, {classe: probabilita})."""
    m = load(kind)
    seq = seq.strip().upper()
    f = compute_sequence_descriptors(seq)
    x = np.array([[f[c] for c in m["features"]]], dtype=np.float32)
    pred = m["model"].predict(x)[0]
    proba = {c: float(p) for c, p in zip(m["model"].classes_, m["model"].predict_proba(x)[0])}
    return pred, proba


def _print(seq_id, pred, proba):
    order = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)
    conf = ", ".join(f"{c}={p:.2f}" for c, p in order[:4])
    print(f"{seq_id}\t-> {pred}\t({conf})")


def main():
    ap = argparse.ArgumentParser(description="Predizione da sequenza eccDNA.")
    ap.add_argument("--model", choices=list(MODELS), default="binary")
    ap.add_argument("--seq", help="una singola sequenza ACGT")
    ap.add_argument("--fasta", help="file FASTA con piu' sequenze")
    a = ap.parse_args()
    if not a.seq and not a.fasta:
        ap.error("fornisci --seq oppure --fasta")
    if a.seq:
        pred, proba = predict(a.seq, a.model)
        _print("seq", pred, proba)
    if a.fasta:
        for sid, seq in read_fasta_stream(a.fasta):
            if set(seq.upper()) <= set("ACGT") and len(seq) >= 4:
                pred, proba = predict(seq, a.model)
                _print(sid, pred, proba)


if __name__ == "__main__":
    main()
