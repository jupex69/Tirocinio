"""Materializza su disco il DATASET COMPATTATO: metadati con le etichette di
malattia FUSE sotto un'unica etichetta canonica (canon), piu' la classe Healthy.
Cosi' l'analisi non dipende piu' dalla funzione in memoria ed e' riproducibile.

Output: data/processed/eccdna_compacted_metadata.tsv
Colonne: id, label (canonica | Healthy), is_disease, source_db, method, length.
Stampa anche il riepilogo per etichetta (n, n_database, n_metodi) con un verdetto
preliminare: 'mono-studio' (1 database) o 'cross-testabile' (>=2 database).
"""

import os
import pandas as pd
import numpy as np

from full_classifier import canon, META, CHUNK

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data/processed/eccdna_compacted_metadata.tsv")
MIN_PER_DB = 150


def main():
    first = True
    counts = {}   # label -> {db: n}
    methods = {}  # label -> set(method)
    n_rows = 0
    for ch in pd.read_csv(META, sep="\t",
                          usecols=["id", "disease", "disease_binary_name", "source_db", "method", "length"],
                          chunksize=CHUNK, low_memory=False):
        healthy = ch["disease_binary_name"].astype(str).str.lower() == "healthy"
        ch["label"] = np.where(healthy, "Healthy", ch["disease"].map(canon))
        ch = ch[ch["label"].notna() & (ch["label"].astype(str) != "None")]
        ch["is_disease"] = np.where(ch["label"] == "Healthy", 0, 1)
        out = ch[["id", "label", "is_disease", "source_db", "method", "length"]]
        out.to_csv(OUT, sep="\t", index=False, mode="w" if first else "a", header=first)
        first = False
        n_rows += len(out)
        for lab, g in ch.groupby("label"):
            for db, n in g["source_db"].value_counts().items():
                counts.setdefault(lab, {})[db] = counts.get(lab, {}).get(db, 0) + n
            methods.setdefault(lab, set()).update(g["method"].astype(str).unique())

    print(f"Scritto {OUT}  ({n_rows} righe)\n")

    rows = []
    for lab, dbc in counts.items():
        if lab == "Healthy":
            continue
        n = sum(dbc.values())
        n_db_ok = sum(1 for v in dbc.values() if v >= MIN_PER_DB)
        rows.append({"malattia": lab, "n_totale": n, "n_database": len(dbc),
                     "n_db_>=150": n_db_ok, "n_metodi": len(methods[lab]),
                     "verdetto": "cross-testabile" if n_db_ok >= 2 else "mono-studio"})
    summ = pd.DataFrame(rows).sort_values(["verdetto", "n_totale"], ascending=[True, False])
    pd.set_option("display.max_rows", 200)
    print("=== Riepilogo per malattia (dataset compattato) ===")
    print(summ.to_string(index=False))
    ndis = len(summ)
    ncross = int((summ["verdetto"] == "cross-testabile").sum())
    print(f"\nMalattie reali (compattate): {ndis}")
    print(f"  mono-studio (1a1, non validabili cross-studio): {ndis - ncross}")
    print(f"  cross-testabili (>=2 database, >=150 ciascuno): {ncross}")
    summ.to_csv(os.path.join(os.path.dirname(OUT), "eccdna_compacted_summary.tsv"), sep="\t", index=False)
    print(f"\nRiepilogo salvato in {os.path.dirname(OUT)}/eccdna_compacted_summary.tsv")


if __name__ == "__main__":
    main()
