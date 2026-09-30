"""EDA sul dataset ORIGINALE completo (eccdna_disease_detection_metadata.tsv,
~3,7M righe) per capire la STRUTTURA delle malattie rispetto ai confondenti.

Per ogni malattia calcola, sull'intero dataset (lettura a blocchi, RAM-safe):
- n campioni
- 'purezza' per METODO      : frazione del metodo dominante (1.0 = un solo protocollo)
- 'purezza' per TESSUTO     : frazione del tessuto dominante
- 'purezza' per SOURCE_DB   : frazione dello studio/DB dominante (effetto batch)
- lunghezza media, gc medio

A cosa serve:
- purezza METODO alta = la malattia e' quasi definita dal protocollo -> classificarla
  potrebbe voler dire riconoscere il protocollo, non la biologia (rilevante sia per
  la multiclasse sia per l'open-set: una 'nuova' malattia con metodo unico e'
  rilevabile in modo banale/confuso col metodo).
- purezza SOURCE_DB alta = 'malattia' ~ 'studio' (batch effect).
- malattie quasi-duplicate (es. Stomach / Stomach cancer / Gastric cancer) con
  stesso tessuto/studio = probabilmente la STESSA entita' biologica etichettata
  diversamente -> candidate a fusione per una multiclasse piu' pulita.
"""

import pandas as pd

METADATA = "data/processed/eccdna_disease_detection_metadata.tsv"
CHUNK = 250000
CAT_COLS = ["method", "tissue", "source_db"]

# le 17 malattie del nostro task (per evidenziarle nell'output)
NOSTRE = {
    "Gastric cancer", "Stomach cancer", "Stomach", "Colorectal cancer", "Colorectal adenoma",
    "Cataract", "Primary pulmonary hypertension", "Hypopharyngeal squamous cell carcinoma",
    "Hypopharynx cancer", "Dilated cardiomyopathy", "Chronic kidney disease", "Breast cancer",
    "Glioblastoma cancer", "Coronary artery disease", "Caid syndrome",
    "Systemic lupus erythematosus", "Liver cancer",
}


def main():
    cont = {c: None for c in CAT_COLS}          # Series contingenza disease x categoria
    numstats = None                              # somma/conteggio length, gc per disease
    n_rows = 0

    usecols = ["disease"] + CAT_COLS + ["length", "gc"]
    for i, chunk in enumerate(pd.read_csv(METADATA, sep="\t", usecols=lambda c: c in usecols,
                                          chunksize=CHUNK, low_memory=False)):
        n_rows += len(chunk)
        chunk["disease"] = chunk["disease"].fillna("Unknown").astype(str)
        for c in CAT_COLS:
            if c not in chunk.columns:
                continue
            chunk[c] = chunk[c].fillna("Unknown").astype(str)
            g = chunk.groupby(["disease", c]).size()
            cont[c] = g if cont[c] is None else cont[c].add(g, fill_value=0)
        ns = chunk.groupby("disease").agg(
            n=("disease", "size"),
            len_sum=("length", "sum"), len_cnt=("length", "count"),
            gc_sum=("gc", "sum"), gc_cnt=("gc", "count"))
        numstats = ns if numstats is None else numstats.add(ns, fill_value=0)
        print(f"  blocco {i+1} elaborato ({n_rows} righe)")

    print(f"\nRighe totali: {n_rows}   Malattie distinte: {numstats.shape[0]}\n")

    def purity(series_col):
        # per ogni disease: (categoria dominante, frazione)
        out = {}
        for disease, sub in series_col.groupby(level=0):
            tot = sub.sum()
            top = sub.idxmax()[1]
            out[disease] = (top, sub.max() / tot if tot else 0.0)
        return out

    pur = {c: purity(cont[c]) for c in CAT_COLS if cont[c] is not None}

    rows = []
    for disease, r in numstats.iterrows():
        row = {"disease": disease, "n": int(r["n"]),
               "len_med": r["len_sum"] / r["len_cnt"] if r["len_cnt"] else float("nan"),
               "gc_med": r["gc_sum"] / r["gc_cnt"] if r["gc_cnt"] else float("nan")}
        for c in CAT_COLS:
            top, frac = pur.get(c, {}).get(disease, ("-", float("nan")))
            row[f"{c}_top"] = top
            row[f"{c}_pur"] = frac
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("n", ascending=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 30)
    print("=== TUTTE le malattie (ordinate per numerosita') ===")
    cols = ["disease", "n", "method_top", "method_pur", "tissue_top", "tissue_pur",
            "source_db_top", "source_db_pur", "len_med", "gc_med"]
    print(df[cols].round(3).to_string(index=False))

    print("\n=== SOLO le 17 del nostro task ===")
    ours = df[df["disease"].isin(NOSTRE)].copy()
    print(ours[cols].round(3).to_string(index=False))

    print("\n=== Gruppi quasi-duplicati: stessa biologia o studi diversi? ===")
    for grp in [["Stomach", "Stomach cancer", "Gastric cancer"],
                ["Colorectal cancer", "Colorectal adenoma"],
                ["Hypopharynx cancer", "Hypopharyngeal squamous cell carcinoma"]]:
        sub = df[df["disease"].isin(grp)]
        if not sub.empty:
            print(sub[["disease", "n", "tissue_top", "tissue_pur",
                       "source_db_top", "source_db_pur", "method_top", "method_pur"]].round(3).to_string(index=False))
            print()

    df.to_csv("data/processed/full_dataset_disease_profile.tsv", sep="\t", index=False)
    print("Profilo per malattia salvato in data/processed/full_dataset_disease_profile.tsv")


if __name__ == "__main__":
    main()
