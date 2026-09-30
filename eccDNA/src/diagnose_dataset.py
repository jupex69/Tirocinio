"""Diagnostica COMPLETA del dataset di partenza (eccdna_disease_detection_metadata.tsv).

Obiettivo: catalogare in modo sistematico TUTTI i problemi del file grezzo, per poi
decidere la soluzione metodologicamente piu' corretta. Lettura a blocchi (RAM-safe).

Sezioni:
 1  Overview e valori mancanti
 2  Etichetta binaria (sano/malato): bilanciamento
 3  CONFONDENTE BINARIO: sano vs malato differiscono per metodo/studio/lunghezza?
 4  Panorama malattie: numero, imbalance
 5  CONFONDENTE MULTICLASSE: purezza per malattia (metodo/studio/tessuto/linea/library)
 6  Linee cellulari: quante 'malattie' sono di fatto una sola cell line
 7  Igiene etichette: doppioni per maiuscole/minuscole, stesso tessuto in piu' studi
 8  Confondente tecnico: lunghezza e gc per metodo
 9  Qualita': frazione di N, sequence_kind
 10 Split/leakage: distribuzione split, coerenza cluster->split, duplicati di locus (campione)
"""

import numpy as np
import pandas as pd

META = "data/processed/eccdna_disease_detection_metadata.tsv"
CHUNK = 250000
CATS = ["method", "source_db", "tissue", "cell_line", "library_type"]


def add(a, b):
    return b if a is None else a.add(b, fill_value=0)


def main():
    n_rows = 0
    nulls = None
    bin_counts = None
    bin_by = {c: None for c in ["method", "source_db"]}
    bin_len = None                                  # length per label
    disease_counts = None
    cont = {c: None for c in CATS}                  # disease x categoria
    lengc_by_method = None                          # length/gc per metodo
    nfrac_stats = {"has_N": 0, "total": 0}
    seqkind = None
    split_counts = None
    split_ext = None
    # leakage su campione
    locus_sample = {}     # (chrom,start,end) -> set(split)
    cluster_sample = {}   # cluster_id -> set(split)

    usecols = ["id", "disease_binary_label", "disease_binary_name", "disease",
               "chrom", "start", "end", "length", "gc", "n_fraction", "source_db",
               "tissue", "cell_line", "method", "library_type", "cluster_id",
               "role", "sequence_kind", "split_cluster", "split_deepcircle_external"]

    for i, ch in enumerate(pd.read_csv(META, sep="\t", usecols=lambda c: c in usecols,
                                       chunksize=CHUNK, low_memory=False)):
        n_rows += len(ch)
        nulls = add(nulls, ch.isnull().sum())
        for c in ["disease", "disease_binary_name"] + CATS:
            if c in ch:
                ch[c] = ch[c].fillna("NA").astype(str)

        bin_counts = add(bin_counts, ch["disease_binary_name"].value_counts())
        for c in ["method", "source_db"]:
            bin_by[c] = add(bin_by[c], ch.groupby(["disease_binary_name", c]).size())
        bin_len = add(bin_len, ch.groupby("disease_binary_name")["length"].agg(["sum", "count"]))

        disease_counts = add(disease_counts, ch["disease"].value_counts())
        for c in CATS:
            cont[c] = add(cont[c], ch.groupby(["disease", c]).size())

        lengc_by_method = add(lengc_by_method, ch.groupby("method").agg(
            len_sum=("length", "sum"), len_cnt=("length", "count"),
            gc_sum=("gc", "sum"), gc_cnt=("gc", "count")))

        if "n_fraction" in ch:
            nfrac_stats["has_N"] += int((ch["n_fraction"].fillna(0) > 0).sum())
            nfrac_stats["total"] += len(ch)
        if "sequence_kind" in ch:
            seqkind = add(seqkind, ch["sequence_kind"].fillna("NA").astype(str).value_counts())
        if "split_cluster" in ch:
            split_counts = add(split_counts, ch["split_cluster"].fillna("NA").astype(str).value_counts())
        if "split_deepcircle_external" in ch:
            split_ext = add(split_ext, ch["split_deepcircle_external"].fillna("NA").astype(str).value_counts())

        # campione per leakage (prime 20k righe di ogni blocco)
        s = ch.head(20000)
        for _, r in s[["chrom", "start", "end", "cluster_id", "split_cluster"]].iterrows():
            sp = str(r["split_cluster"])
            locus_sample.setdefault((r["chrom"], r["start"], r["end"]), set()).add(sp)
            cluster_sample.setdefault(r["cluster_id"], set()).add(sp)

    def pct(x): return f"{100*x:.1f}%"

    print("="*70)
    print(f"1) OVERVIEW  righe={n_rows:,}")
    print("Valori mancanti (colonne con >0):")
    for c, v in nulls[nulls > 0].sort_values(ascending=False).items():
        print(f"   {c:26s} {int(v):>10,}  ({pct(v/n_rows)})")

    print("="*70)
    print("2) ETICHETTA BINARIA (bilanciamento)")
    print(bin_counts.astype(int).to_string())
    tot = bin_counts.sum()
    print(f"   totale={int(tot):,}")

    print("="*70)
    print("3) CONFONDENTE BINARIO: sano vs malato per metodo / studio / lunghezza")
    for c in ["method", "source_db"]:
        print(f"\n  distribuzione '{c}' entro ciascuna classe binaria (riga=classe):")
        tab = bin_by[c].unstack(fill_value=0)
        frac = tab.div(tab.sum(1), axis=0)
        print(frac.round(3).to_string())
    print("\n  lunghezza media per classe:")
    bl = bin_len.copy(); bl["len_media"] = bl["sum"]/bl["count"]
    print(bl[["count", "len_media"]].round(1).to_string())

    print("="*70)
    print(f"4) PANORAMA MALATTIE: {disease_counts.shape[0]} etichette distinte")
    dc = disease_counts.sort_values(ascending=False)
    print("  Top 8:"); print(dc.head(8).astype(int).to_string())
    print("  Bottom 8:"); print(dc.tail(8).astype(int).to_string())
    print(f"  Imbalance: max={int(dc.max()):,}  min={int(dc.min())}  ratio={dc.max()/dc.min():,.0f}x")

    print("="*70)
    print("5) CONFONDENTE MULTICLASSE: purezza per malattia (frazione categoria dominante)")

    def purity_table(series):
        out = {}
        for d, sub in series.groupby(level=0):
            tot = sub.sum()
            out[d] = sub.max()/tot if tot else 0.0
        return pd.Series(out)

    pur = {c: purity_table(cont[c]) for c in CATS}
    purdf = pd.DataFrame(pur)
    print("  Quante malattie hanno purezza == 1.0 (categoria unica):")
    for c in CATS:
        n1 = int((purdf[c] >= 0.999).sum())
        print(f"    {c:14s} {n1}/{purdf.shape[0]} malattie  (purezza media={purdf[c].mean():.3f})")

    print("="*70)
    print("6) LINEE CELLULARI: 'malattie' che sono ~una sola cell line (non paziente)")
    cl = purity_table(cont["cell_line"])
    # top dominante cell_line per malattia
    top_cl = {}
    for d, sub in cont["cell_line"].groupby(level=0):
        top_cl[d] = sub.idxmax()[1]
    cell_diseases = [(d, cl[d], top_cl[d]) for d in cl.index
                     if cl[d] >= 0.8 and str(top_cl[d]) not in ("NA", "nan", "Unknown", "None")]
    print(f"  malattie con >=80% una sola cell line: {len(cell_diseases)}")
    for d, p, name in sorted(cell_diseases, key=lambda x: -x[1])[:15]:
        print(f"    {d:38s} {pct(p)} {name}")

    print("="*70)
    print("7) IGIENE ETICHETTE: doppioni per maiuscole/minuscole")
    low = {}
    for d, n in disease_counts.items():
        low.setdefault(str(d).lower(), []).append((d, int(n)))
    dups = {k: v for k, v in low.items() if len(v) > 1}
    print(f"  gruppi di etichette che differiscono solo per maiuscole/spelling: {len(dups)}")
    for k, v in list(dups.items())[:12]:
        print(f"    {k}: {v}")

    print("="*70)
    print("8) CONFONDENTE TECNICO: lunghezza / gc per metodo")
    lm = lengc_by_method.copy()
    lm["len_media"] = lm["len_sum"]/lm["len_cnt"]; lm["gc_media"] = lm["gc_sum"]/lm["gc_cnt"]
    print(lm[["len_cnt", "len_media", "gc_media"]].sort_values("len_cnt", ascending=False).round(3).to_string())

    print("="*70)
    print("9) QUALITA'")
    print(f"  sequenze con N (n_fraction>0): {nfrac_stats['has_N']:,} / {nfrac_stats['total']:,} ({pct(nfrac_stats['has_N']/nfrac_stats['total'])})")
    if seqkind is not None:
        print("  sequence_kind:"); print(seqkind.astype(int).to_string())

    print("="*70)
    print("10) SPLIT / LEAKAGE")
    if split_counts is not None:
        print("  split_cluster:"); print(split_counts.astype(int).to_string())
    if split_ext is not None:
        print("  split_deepcircle_external:"); print(split_ext.astype(int).to_string())
    n_multi_locus = sum(1 for v in locus_sample.values() if len(v) > 1)
    n_multi_clus = sum(1 for v in cluster_sample.values() if len(v) > 1)
    print(f"  [campione ~{len(locus_sample):,} loci] loci (chrom,start,end) presenti in PIU' split: {n_multi_locus}")
    print(f"  [campione ~{len(cluster_sample):,} cluster] cluster_id presenti in PIU' split: {n_multi_clus}")
    print("  (loci multi-split = stessa sequenza in train e test = leakage; "
          "cluster multi-split = split per cluster non rispettato)")


if __name__ == "__main__":
    main()
