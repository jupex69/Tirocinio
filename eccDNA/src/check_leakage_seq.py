"""Verifica leakage ROBUSTA (indipendente dal database): confronto delle SEQUENZE.

Il test sulle coordinate era inconcludente (0 ovunque -> coordinate non
confrontabili tra DB). Qui si confrontano le SEQUENZE stesse: se PPH/DCM in
eccDNABase e in CircleBaseV2 condividono sequenze IDENTICHE -> leakage reale.
In piu' si guarda la lunghezza mediana per (malattia, DB): se PPH/DCM hanno una
lunghezza estrema e uguale nei due DB, il loro 'transfer' e' una firma tecnica
(tipo di campione) piu' che biologia.
"""

import hashlib
import os
import numpy as np
import pandas as pd

from eccdna_utils import read_fasta_stream
from full_classifier import canon, META, FASTA, CHUNK, RESULTS_DIR

CAP = 4000
DBS = ["eccDNABase", "CircleBaseV2"]
TARGET = {"Primary Pulmonary Hypertension", "Dilated Cardiomyopathy",  # sospette
          "Colorectal Cancer", "Prostate Cancer"}                     # contrasti (non trasferiscono)


def main():
    # id + lunghezza per (malattia, db)
    want = {(t, db): [] for t in TARGET for db in DBS}
    for ch in pd.read_csv(META, sep="\t",
                          usecols=["id", "disease", "disease_binary_name", "source_db", "length"],
                          chunksize=CHUNK, low_memory=False):
        ch = ch[ch["source_db"].isin(DBS)]
        ch = ch[ch["disease_binary_name"].astype(str).str.lower() != "healthy"]
        ch["lab"] = ch["disease"].map(canon)
        ch = ch[ch["lab"].isin(TARGET)]
        for (lab, db), g in ch.groupby(["lab", "source_db"]):
            buf = want[(lab, db)]
            if len(buf) < CAP:
                buf.extend(g[["id", "length"]].head(CAP - len(buf)).itertuples(index=False, name=None))
    id2key = {}
    lengths = {}
    for (lab, db), rows in want.items():
        lengths[(lab, db)] = [r[1] for r in rows]
        for rid, _ in rows:
            id2key[str(rid)] = (lab, db)

    # sequenze -> hash per (malattia, db)
    seqhash = {(t, db): set() for t in TARGET for db in DBS}
    for sid, seq in read_fasta_stream(FASTA, wanted_ids=set(id2key)):
        k = id2key.get(str(sid))
        if k is not None:
            seqhash[k].add(hashlib.md5(seq.encode()).hexdigest())

    print("=== Sequenze IDENTICHE condivise tra i due database ===")
    print("   (alta = leakage; bassa = studi indipendenti)\n")
    rows = []
    for t in sorted(TARGET):
        a, b = seqhash[(t, "eccDNABase")], seqhash[(t, "CircleBaseV2")]
        if not a or not b:
            print(f"  {t}: presente in un solo DB nel campione, salto"); continue
        inter = len(a & b); ov = inter / min(len(a), len(b))
        la = np.median(lengths[(t, "eccDNABase")]); lb = np.median(lengths[(t, "CircleBaseV2")])
        rows.append({"malattia": t, "seq_ecc": len(a), "seq_CB2": len(b), "identiche": inter,
                     "sovrapp_seq": round(ov, 3), "len_med_ecc": int(la), "len_med_CB2": int(lb)})
    res = pd.DataFrame(rows).sort_values("sovrapp_seq", ascending=False)
    print(res.to_string(index=False))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "leakage_check_seq.tsv"), sep="\t", index=False)

    print("\n>>> Lettura:")
    for _, r in res.iterrows():
        if r["sovrapp_seq"] >= 0.3:
            print(f"    {r['malattia']}: {r['sovrapp_seq']:.2f} sequenze identiche -> LEAKAGE tra i due DB")
        else:
            print(f"    {r['malattia']}: {r['sovrapp_seq']:.2f} -> no leakage; "
                  f"len {r['len_med_ecc']}/{r['len_med_CB2']} bp"
                  + ("  (lunghezza estrema/coerente = firma tecnica)" if r["len_med_ecc"] < 200 else ""))
    print(f"\nSalvato in {RESULTS_DIR}/leakage_check_seq.tsv")


if __name__ == "__main__":
    main()
