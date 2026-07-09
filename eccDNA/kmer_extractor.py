"""Estrazione delle frequenze di k-meri dalle sequenze eccDNA.

Sostituisce le due versioni precedenti (kmer_extractor.py per k=3 e
kmer_4_extractor.py per k=4, quasi identiche) con un unico script
parametrizzato via --k. Corregge inoltre due bug della versione precedente:

1. kmer_4_extractor.py usava percorsi assoluti Windows hardcoded
   (non portabile) mentre kmer_extractor.py usava percorsi relativi:
   ora si usano sempre percorsi relativi allo script, coerenti in entrambi
   i casi (k=3 e k=4).
2. Il file di output veniva aperto e chiuso ad ogni singola sequenza
   (~3,7 milioni di volte): ora si scrive con un unico file handle,
   bufferizzando le righe e facendo il flush a blocchi.

Il parsing del FASTA viene riusato da eccdna_utils.read_fasta_stream
(prima era duplicato in questo script, in kmer_4_extractor.py e in
debug_ids.py).

Uso:
    python kmer_extractor.py --k 3
    python kmer_extractor.py --k 4
"""

import argparse
import itertools
import os
from collections import Counter

import pandas as pd

from eccdna_utils import read_fasta_stream

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WRITE_BUFFER_SIZE = 5000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="lunghezza dei k-meri (default: 3)")
    parser.add_argument(
        "--fasta-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection.body.fa"),
        help="percorso del FASTA con le sequenze eccDNA",
    )
    parser.add_argument(
        "--metadata-clean-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_metadata_CLEAN.tsv"),
        help="percorso del TSV pulito (usato solo per la lista di id validi)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="percorso del TSV di output (default: data/processed/eccdna_kmer_{k}_features.tsv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    k = args.k
    output_path = args.output_path or os.path.join(SCRIPT_DIR, f"data/processed/eccdna_kmer_{k}_features.tsv")

    print("--- FASE 1: INIZIALIZZAZIONE BIOINFORMATICA ---")
    basi = ["A", "C", "G", "T"]
    tutti_i_kmeri = ["".join(p) for p in itertools.product(basi, repeat=k)]
    print(f"Vocabolario generato: {len(tutti_i_kmeri)} {k}-meri (da {'A' * k} a {'T' * k}).")

    print("\n--- FASE 2: CARICAMENTO METADATI ---")
    df_clean = pd.read_csv(args.metadata_clean_path, sep="\t", usecols=["id"])
    id_validi = set(df_clean["id"].astype(str))
    print(f"Dobbiamo estrarre i {k}-meri per {len(id_validi)} frammenti validi.")

    print("\n--- FASE 3: ESTRAZIONE K-MERI DAL FASTA (RAM-Safe, scrittura bufferizzata) ---")
    header = ["id"] + tutti_i_kmeri

    sequenze_processate = 0
    sequenze_salvate = 0
    buffer = []

    with open(output_path, "w") as f_out:
        f_out.write("\t".join(header) + "\n")

        for seq_id, sequence in read_fasta_stream(args.fasta_path, wanted_ids=id_validi):
            sequenze_processate += 1

            if "N" not in sequence and len(sequence) >= k:
                kmeri_estratti = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
                conteggi = Counter(kmeri_estratti)
                totale_kmeri = len(kmeri_estratti)

                frequenze = {kmer: 0.0 for kmer in tutti_i_kmeri}
                for kmer, count in conteggi.items():
                    if kmer in frequenze:
                        frequenze[kmer] = count / totale_kmeri

                riga = [seq_id] + [str(frequenze[kmer]) for kmer in tutti_i_kmeri]
                buffer.append("\t".join(riga))
                sequenze_salvate += 1

                if len(buffer) >= WRITE_BUFFER_SIZE:
                    f_out.write("\n".join(buffer) + "\n")
                    buffer.clear()

            if sequenze_processate % 50000 == 0:
                print(f"  Lette {sequenze_processate} sequenze dal FASTA. Salvate {sequenze_salvate} feature...")

        if buffer:
            f_out.write("\n".join(buffer) + "\n")

    print("\n--- FASE 4: COMPLETAMENTO ---")
    print(f"✅ Estrazione completata! Le firme biologiche ({len(tutti_i_kmeri)} {k}-meri) sono state salvate in: {output_path}")


if __name__ == "__main__":
    main()
