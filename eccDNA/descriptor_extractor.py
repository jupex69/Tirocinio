"""Estrazione su larga scala dei descrittori biologici/statistici eccDNA.

Stadio di PRODUZIONE della pipeline: calcola per ogni sequenza valida i
descrittori definiti in eccdna_utils.compute_sequence_descriptors e li scrive
su TSV, con lo stesso pattern RAM-safe/bufferizzato di kmer_extractor.py.

Da lanciare SOLO dopo aver stabilito, con descriptor_understanding.py, quali
descrittori sono davvero informativi rispetto a disease_binary_label: questo
script estrae su TUTTO il dataset (milioni di sequenze, FASTA da 6+ GB) e non
serve a scoprire i descrittori fondamentali, solo a produrre il file finale
per il training una volta che sappiamo quali tenere.

NOTA: la lunghezza della sequenza NON viene inclusa tra i descrittori.
E' stata deliberatamente esclusa da dataCleaning.py perche' identificata
come feature leaking (vedi dataUnderstanding.py) e non va reintrodotta senza
rifare quell'analisi.

Uso:
    python descriptor_extractor.py
"""

import argparse
import os

from eccdna_utils import DESCRIPTOR_NAMES, compute_sequence_descriptors, read_fasta_stream
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WRITE_BUFFER_SIZE = 5000
HEADER = ["id"] + DESCRIPTOR_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_descriptor_features.tsv"),
        help="percorso del TSV di output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("--- FASE 1: CARICAMENTO METADATI ---")
    df_clean = pd.read_csv(args.metadata_clean_path, sep="\t", usecols=["id"])
    id_validi = set(df_clean["id"].astype(str))
    print(f"Dobbiamo estrarre i descrittori per {len(id_validi)} frammenti validi.")

    print("\n--- FASE 2: ESTRAZIONE DESCRITTORI DAL FASTA (RAM-Safe, scrittura bufferizzata) ---")

    sequenze_processate = 0
    sequenze_salvate = 0
    buffer = []

    with open(args.output_path, "w") as f_out:
        f_out.write("\t".join(HEADER) + "\n")

        for seq_id, sequence in read_fasta_stream(args.fasta_path, wanted_ids=id_validi):
            sequenze_processate += 1

            if "N" not in sequence and len(sequence) >= 3:
                descrittori = compute_sequence_descriptors(sequence)
                # lz_complexity puo' essere None sulle sequenze molto lunghe (vedi
                # eccdna_utils.LZ_COMPLEXITY_MAX_LENGTH): si scrive "nan", pandas
                # lo interpreta correttamente in lettura.
                riga = [seq_id] + [str(descrittori[nome]) if descrittori[nome] is not None else "nan"
                                    for nome in DESCRIPTOR_NAMES]
                buffer.append("\t".join(riga))
                sequenze_salvate += 1

                if len(buffer) >= WRITE_BUFFER_SIZE:
                    f_out.write("\n".join(buffer) + "\n")
                    buffer.clear()

            if sequenze_processate % 50000 == 0:
                print(f"  Lette {sequenze_processate} sequenze dal FASTA. Salvate {sequenze_salvate} descrittori...")

        if buffer:
            f_out.write("\n".join(buffer) + "\n")

    print("\n--- FASE 3: COMPLETAMENTO ---")
    print(f"Estrazione completata! {len(DESCRIPTOR_NAMES)} descrittori salvati in: {args.output_path}")


if __name__ == "__main__":
    main()
