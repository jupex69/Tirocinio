import pandas as pd

# 1. Controlliamo il file TSV
tsv_path = "data/processed/eccdna_metadata_CLEAN.tsv"
df = pd.read_csv(tsv_path, sep="\t", nrows=5)
print("--- PRIMI 5 ID NEL FILE TSV ---")
for val in df['id'].astype(str).tolist():
    print(f"'{val}'")

# 2. Controlliamo il file FASTA
fasta_path = "data/processed/eccdna_disease_detection.body.fa"
print("\n--- PRIME 5 INTESTAZIONI NEL FASTA ---")
with open(fasta_path, 'r') as f:
    count = 0
    for line in f:
        if line.startswith(">"):
            print(f"Originale: '{line.strip()}' -> Estratto dal codice: '{line[1:].split()[0]}'")
            count += 1
            if count == 5:
                break