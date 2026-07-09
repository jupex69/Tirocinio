import pandas as pd
import itertools
from collections import Counter
import os

print("--- FASE 1: INIZIALIZZAZIONE BIOINFORMATICA ---")

# IMPOSTAZIONE TEST: k=4 (Per l'A/B Testing contro k=3)
k = 4

# Generiamo il vocabolario di tutti i possibili 4-meri (4^4 = 256 combinazioni)
basi = ['A', 'C', 'G', 'T']
tutti_i_kmeri = [''.join(p) for p in itertools.product(basi, repeat=k)]
print(f"Vocabolario generato: {len(tutti_i_kmeri)} {k}-meri (da AAAA a TTTT).")

# PERCORSI DEI FILE
fasta_path = r"C:\Users\annun\Desktop\ioNoNstudio\Tirocinio\eccDNA\data\processed\eccdna_disease_detection.body.fa"
metadata_clean_path = r"C:\Users\annun\Desktop\ioNoNstudio\Tirocinio\eccDNA\data\processed\eccdna_metadata_CLEAN.tsv"
output_path = rf"C:\Users\annun\Desktop\ioNoNstudio\Tirocinio\eccDNA\data\processed\eccdna_kmer_{k}_features.tsv"

print("\n--- FASE 2: CARICAMENTO METADATI ---")
df_clean = pd.read_csv(metadata_clean_path, sep="\t", usecols=['id', 'split_cluster', 'disease', 'disease_binary_label'])
id_validi = set(df_clean['id'].astype(str))
print(f"Dobbiamo estrarre i {k}-meri per {len(id_validi)} frammenti validi.")

print("\n--- FASE 3: ESTRAZIONE K-MERI DAL FASTA (RAM-Safe) ---")
header = ['id'] + tutti_i_kmeri
with open(output_path, 'w') as f_out:
    f_out.write('\t'.join(header) + '\n')

sequenze_processate = 0
sequenze_salvate = 0

def read_fasta(file_path):
    seq_id = None
    seq = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    yield seq_id, "".join(seq)
                # Isoliamo l'ID dal FASTA
                seq_id = line[1:].split('|')[0].strip()
                seq = []
            else:
                seq.append(line.upper())
        if seq_id:
            yield seq_id, "".join(seq)

# Processiamo il FASTA
for seq_id, sequence in read_fasta(fasta_path):
    sequenze_processate += 1
    
    if seq_id in id_validi:
        if 'N' not in sequence and len(sequence) >= k:
            # Estrazione e conteggio
            kmeri_estratti = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
            conteggi = Counter(kmeri_estratti)
            totale_kmeri = len(kmeri_estratti)
            
            # Calcolo frequenze
            frequenze = {kmer: 0.0 for kmer in tutti_i_kmeri}
            for kmer, count in conteggi.items():
                if kmer in frequenze:
                    frequenze[kmer] = count / totale_kmeri
            
            # Salvataggio
            riga = [seq_id] + [str(frequenze[kmer]) for kmer in tutti_i_kmeri]
            with open(output_path, 'a') as f_out:
                f_out.write('\t'.join(riga) + '\n')
            
            sequenze_salvate += 1
            
    if sequenze_processate % 50000 == 0:
        print(f"  Lette {sequenze_processate} sequenze dal FASTA. Salvate {sequenze_salvate} feature...")

print("\n--- FASE 4: COMPLETAMENTO ---")
print(f"✅ Estrazione completata! Le firme biologiche ({len(tutti_i_kmeri)} {k}-meri) sono state salvate in: {output_path}")