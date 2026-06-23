import pandas as pd
import os

print("--- FASE 1: INIZIALIZZAZIONE ---")
input_path = "data/processed/eccdna_disease_detection_metadata.tsv"
output_path = "data/processed/eccdna_metadata_CLEAN.tsv"

print(f"Lettura a blocchi (Chunking) da: {input_path}...")

# Definiamo le colonne da tenere (abbiamo escluso 'length' e 'tissue')
colonne_da_tenere = [
    "id", 
    "split_cluster", 
    "disease", 
    "disease_binary_label", 
    "chrom", 
    "start", 
    "end", 
    "gc"
]

# Dimensione del blocco: 250.000 righe alla volta è un numero sicurissimo per la RAM
chunk_size = 250000 
righe_totali_salvate = 0
primo_blocco = True

print("\n--- FASE 2: PULIZIA E SALVATAGGIO IN TEMPO REALE ---")

# Iteriamo sul file leggendolo a blocchi
for i, chunk in enumerate(pd.read_csv(input_path, sep="\t", chunksize=chunk_size, low_memory=False)):
    print(f"🔄 Elaborazione blocco {i+1} (Righe elaborate finora: {i * chunk_size})...")
    
    # 1. Correzione dei refusi
    if 'disease' in chunk.columns:
        chunk['disease'] = chunk['disease'].replace({'Health': 'Healthy'})
        chunk['disease'] = chunk['disease'].astype(str).str.capitalize()
    
    # 2. Selezione chirurgica (teniamo solo le colonne che ci servono)
    # Assicuriamoci che tutte le colonne richieste esistano nel blocco
    colonne_presenti = [col for col in colonne_da_tenere if col in chunk.columns]
    chunk_clean = chunk[colonne_presenti]
    
    # 3. Pulizia anomalie (via le righe senza descrittori genomici validi)
    subset_dropna = [col for col in ["gc", "chrom", "start", "end"] if col in chunk_clean.columns]
    chunk_clean = chunk_clean.dropna(subset=subset_dropna)
    
    # 4. Salvataggio su disco (Append)
    if primo_blocco:
        # Il primo blocco crea il file e scrive l'intestazione (header)
        chunk_clean.to_csv(output_path, sep="\t", index=False, mode='w', header=True)
        primo_blocco = False
    else:
        # I blocchi successivi si accodano al file senza riscrivere l'intestazione
        chunk_clean.to_csv(output_path, sep="\t", index=False, mode='a', header=False)
        
    righe_totali_salvate += len(chunk_clean)

print("\n--- FASE 3: COMPLETAMENTO ---")
print(f"Il nuovo dataset pulito è stato salvato in: {output_path}")
print(f"Sono state salvate {righe_totali_salvate} righe perfettamente pulite.")