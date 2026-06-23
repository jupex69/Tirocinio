import pandas as pd
import os

print("--- FASE 1: CARICAMENTO DATI ---")
metadata_path = "data/processed/eccdna_disease_detection_metadata.tsv"
print(f"Lettura del dataset originale da: {metadata_path}...")

# Caricamento del dataset (low_memory=False per gestire tipi misti)
df = pd.read_csv(metadata_path, sep="\t", low_memory=False)
print(f"Dataset caricato: {len(df)} righe e {len(df.columns)} colonne.")

print("\n--- FASE 2: PULIZIA E CORREZIONE ---")

# 1. Unificazione delle etichette (Risoluzione dei refusi)
print("1. Correzione dei refusi nei nomi delle malattie...")
# Sostituiamo 'Health' con 'Healthy'
df['disease'] = df['disease'].replace({'Health': 'Healthy'})
# Uniformiamo il formato: tutto minuscolo con la prima lettera maiuscola
df['disease'] = df['disease'].str.capitalize() 

# 2. Gestione Valori Mancanti
print("2. Riempimento dei valori nulli in 'tissue'...")
df['tissue'] = df['tissue'].fillna('Unknown')

# 3. Rimozione Bias e Feature Inutili
print("3. Selezione delle feature (Rimozione 'length' e colonne inutili)...")
colonne_da_tenere = [
    "id",                     # Identificativo unico (collega al FASTA)
    "split_cluster",          # Fondamentale per non mischiare Train/Val/Test
    "disease",                # Target specifico (vitale per le Triplette)
    "disease_binary_label",   # Target binario (Sano vs Malato)
    "chrom",                  # Descrittore genomico
    "start",                  # Descrittore genomico
    "end",                    # Descrittore genomico
    "gc",                     # Descrittore statistico
    "tissue"                  # Metadato biologico ripulito
]

# Riduciamo il dataframe solo all'essenziale
df_clean = df[colonne_da_tenere]

# 4. Pulizia anomalie 
print("4. Rimozione di eventuali righe senza descrittori genomici validi...")
# Se per qualche motivo manca il GC o le coordinate, la riga è inservibile
df_clean = df_clean.dropna(subset=["gc", "chrom", "start", "end"])

print("\n--- FASE 3: SALVATAGGIO ---")
# Creiamo il percorso per il file pulito
output_path = "data/processed/eccdna_metadata_CLEAN.tsv"
df_clean.to_csv(output_path, sep="\t", index=False)

print(f"✅ Data Cleaning completato con successo!")
print(f"Il nuovo dataset pulito è stato salvato in: {output_path}")
print(f"Dimensioni finali: {len(df_clean)} righe e {len(df_clean.columns)} colonne.")