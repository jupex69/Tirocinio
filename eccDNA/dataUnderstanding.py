import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("--- FASE 1: INIZIALIZZAZIONE E CARICAMENTO A BLOCCHI ---")
metadata_path = "data/processed/eccdna_disease_detection_metadata.tsv"

print("Lettura del dataset in blocchi da 250.000 righe...")

# Colonne vitali per l'EDA. Scartiamo fin da subito i pesanti ID e le colonne inutili.
colonne_analisi = ['disease', 'disease_binary_label', 'length', 'gc', 'chrom', 'tissue', 'start', 'end']

chunk_size = 250000
lista_blocchi = []
righe_totali = 0

# Iteriamo leggendo tutto il file a blocchi
for i, chunk in enumerate(pd.read_csv(metadata_path, sep="\t", chunksize=chunk_size, low_memory=False)):
    print(f"🔄 Elaborazione blocco {i+1}...")
    righe_totali += len(chunk)
    
    # Teniamo solo le colonne che ci interessano (se esistono nel blocco)
    colonne_presenti = [col for col in colonne_analisi if col in chunk.columns]
    chunk_snello = chunk[colonne_presenti]
    
    # Aggiungiamo il blocco snellito alla nostra lista
    lista_blocchi.append(chunk_snello)

# Uniamo tutti i blocchi in un unico DataFrame
df = pd.concat(lista_blocchi, ignore_index=True)

print(f"\n✅ Lettura completata! Abbiamo in memoria il 100% dei dati ({len(df)} righe).")

print("\n--- FASE 2: DATA UNDERSTANDING COMPLETO (ESATTO SUL 100%) ---")

print("\n1. Valori Mancanti (Nulli) per colonna:")
nulls = df.isnull().sum()
if len(nulls[nulls > 0]) > 0:
    print(nulls[nulls > 0])
else:
    print("Nessun valore nullo trovato nelle colonne principali.")

print("\n2. Analisi delle FEATURE NUMERICHE (Calcoli Statistici):")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
numerics = df.select_dtypes(include=['int64', 'float64']).columns
print(df[numerics].describe())

print("\n3. Analisi delle FEATURE CATEGORICHE E TESTUALI (Frequenze):")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    valori_unici = df[col].nunique()
    print(f"\n▶ Feature: '{col}' (Valori unici totali: {valori_unici})")
    conteggi = df[col].value_counts(normalize=True).head(3) * 100
    for nome, perc in conteggi.items():
        print(f"    * {nome}: {perc:.1f}%")

print("\n--- FASE 3: ANALISI DELLE DIPENDENZE (FEATURE IMPORTANCE) ---")
print("Calcolo del peso delle variabili rispetto al target (su un campione RAM-Safe di 150.000 righe)...")

# TRUCCO RAM-SAFE: Estraiamo un campione rappresentativo per la Random Forest
df_rf = df.sample(n=min(150000, len(df)), random_state=42).copy()

# Pulizia e preparazione per il modello spia
df_rf['tissue'] = df_rf['tissue'].fillna('Unknown')
df_rf = df_rf.dropna(subset=['length', 'gc', 'start', 'end', 'chrom', 'tissue'])

le = LabelEncoder()
df_rf['chrom_encoded'] = le.fit_transform(df_rf['chrom'].astype(str))
df_rf['tissue_encoded'] = le.fit_transform(df_rf['tissue'].astype(str))

X = df_rf[['length', 'gc', 'start', 'end', 'chrom_encoded', 'tissue_encoded']]
y = df_rf['disease_binary_label']

# Addestriamo la Random Forest "Spia"
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
rf_model.fit(X, y)

importanza = rf_model.feature_importances_
nomi_feature = ['Lunghezza (length)', 'Contenuto GC (gc)', 'Inizio (start)', 'Fine (end)', 'Cromosoma (chrom)', 'Tessuto (tissue)']

df_importanza = pd.DataFrame({
    'Feature': nomi_feature,
    'Importanza (%)': importanza * 100
}).sort_values(by='Importanza (%)', ascending=False)

print("\nCLASSIFICA DIPENDENZA DAL TARGET (0-100%):")
print(df_importanza.to_string(index=False))

print("\n--- ANALISI COMPLETATA ---")