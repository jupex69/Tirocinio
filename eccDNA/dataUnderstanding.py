"""Analisi esplorativa (EDA) e rilevamento di data leakage sui metadati
grezzi delle eccDNA, PRIMA di qualunque pulizia.

Cosa fa: legge data/processed/eccdna_disease_detection_metadata.tsv per
intero (a blocchi), riporta valori mancanti, statistiche sulle feature
numeriche (length, gc, start, end) e le frequenze delle feature
categoriche (disease, chrom, tissue, ...). Poi allena un RandomForest
"spia" per stimare quanto ciascuna feature (length, gc, start, end,
chrom, tissue) e' predittiva della label disease_binary_label.

Come lo fa: lettura a blocchi da 250.000 righe per restare RAM-safe sul
file completo (~3,7M righe); il RandomForest viene invece allenato su
un campione di 150.000 righe (troppo pesante su tutto il dataset).

Altre info: e' questo script che ha rivelato che 'tissue' e' quasi un
proxy diretto della disease (alta feature importance) - per questo
dataCleaning.py lo esclude di proposito dal dataset pulito. Il modello
spia testa anche 'method' e 'source_db' (protocollo/database di
sequenziamento, es. Circle-seq vs ATAC-seq vs WGS): sull'ultima esecuzione
'method' e' risultato il secondo confondente piu' forte in assoluto
(14.9% di importanza, contro il 61.3% di tissue), anche testato in modo
aggregato su disease_binary_label - per molti sottotipi la confusione con
la disease e' quasi totale (es. Fetal Growth Restriction ed Esophageal
Cancer sono ~100% WGS, mentre il pool sano e' quasi 0% WGS). Vedi il
campionamento bilanciato per metodo in
descriptor_understanding_by_disease.py, che e' nato proprio da questa
scoperta. Nota: qui il test e' aggregato su disease_binary_label (sano vs
qualunque malattia); un confondente che varia in direzioni diverse da
sottotipo a sottotipo puo' risultare piu' debole qui di quanto lo sia in
realta' per un singolo sottotipo - per questo l'analisi per-sottotipo in
descriptor_understanding_by_disease.py resta la verifica definitiva. Lo
script e' puramente di analisi/diagnostica: non produce file di output e
non fa parte della catena dati -> modello.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("--- FASE 1: INIZIALIZZAZIONE E CARICAMENTO A BLOCCHI ---")
metadata_path = "data/processed/eccdna_disease_detection_metadata.tsv"

print("Lettura del dataset in blocchi da 250.000 righe...")

# Colonne vitali per l'EDA. Scartiamo fin da subito i pesanti ID e le colonne inutili.
colonne_analisi = ['disease', 'disease_binary_label', 'length', 'gc', 'chrom', 'tissue', 'start', 'end', 'method', 'source_db']

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
df_rf['method'] = df_rf['method'].fillna('Unknown')
df_rf['source_db'] = df_rf['source_db'].fillna('Unknown')
df_rf = df_rf.dropna(subset=['length', 'gc', 'start', 'end', 'chrom', 'tissue', 'method', 'source_db'])

le = LabelEncoder()
df_rf['chrom_encoded'] = le.fit_transform(df_rf['chrom'].astype(str))
df_rf['tissue_encoded'] = le.fit_transform(df_rf['tissue'].astype(str))
df_rf['method_encoded'] = le.fit_transform(df_rf['method'].astype(str))
df_rf['source_db_encoded'] = le.fit_transform(df_rf['source_db'].astype(str))

X = df_rf[['length', 'gc', 'start', 'end', 'chrom_encoded', 'tissue_encoded', 'method_encoded', 'source_db_encoded']]
y = df_rf['disease_binary_label']

# Addestriamo la Random Forest "Spia"
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
rf_model.fit(X, y)

importanza = rf_model.feature_importances_
nomi_feature = ['Lunghezza (length)', 'Contenuto GC (gc)', 'Inizio (start)', 'Fine (end)', 'Cromosoma (chrom)',
                'Tessuto (tissue)', 'Metodo sequenziamento (method)', 'Database sorgente (source_db)']

df_importanza = pd.DataFrame({
    'Feature': nomi_feature,
    'Importanza (%)': importanza * 100
}).sort_values(by='Importanza (%)', ascending=False)

print("\nCLASSIFICA DIPENDENZA DAL TARGET (0-100%):")
print(df_importanza.to_string(index=False))

print("\n--- ANALISI COMPLETATA ---")