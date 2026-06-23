import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("--- FASE 1: INIZIALIZZAZIONE E CARICAMENTO (RAM-SAFE) ---")
os.makedirs('eda_plots', exist_ok=True)
metadata_path = "data/processed/eccdna_disease_detection_metadata.tsv"
print("Lettura intelligente del dataset (campionamento del 5% in tempo reale)...")

# TRUCCO RAM-SAFE: Leggiamo solo il 5% delle righe casualmente
prob_to_keep = 0.05
df = pd.read_csv(
    metadata_path, 
    sep="\t", 
    header=0, 
    skiprows=lambda i: i > 0 and random.random() > prob_to_keep,
    low_memory=False
)
print(f"Dataset caricato in sicurezza! Dimensioni del campione: {len(df)} righe e {len(df.columns)} colonne.")

print("\n--- FASE 2: DATA UNDERSTANDING COMPLETO ---")

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

print("\n--- FASE 3: GENERAZIONE GRAFICI E DISTRIBUZIONI ---")
print("Elaborazione dei grafici in corso (attendere)...")

# Creiamo una copia per sicurezza e impostiamo lo stile
df_plot = df.copy()
df_plot['Target'] = df_plot['disease_binary_label'].map({0: 'Healthy', 1: 'Disease'})
sns.set_theme(style="whitegrid", palette="muted")

# Grafico 1: Bilanciamento
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_plot, x='Target', order=['Healthy', 'Disease'])
plt.title('Distribuzione Sani vs Malati (sul campione)')
plt.savefig('eda_plots/01_class_distribution.png', dpi=300)
plt.close()

# Grafico 2: GC Content
if 'gc' in df_plot.columns:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_plot, x='gc', hue='Target', fill=True, common_norm=False, alpha=0.5)
    plt.title('Distribuzione del Contenuto GC')
    plt.savefig('eda_plots/02_gc_distribution.png', dpi=300)
    plt.close()

# Grafico 3: Lunghezza (Log10)
if 'length' in df_plot.columns:
    plt.figure(figsize=(10, 6))
    df_plot['log_length'] = np.log10(df_plot['length'] + 1)
    sns.kdeplot(data=df_plot, x='log_length', hue='Target', fill=True, common_norm=False, alpha=0.5)
    plt.title('Distribuzione della Lunghezza eccDNA (Scala Log10)')
    plt.savefig('eda_plots/03_length_distribution.png', dpi=300)
    plt.close()

print("Grafici delle distribuzioni salvati.")

print("\n--- FASE 4: ANALISI DELLE DIPENDENZE (FEATURE IMPORTANCE) ---")
print("Calcolo del peso delle variabili rispetto al target...")

# Pulizia veloce e codifica sul campione per il modello spia
df_rf = df_plot.copy()
df_rf['tissue'] = df_rf['tissue'].fillna('Unknown')
# Teniamo solo le colonne che hanno senso confrontare
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

# Grafico 4: Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(data=df_importanza, x='Importanza (%)', y='Feature', palette='viridis')
plt.title('Dipendenza delle Feature dal Target', fontsize=14)
plt.tight_layout()
plt.savefig('eda_plots/04_feature_importanza.png', dpi=300)
plt.close()

print("\n✅ Grafico delle dipendenze salvato come '04_feature_importanza.png'")
print("\n--- ANALISI COMPLETATA ---")