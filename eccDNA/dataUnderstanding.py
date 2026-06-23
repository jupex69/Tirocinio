import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print("--- FASE 1: INIZIALIZZAZIONE E CARICAMENTO ---")
os.makedirs('eda_plots', exist_ok=True)
metadata_path = "data/processed/eccdna_disease_detection_metadata.tsv"
print(f"Caricamento del dataset da: {metadata_path}...")

# Caricamento del dataset
df = pd.read_csv(metadata_path, sep="\t", low_memory=False)
print(f"Dataset caricato! Dimensioni totali: {len(df)} righe e {len(df.columns)} colonne.")

print("\n--- FASE 2: DATA UNDERSTANDING COMPLETO ---")

print("\n1. Valori Mancanti (Nulli) per colonna:")
nulls = df.isnull().sum()
if len(nulls[nulls > 0]) > 0:
    print(nulls[nulls > 0])
else:
    print("Nessun valore nullo trovato nelle colonne principali.")

print("\n2. Analisi delle FEATURE NUMERICHE (Calcoli Statistici):")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Selezioniamo in automatico tutte le colonne numeriche
numerics = df.select_dtypes(include=['int64', 'float64']).columns
print(df[numerics].describe())

print("\n3. Analisi delle FEATURE CATEGORICHE E TESTUALI (Frequenze):")
# Selezioniamo in automatico tutte le colonne di testo
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    valori_unici = df[col].nunique()
    print(f"\n▶ Feature: '{col}' (Valori unici totali: {valori_unici})")
    print("  I 3 valori più frequenti:")
    # Calcoliamo le percentuali sui dati presenti
    conteggi = df[col].value_counts(normalize=True).head(3) * 100
    for nome, perc in conteggi.items():
        print(f"    * {nome}: {perc:.1f}%")

print("\n--- FASE 3: GENERAZIONE GRAFICI ---")
print("Campionamento dati ed elaborazione dei grafici in corso (attendere)...")

# Campionamento sicuro per non far crashare la RAM con i grafici
df_sample = df.sample(n=min(100000, len(df)), random_state=42)
df_sample['Target'] = df_sample['disease_binary_label'].map({0: 'Healthy', 1: 'Disease'})

sns.set_theme(style="whitegrid", palette="muted")

# Grafico 1: Bilanciamento
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_sample, x='Target', order=['Healthy', 'Disease'])
plt.title('Distribuzione Sani vs Malati (sul Sample)')
plt.ylabel('Numero di campioni')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('eda_plots/01_class_distribution.png', dpi=300)
plt.close()

# Grafico 2: GC Content
if 'gc' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_sample, x='gc', hue='Target', fill=True, common_norm=False, alpha=0.5)
    plt.title('Distribuzione del Contenuto GC: Healthy vs Disease')
    plt.xlabel('Percentuale GC')
    plt.ylabel('Densità')
    plt.tight_layout()
    plt.savefig('eda_plots/02_gc_distribution.png', dpi=300)
    plt.close()

# Grafico 3: Lunghezza (Log10)
if 'length' in df.columns:
    plt.figure(figsize=(10, 6))
    df_sample['log_length'] = np.log10(df_sample['length'] + 1)
    sns.kdeplot(data=df_sample, x='log_length', hue='Target', fill=True, common_norm=False, alpha=0.5)
    plt.title('Distribuzione della Lunghezza eccDNA (Scala Log10)')
    plt.xlabel('Log10(Lunghezza)')
    plt.ylabel('Densità')
    plt.tight_layout()
    plt.savefig('eda_plots/03_length_distribution.png', dpi=300)
    plt.close()

# Grafico 4: Cromosomi
if 'chrom' in df.columns:
    plt.figure(figsize=(14, 6))
    chrom_order = df_sample['chrom'].value_counts().index
    sns.countplot(data=df_sample, x='chrom', hue='Target', order=chrom_order)
    plt.title('Distribuzione eccDNA nei vari Cromosomi')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Conteggio')
    plt.legend(title='Stato')
    plt.tight_layout()
    plt.savefig('eda_plots/04_chromosome_distribution.png', dpi=300)
    plt.close()

print("\n✅ Tutti i grafici sono stati salvati nella cartella 'eda_plots'!")
print("\n--- ANALISI COMPLETATA ---")