import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

print("--- FASE 1: INIZIALIZZAZIONE E CARICAMENTO ---")
# Creiamo una cartella per salvare i grafici se non esiste
os.makedirs("eda_plots", exist_ok=True)

# Percorso del file (usiamo solo le colonne che ci interessano per alleggerire la RAM)
metadata_path = "data/processed/eccdna_disease_detection_metadata.tsv"
cols_to_load = ["id", "disease_binary_label", "disease", "chrom", "start", "end", "length", "gc", "tissue", "source_db"]

print(f"Caricamento del dataset da: {metadata_path}...")
df = pd.read_csv(metadata_path, sep="\t", usecols=cols_to_load, low_memory=False)

print(f"Dataset caricato! Dimensioni totali: {df.shape[0]} righe e {df.shape[1]} colonne.\n")

print("--- FASE 2: DATA UNDERSTANDING DI BASE ---")
print("1. Valori Mancanti (Nulli) per colonna:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])
if missing_data.sum() == 0:
    print("Nessun valore mancante trovato nelle colonne selezionate!")

print("\n2. Statistiche descrittive delle variabili numeriche (length, gc):")
# Usiamo i float per evitare la notazione scientifica illeggibile
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(df[["length", "gc"]].describe())

print("\n--- FASE 3: GENERAZIONE GRAFICI ---")
print("Campionamento dati ed elaborazione dei grafici in corso (attendere)...")
# Per non far crashare Matplotlib con 3.7 milioni di punti, prendiamo un sample bilanciato
df_sample = df.sample(n=min(100000, len(df)), random_state=42)

# Mappatura della label per comodità visiva
df_sample['Target'] = df_sample['disease_binary_label'].map({0: 'Healthy', 1: 'Disease'})

# Stile globale dei grafici
sns.set_theme(style="whitegrid", palette="muted")

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_sample, x='Target', order=['Healthy', 'Disease'])
plt.title('Distribuzione Sani vs Malati (sul Sample)')
plt.ylabel('Numero di campioni')
# Aggiungiamo i numeri sopra le barre
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')
plt.tight_layout()
plt.savefig('eda_plots/01_class_distribution.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_sample, x='gc', hue='Target', fill=True, common_norm=False, alpha=0.5)
plt.title('Distribuzione del Contenuto GC: Healthy vs Disease')
plt.xlabel('Percentuale GC')
plt.ylabel('Densità')
plt.tight_layout()
plt.savefig('eda_plots/02_gc_distribution.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
# Usiamo la scala logaritmica perché il DNA varia da poche centinaia a milioni di basi
df_sample['log_length'] = np.log10(df_sample['length'] + 1)
sns.kdeplot(data=df_sample, x='log_length', hue='Target', fill=True, common_norm=False, alpha=0.5)
plt.title('Distribuzione della Lunghezza eccDNA (Scala Log10)')
plt.xlabel('Log10(Lunghezza)')
plt.ylabel('Densità')
plt.tight_layout()
plt.savefig('eda_plots/03_length_distribution.png', dpi=300)
plt.close()

plt.figure(figsize=(14, 6))
# Ordiniamo i cromosomi in modo logico se possibile
chrom_order = df_sample['chrom'].value_counts().index
sns.countplot(data=df_sample, x='chrom', hue='Target', order=chrom_order)
plt.title('Distribuzione eccDNA nei vari Cromosomi')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Conteggio')
plt.legend(title='Stato')
plt.tight_layout()
plt.savefig('eda_plots/04_chromosome_distribution.png', dpi=300)
plt.close()

print("\n✅ Tutti i grafici sono stati creati e salvati con successo nella cartella 'eda_plots'!")
print("\n--- ANALISI COMPLETATA ---")