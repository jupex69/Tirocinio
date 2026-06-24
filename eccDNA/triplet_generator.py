import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import os

print("--- FASE 1: CARICAMENTO DATI PULITI ---")
input_path = "data/processed/eccdna_metadata_CLEAN.tsv"
df = pd.read_csv(input_path, sep="\t", low_memory=False)

print(f"Dataset caricato: {len(df)} righe. Inizio separazione tramite 'split_cluster'...")

# Separiamo rigorosamente i dati rispettando i cluster del tutor
df_train = df[df['split_cluster'] == 'train'].copy()
df_val = df[df['split_cluster'] == 'val'].copy()
df_test = df[df['split_cluster'] == 'test'].copy()

print(f"Dimensioni: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

print("\n--- FASE 2: FEATURE SCALING (Senza Data Leakage) ---")
# Le coordinate sono a 8 cifre, il GC è tra 0 e 1. Dobbiamo scalarle!
# IMPORTANTE: Addestriamo lo scaler SOLO sul Train per evitare leakage
scaler = StandardScaler()

colonne_da_scalare = ['start', 'end']
# Calcola media e varianza SOLO sul train
scaler.fit(df_train[colonne_da_scalare])

# Applica la trasformazione a tutti e tre gli split
df_train[colonne_da_scalare] = scaler.transform(df_train[colonne_da_scalare])
df_val[colonne_da_scalare] = scaler.transform(df_val[colonne_da_scalare])
df_test[colonne_da_scalare] = scaler.transform(df_test[colonne_da_scalare])

print("Scaling di 'start' ed 'end' completato con successo (Z-score).")

print("\n--- FASE 3: GENERAZIONE TRIPLETTE (CON BILANCIAMENTO) ---")

def generate_balanced_triplets(data, num_triplets):
    """
    Genera triplette (Ancora, Positivo, Negativo) bilanciando perfettamente
    le classi Sano (0) e Malato (1) come Ancora.
    """
    sani = data[data['disease_binary_label'] == 0]['id'].tolist()
    malati = data[data['disease_binary_label'] == 1]['id'].tolist()
    
    triplets = []
    
    # Metà delle triplette avrà un'ancora Sana, metà un'ancora Malata (Balancing)
    half_size = num_triplets // 2
    
    # 1. Generazione triplette con Ancora MALATA (Disease)
    for _ in range(half_size):
        anchor = random.choice(malati)
        positive = random.choice(malati)
        negative = random.choice(sani)
        # Evitiamo che ancora e positivo siano esattamente la stessa riga
        while positive == anchor:
            positive = random.choice(malati)
        triplets.append({'anchor_id': anchor, 'positive_id': positive, 'negative_id': negative})
        
    # 2. Generazione triplette con Ancora SANA (Healthy)
    for _ in range(half_size):
        anchor = random.choice(sani)
        positive = random.choice(sani)
        negative = random.choice(malati)
        while positive == anchor:
            positive = random.choice(sani)
        triplets.append({'anchor_id': anchor, 'positive_id': positive, 'negative_id': negative})
        
    # Mescoliamo le triplette per non darle alla rete tutte in ordine
    random.shuffle(triplets)
    return pd.DataFrame(triplets)

# Generiamo un numero ragionevole di triplette per addestrare la rete
# (Puoi aumentare questi numeri in base alla potenza del server/PC)
print("Generazione 100.000 triplette di Train (50% Sani, 50% Malati)...")
train_triplets = generate_balanced_triplets(df_train, 100000)

print("Generazione 20.000 triplette di Validation...")
val_triplets = generate_balanced_triplets(df_val, 20000)

print("Generazione 10.000 triplette di Test...")
test_triplets = generate_balanced_triplets(df_test, 10000)

print("\n--- FASE 4: SALVATAGGIO ---")
# CREIAMO UN NUOVO FILE PER I DATI SCALATI INVECE DI SOVRASCRIVERE IL VECCHIO
df_scaled = pd.concat([df_train, df_val, df_test])
scaled_output_path = "data/processed/eccdna_metadata_CLEAN_SCALED.tsv"
df_scaled.to_csv(scaled_output_path, sep="\t", index=False)
print(f"Dataset scalato salvato in: {scaled_output_path}")

# Salviamo i file delle triplette
os.makedirs("data/triplets", exist_ok=True)
train_triplets.to_csv("data/triplets/train_triplets.csv", index=False)
val_triplets.to_csv("data/triplets/val_triplets.csv", index=False)
test_triplets.to_csv("data/triplets/test_triplets.csv", index=False)

print("✅ Data Preparation conclusa! Triplette salvate in 'data/triplets/'.")