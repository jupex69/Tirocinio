import pandas as pd
import random
import os

print("--- FASE 1: CARICAMENTO E SINCRONIZZAZIONE DATI ---")
# Carichiamo i metadati puliti
metadata_path = "data/processed/eccdna_metadata_CLEAN.tsv"
df_meta = pd.read_csv(metadata_path, sep="\t", low_memory=False)

# Siccome l'estrattore ha scartato ~5200 sequenze anomale, dobbiamo assicurarci 
# di usare SOLO gli ID che sono sopravvissuti e salvati nel file delle feature.
kmer_features_path = "data/processed/eccdna_kmer_3_features.tsv"
print("Lettura degli ID validi dal file dei k-meri...")
# Leggiamo solo la colonna 'id' per non intasarci la RAM
df_valid_ids = pd.read_csv(kmer_features_path, sep="\t", usecols=['id'])
valid_ids_set = set(df_valid_ids['id'].astype(str))

# Filtriamo i metadati mantenendo solo i sopravvissuti
df_meta = df_meta[df_meta['id'].astype(str).isin(valid_ids_set)].copy()
print(f"Metadati sincronizzati: {len(df_meta)} righe pronte per il bilanciamento.")

print("\n--- FASE 2: PREPARAZIONE DEGLI SPLIT ---")
df_train = df_meta[df_meta['split_cluster'] == 'train'].copy()
df_val = df_meta[df_meta['split_cluster'] == 'val'].copy()
df_test = df_meta[df_meta['split_cluster'] == 'test'].copy()


def generate_smart_triplets(df_split, num_triplets):
    """
    Genera triplette bilanciando perfettamente Sani/Malati e,
    all'interno dei malati, bilanciando uniformemente le diverse malattie
    per azzerare il bias del Cancro allo Stomaco.
    """
    # 1. Isoliamo i Sani
    sani_ids = df_split[df_split['disease_binary_label'] == 0]['id'].tolist()
    
    # 2. Raggruppiamo i Malati per tipo di malattia (Il trucco del Tutor)
    df_malati = df_split[df_split['disease_binary_label'] == 1]
    tipi_di_malattie = df_malati['disease'].unique().tolist()
    
    malati_per_tipo = {}
    for malattia in tipi_di_malattie:
        malati_per_tipo[malattia] = df_malati[df_malati['disease'] == malattia]['id'].tolist()
        
    def pick_random_disease_id():
        # Sceglie uniformemente un TIPO di malattia (Gastrico, Seno, ecc. hanno la stessa chance 1/N)
        scelta_malattia = random.choice(tipi_di_malattie)
        # Poi pesca un ID a caso da quel gruppo specifico
        return random.choice(malati_per_tipo[scelta_malattia])

    triplets = []
    half_size = num_triplets // 2
    
    # --- METÀ 1: Ancora Malata (Positivo=Malato, Negativo=Sano) ---
    for _ in range(half_size):
        anchor = pick_random_disease_id()
        positive = pick_random_disease_id()
        negative = random.choice(sani_ids)
        
        while positive == anchor:
            positive = pick_random_disease_id()
            
        triplets.append({'anchor_id': anchor, 'positive_id': positive, 'negative_id': negative})
        
    # --- METÀ 2: Ancora Sana (Positivo=Sano, Negativo=Malato) ---
    for _ in range(half_size):
        anchor = random.choice(sani_ids)
        positive = random.choice(sani_ids)
        negative = pick_random_disease_id()
        
        while positive == anchor:
            positive = random.choice(sani_ids)
            
        triplets.append({'anchor_id': anchor, 'positive_id': positive, 'negative_id': negative})
        
    random.shuffle(triplets)
    return pd.DataFrame(triplets)


print("\n--- FASE 3: GENERAZIONE TRIPLETTE BILANCIATE ---")
print("Generazione 100.000 triplette di Train (Anti-Bias Cancro Gastrico)...")
train_triplets = generate_smart_triplets(df_train, 100000)

print("Generazione 20.000 triplette di Validation...")
val_triplets = generate_smart_triplets(df_val, 20000)

print("Generazione 10.000 triplette di Test...")
test_triplets = generate_smart_triplets(df_test, 10000)

print("\n--- FASE 4: SALVATAGGIO ---")
os.makedirs("data/triplets", exist_ok=True)
train_triplets.to_csv("data/triplets/train_triplets.csv", index=False)
val_triplets.to_csv("data/triplets/val_triplets.csv", index=False)
test_triplets.to_csv("data/triplets/test_triplets.csv", index=False)

print("✅ Data Balancing concluso! Triplette perfette salvate in 'data/triplets/'.")