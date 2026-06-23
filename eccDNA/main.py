from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# --- IMPOSTAZIONI DI SICUREZZA PER LA RAM ---
NUM_SAMPLES = 50000  # Iniziamo con 50k per testare, poi potrai aumentarlo!

print("1. Caricamento Metadati (TSV)...")
metadata_path = "data/processed/eccdna_disease_detection_metadata.tsv"
df = pd.read_csv(metadata_path, sep="\t", usecols=["id", "split_cluster", "disease_binary_label"])

# Filtriamo per split e prendiamo un campione casuale per non saturare la memoria
train_meta = df[df["split_cluster"] == "train"].sample(n=NUM_SAMPLES, random_state=42)
val_meta = df[df["split_cluster"] == "val"].sample(n=int(NUM_SAMPLES * 0.2), random_state=42)

# Creiamo dei set velocissimi per la ricerca degli ID
valid_ids = set(train_meta["id"]).union(set(val_meta["id"]))

print("2. Lettura Intelligente del FASTA (solo le sequenze necessarie)...")
sequences = {}
with Path("data/processed/eccdna_disease_detection.body.fa").open() as handle:
    current_id = None
    chunks = []
    for line in handle:
        line = line.strip()
        if not line: continue
        
        if line.startswith(">"):
            # Salviamo la sequenza precedente se fa parte del nostro campione
            if current_id in valid_ids:
                sequences[current_id] = "".join(chunks)
            
            # Leggiamo il nuovo ID
            current_id = line[1:].split("|", 1)[0]
            chunks = []
        else:
            if current_id in valid_ids:
                chunks.append(line)
    
    # Non dimentichiamo l'ultima sequenza del file
    if current_id in valid_ids:
        sequences[current_id] = "".join(chunks)

print(f"Estratte {len(sequences)} sequenze genomiche.")

print("3. Estrazione K-mers (Traduzione DNA -> Parole)...")
def prepare_sequences(meta_df, seq_dict, k=4):
    X_str, y = [], []
    for _, row in meta_df.iterrows():
        seq_id = row["id"]
        if seq_id in seq_dict:
            seq = seq_dict[seq_id]
            # Tagliamo la sequenza in pezzetti di k lettere
            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            X_str.append(" ".join(kmers))
            y.append(row["disease_binary_label"])
    return X_str, y

X_train_str, y_train = prepare_sequences(train_meta, sequences)
X_val_str, y_val = prepare_sequences(val_meta, sequences)

from sklearn.feature_extraction.text import TfidfVectorizer

print("4. Vettorizzazione Bilanciata (TfidfVectorizer)...")
# Usa Tfidf per bilanciare le lunghezze diverse delle sequenze
vectorizer = TfidfVectorizer()
X_train_num = vectorizer.fit_transform(X_train_str)
X_val_num = vectorizer.transform(X_val_str)

print("5. Addestramento Modello (Logistic Regression)...")
# Rimosso n_jobs=-1 per evitare il warning
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_num, y_train)

print("\n--- RISULTATI ANALISI SEQUENZE (DNA) ---")
y_val_pred = model.predict(X_val_num)
y_val_proba = model.predict_proba(X_val_num)[:, 1]

print(classification_report(y_val, y_val_pred, target_names=["Healthy (0)", "Disease (1)"]))
print(f"ROC-AUC Score: {roc_auc_score(y_val, y_val_proba):.4f}")