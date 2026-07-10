"""Funzioni condivise dalla pipeline eccDNA.

lettura streaming del FASTA, codifica one-hot
con augmentation circolare (per il classificatore CNN) e campionamento
bilanciato sano/malato (usato da build_classification_dataset.py).

NOTA: triplet_generator.py NON usa queste funzioni di proposito, per non
cambiare la logica random che ha generato i triplet CSV gia' committati in
data/triplets/.
"""

import random

import numpy as np
import pandas as pd

BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}


def extract_fasta_id(header_line):
    """Estrae l'id dalla riga di intestazione FASTA '>id|label=...|...'."""
    return header_line[1:].strip().split("|")[0].strip()


def read_fasta_stream(fasta_path, wanted_ids=None):
    """Generatore (id, sequenza) che legge il FASTA una riga alla volta.

    Se 'wanted_ids' e' un set, le sequenze non richieste vengono scartate
    subito (non accumulate in memoria) - utile per caricare solo le poche
    decine di migliaia di sequenze necessarie al training senza portare in
    RAM l'intero file da 6+ GB.
    """
    seq_id = None
    chunks = []

    def _wanted(_id):
        return wanted_ids is None or _id in wanted_ids

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None and _wanted(seq_id):
                    yield seq_id, "".join(chunks)
                seq_id = extract_fasta_id(line)
                chunks = []
            else:
                chunks.append(line.upper())
        if seq_id is not None and _wanted(seq_id):
            yield seq_id, "".join(chunks)


def load_sequences(fasta_path, wanted_ids):
    """Carica in un dict SOLO le sequenze i cui id sono in 'wanted_ids'."""
    wanted_ids = set(str(i) for i in wanted_ids)
    sequences = {}
    for seq_id, seq in read_fasta_stream(fasta_path, wanted_ids=wanted_ids):
        sequences[seq_id] = seq
    return sequences


def one_hot_encode_circular(sequence, window_size=1024, wrap_len=64):
    """Codifica one-hot (4, window_size) con augmentation circolare.

    L'eccDNA e' una molecola circolare: non ha un vero "inizio" o "fine".
    Invece di riempire con zeri le sequenze piu' corte della finestra
    (come farebbe un padding lineare classico), la sequenza viene
    "arrotolata" (tiling circolare) fino a riempire la finestra - idea
    alla base della cyclic-padding di ECCNET. In coda alla finestra viene
    inoltre sempre appesa una piccola porzione iniziale della sequenza
    (default 64 basi, come l'augmentation di eccDNAMamba) per rinforzare
    l'adiacenza testa-coda anche quando la sequenza viene troncata perche'
    piu' lunga della finestra.

    Basi non standard (es. 'N') vengono codificate come colonna tutta zero,
    seguendo la stessa convenzione usata da DeepECC.
    """
    if window_size <= wrap_len:
        raise ValueError("window_size deve essere maggiore di wrap_len")
    if len(sequence) == 0:
        raise ValueError("sequenza vuota")

    core_len = window_size - wrap_len
    sequence = sequence.upper()
    seq_len = len(sequence)

    def _circular_take(s, length):
        if len(s) >= length:
            return s[:length]
        reps = length // len(s) + 1
        return (s * reps)[:length]

    core = _circular_take(sequence, core_len)
    wrap = _circular_take(sequence, wrap_len)
    window = core + wrap

    encoded = np.zeros((4, window_size), dtype=np.float32)
    for i, base in enumerate(window):
        idx = BASE_TO_IDX.get(base)
        if idx is not None:
            encoded[idx, i] = 1.0
    return encoded


def sample_balanced_ids(df_split, n, seed=42, id_col="id",
                         label_col="disease_binary_label", disease_col="disease"):
    """Campiona ~n righe da df_split, bilanciate 50/50 sano/malato e,
    tra i malati, il piu' possibile uniformi tra i diversi tipi di malattia
    (stessa idea anti-bias-Cancro-Gastrico di triplet_generator.py, ma qui
    si campionano righe unica (senza ripetizioni) invece di ancore/triplette).

    Ritorna un DataFrame con colonne: id, label, disease.
    """
    rng = random.Random(seed)

    healthy_ids = df_split.loc[df_split[label_col] == 0, id_col].astype(str).tolist()
    rng.shuffle(healthy_ids)

    disease_df = df_split.loc[df_split[label_col] == 1].copy()
    disease_df[id_col] = disease_df[id_col].astype(str)
    id_to_disease = dict(zip(disease_df[id_col], disease_df[disease_col]))

    disease_types = sorted(disease_df[disease_col].dropna().unique().tolist())
    ids_by_type = {
        t: disease_df.loc[disease_df[disease_col] == t, id_col].tolist()
        for t in disease_types
    }
    for ids in ids_by_type.values():
        rng.shuffle(ids)

    half = n // 2
    healthy_sample = healthy_ids[:half]

    disease_sample = []
    pointers = {t: 0 for t in disease_types}
    exhausted = set()
    i = 0
    while len(disease_sample) < half and len(exhausted) < len(disease_types):
        t = disease_types[i % len(disease_types)]
        i += 1
        if t in exhausted:
            continue
        p = pointers[t]
        if p < len(ids_by_type[t]):
            disease_sample.append(ids_by_type[t][p])
            pointers[t] = p + 1
        else:
            exhausted.add(t)

    rows = [{"id": _id, "label": 0, "disease": "Healthy"} for _id in healthy_sample]
    rows += [{"id": _id, "label": 1, "disease": id_to_disease.get(_id, "Unknown")}
             for _id in disease_sample]

    result = pd.DataFrame(rows)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    return result


def check_fasta_metadata_alignment(fasta_path, tsv_path, n=5):
    """Controllo di coerenza id TSV <-> id FASTA (sostituisce debug_ids.py).

    Ritorna True se i primi 'n' id del TSV combaciano con i primi 'n' id
    letti dal FASTA nello stesso ordine, False altrimenti.
    """
    df_head = pd.read_csv(tsv_path, sep="\t", nrows=n)
    tsv_ids = df_head["id"].astype(str).tolist()

    fasta_ids = []
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                fasta_ids.append(extract_fasta_id(line))
                if len(fasta_ids) == n:
                    break

    ok = tsv_ids == fasta_ids
    if not ok:
        print("ATTENZIONE: gli id del TSV e del FASTA non combaciano nell'ordine atteso.")
        print("TSV:  ", tsv_ids)
        print("FASTA:", fasta_ids)
    return ok


if __name__ == "__main__":
    # Piccolo self-test manuale (non fa parte della pipeline).
    demo = "ACGTACGTNN"
    enc = one_hot_encode_circular(demo, window_size=16, wrap_len=4)
    print("shape:", enc.shape, "sum per colonna (0 o 1):", enc.sum(axis=0))
