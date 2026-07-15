"""Funzioni condivise dalla pipeline eccDNA.

Attivamente usate oggi (ricerca sui descrittori biologici/statistici):
- read_fasta_stream / load_sequences: lettura streaming del FASTA, RAM-safe
- compute_sequence_descriptors e le funzioni di supporto (entropia,
  complessita' di Lempel-Ziv, eterogeneita' interna a finestre): usate da
  descriptor_extractor.py e descriptor_understanding_by_disease.py
- check_fasta_metadata_alignment: controllo di coerenza id TSV <-> FASTA

Funzioni storiche (usate solo dagli script archiviati in vecchi/, tenute
qui per riferimento/compatibilita' ma non toccate dalla pipeline attuale):
- one_hot_encode_circular: codifica per il classificatore CNN (model.py,
  train_classifier.py)
- sample_balanced_ids: campionamento bilanciato sano/malato usato da
  build_classification_dataset.py

NOTA: triplet_generator.py (anch'esso in vecchi/) non usava queste funzioni
di proposito, per non cambiare la logica random che aveva generato i
triplet CSV.
"""

import math
import random
import statistics
from collections import Counter

import numpy as np
import pandas as pd

BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
# Ridotto da 15 a questi 6 dopo l'analisi per sottotipo in
# descriptor_understanding_by_disease.py: sono la famiglia entropia/complessita'
# (entropy_tri, cond_entropy_1, cond_entropy_2, lz_complexity) piu' quella di
# eterogeneita' interna (gc_window_std, entropy_tri_window_std) - insieme
# classificano bene 12 delle 17 malattie con segnale robusto confermato,
# contro un contributo trascurabile delle altre 9 (composizione/ripetizioni).
DESCRIPTOR_NAMES = [
    "entropy_tri", "cond_entropy_1", "cond_entropy_2",
    "lz_complexity", "gc_window_std", "entropy_tri_window_std",
]
LZ_COMPLEXITY_MAX_LENGTH = 20000  # oltre questa soglia lz76_complexity (O(n^2) nel caso peggiore) viene saltata
WINDOWED_HETEROGENEITY_WINDOW = 150  # dimensione delle finestre non sovrapposte per gc/entropy_tri_window_std
WINDOWED_HETEROGENEITY_MIN_WINDOWS = 3  # sotto questa soglia la sequenza e' troppo corta per misurare eterogeneita'


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


def _kmer_counts(sequence, k):
    """Counter dei k-meri sovrapposti + il loro totale (denominatore per le frequenze)."""
    kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers), len(kmers)


def _entropy_bits_from_counts(counts, total):
    """Entropia di Shannon in bit (NON normalizzata) da un Counter di conteggi gia' fatto."""
    entropia = 0.0
    for count in counts.values():
        p = count / total
        entropia -= p * math.log2(p)
    return entropia


def lz76_complexity(sequence):
    """Complessita' di Lempel-Ziv (algoritmo di Kaspar & Schuster, 1987).

    Conta il numero di sottostringhe distinte prodotte dalla scansione
    incrementale della sequenza: una sequenza ripetitiva produce poche
    sottostringhe (c basso), una casuale ne produce molte (c alto). E' una
    misura di "disordine" complementare all'entropia: l'entropia guarda solo
    le frequenze dei k-meri, la complessita' di LZ cattura anche ripetizioni
    e struttura a lungo raggio che l'entropia puo' non vedere (es. un
    tandem repeat di un motivo che compare comunque raramente in frequenza).
    """
    n = len(sequence)
    if n < 2:
        return 1
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    while True:
        if sequence[i + k - 1] == sequence[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


def lz_complexity_normalized(sequence):
    """Complessita' di LZ normalizzata: c(n) * log2(n) / (2n).

    Tende a 1 per una sequenza casuale su alfabeto di 4 simboli, si abbassa
    verso 0 per una sequenza ripetitiva - stessa convenzione (0=ripetitivo,
    1=disordine massimo) usata per le entropie, cosi' e' confrontabile.
    Ritorna None per sequenze piu' lunghe di LZ_COMPLEXITY_MAX_LENGTH:
    l'algoritmo e' O(n^2) nel caso peggiore e diventerebbe troppo costoso
    su sequenze molto lunghe estratte su milioni di record.
    """
    n = len(sequence)
    if n > LZ_COMPLEXITY_MAX_LENGTH:
        return None
    if n < 4:
        return 0.0
    c = lz76_complexity(sequence)
    return c * math.log2(n) / (2 * n)


def windowed_heterogeneity(sequence, window_size=WINDOWED_HETEROGENEITY_WINDOW,
                            min_windows=WINDOWED_HETEROGENEITY_MIN_WINDOWS):
    """gc_window_std, entropy_tri_window_std: quanto composizione e complessita'
    variano tra finestre non sovrapposte della stessa sequenza.

    Tutti gli altri descrittori sono medie globali sull'intera sequenza e non
    vedono l'eterogeneita' INTERNA: una sequenza "a mosaico" (piu' regioni
    genomiche di origine diversa fuse insieme - un fenomeno di eccDNA
    "complesso" documentato in letteratura per alcuni tumori) puo' avere la
    stessa media di una sequenza omogenea ma varianza locale molto diversa.
    Ritorna (0.0, 0.0) se la sequenza e' troppo corta per almeno min_windows
    finestre (eterogeneita' non misurabile in modo affidabile).
    """
    n = len(sequence)
    n_windows = n // window_size
    if n_windows < min_windows:
        return 0.0, 0.0

    gc_vals, entropy_vals = [], []
    for i in range(n_windows):
        window = sequence[i * window_size:(i + 1) * window_size]
        g, c = window.count("G"), window.count("C")
        gc_vals.append((g + c) / len(window))
        tri_counts, tri_total = _kmer_counts(window, 3)
        if tri_total > 0:
            entropy_vals.append(_entropy_bits_from_counts(tri_counts, tri_total) / 6)

    gc_std = statistics.pstdev(gc_vals) if len(gc_vals) >= 2 else 0.0
    entropy_std = statistics.pstdev(entropy_vals) if len(entropy_vals) >= 2 else 0.0
    return gc_std, entropy_std


def compute_sequence_descriptors(sequence):
    """Calcola i 6 descrittori biologici/statistici per una sequenza.

    Ritorna un dict con le chiavi in DESCRIPTOR_NAMES. La sequenza deve
    essere gia' filtrata a monte (niente 'N', lunghezza minima >= 3): qui
    non viene rifatto quel controllo per evitare di duplicare la logica di
    filtro tra descriptor_extractor.py e descriptor_understanding*.py.

    lz_complexity puo' essere None per sequenze molto lunghe (vedi
    lz_complexity_normalized) - va gestito a valle (es. dropna prima di
    allenare un modello).
    """
    lunghezza = len(sequence)
    base_counts = Counter(sequence)
    di_counts, di_total = _kmer_counts(sequence, 2)
    tri_counts, tri_total = _kmer_counts(sequence, 3)

    h1 = _entropy_bits_from_counts(base_counts, lunghezza)
    h2 = _entropy_bits_from_counts(di_counts, di_total)
    h3 = _entropy_bits_from_counts(tri_counts, tri_total)

    gc_window_std, entropy_tri_window_std = windowed_heterogeneity(sequence)

    return {
        "entropy_tri": h3 / 6,
        "cond_entropy_1": (h2 - h1) / 2,
        "cond_entropy_2": (h3 - h2) / 2,
        "lz_complexity": lz_complexity_normalized(sequence),
        "gc_window_std": gc_window_std,
        "entropy_tri_window_std": entropy_tri_window_std,
    }


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
