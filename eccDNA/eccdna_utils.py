"""Funzioni condivise dalla pipeline eccDNA.

- read_fasta_stream: lettura streaming del FASTA, RAM-safe
- compute_sequence_descriptors e le funzioni di supporto (composizione/skew,
  CpG e firma dinucleotidica, ripetizioni, entropia, termodinamica del duplex,
  periodicita'): usate da descriptor_extractor.py e
  descriptor_understanding_by_disease.py
"""

import math
import statistics
from collections import Counter

import numpy as np

BASES = "ACGT"
# lz_complexity (complessita' di Lempel-Ziv) e' stata rimossa: dipende troppo
# dalla metodologia di sequenziamento. Sopra LZ_COMPLEXITY_MAX_LENGTH (ex
# 20000 bp) l'algoritmo O(n^2) veniva saltato e ritornava None, e quella
# soglia di lunghezza correla con 'method' (i protocolli producono frammenti
# di lunghezza tipica diversa) - un canale con cui il confondente rientrava
# dalla porta sul retro anche a valle del campionamento bilanciato per
# metodo. Anche da calcolata, la normalizzazione c(n)*log2(n)/(2n) resta
# distorta su sequenze corte, un'altra via indiretta per cui la lunghezza
# poteva trapelare nel descrittore. Vedi README per il dettaglio.
#
# Al suo posto sono stati reintrodotti/aggiunti descrittori puramente
# compositivi (rapporti/frazioni/medie per passo, indipendenti da lunghezza e
# metodo di sequenziamento - vedi i docstring delle singole funzioni),
# organizzati in famiglie biologico/statistiche:
# - composizione/skew: gc_content, gc_skew, at_skew, purine_pyrimidine_skew
# - bias a coppie di basi: cpg_oe, dinuc_signature_dist
# - ripetizioni: tandem_repeat_fraction
# - entropia/complessita': entropy_tri, cond_entropy_1, cond_entropy_2
# - termodinamica del duplex: nn_stability_mean, nn_stability_std
# - periodicita' di sequenza: periodicity_3bp, periodicity_10bp
#
# Questi 14 sono il set FINALE, selezionato dopo aver validato 18 candidati con
# descriptor_understanding_by_disease.py (importanza RF + AUC univariata,
# aggregate sulle 17 malattie robuste, nel blocco method+length-matched). Sono
# stati scartati 4 descrittori a contributo trascurabile una volta controllati
# i confondenti:
# - g4_fraction (G-quadruplex): 1.4% di importanza, ultimo di tutti - la
#   biologia c'era ma il segnale nei dati no.
# - gc_window_std, entropy_tri_window_std (eterogeneita' interna a finestre):
#   AUC univariata alta sul grezzo (~0.14) ma che CROLLA a ~0.02 sotto il
#   controllo lunghezza+metodo - misurando la varianza tra finestre catturavano
#   in parte la lunghezza (piu' finestre = sequenza piu' lunga), un confondente.
# - palindrome_density: consistentemente nel gruppo debole.
# Delle 3 famiglie nuove aggiunte (termodinamica, periodicita', G-quadruplex),
# le prime due hanno portato segnale reale (nn_stability_mean e' il 4o
# descrittore piu' importante in assoluto), la terza no ed e' stata rimossa -
# esito coerente con l'obiettivo "pochi descrittori ma molto informativi".
DESCRIPTOR_NAMES = [
    "gc_content", "gc_skew", "at_skew", "purine_pyrimidine_skew",
    "cpg_oe", "dinuc_signature_dist",
    "tandem_repeat_fraction",
    "entropy_tri", "cond_entropy_1", "cond_entropy_2",
    "nn_stability_mean", "nn_stability_std",
    "periodicity_3bp", "periodicity_10bp",
]
TANDEM_REPEAT_MAX_UNIT = 6  # lunghezza massima dell'unita' ripetuta cercata (1..6 bp)
TANDEM_REPEAT_MIN_COPIES = 3  # minimo di copie consecutive per contare come ripetizione

# Energia libera nearest-neighbor (ΔG a 37 gradi C, kcal/mol) per ciascuno dei
# 16 passi dinucleotidici - parametri unificati di SantaLucia (1998), lo
# standard per la stabilita' del duplex di DNA. Valori piu' negativi = coppia
# di basi adiacenti che lega piu' forte (es. GC/CG molto stabili, TA/AT deboli).
# La tabella e' simmetrica per complemento inverso (AA=TT, CA=TG, ...): e' una
# proprieta' fisica del passo, non del singolo filamento.
NN_DELTA_G = {
    "AA": -1.00, "AC": -1.44, "AG": -1.28, "AT": -0.88,
    "CA": -1.45, "CC": -1.84, "CG": -2.17, "CT": -1.28,
    "GA": -1.30, "GC": -2.24, "GG": -1.84, "GT": -1.44,
    "TA": -0.58, "TC": -1.30, "TG": -1.45, "TT": -1.00,
}
PERIODICITY_MIN_LENGTH = 40  # sotto questa lunghezza l'autocorrelazione a lag 10 e' troppo rumorosa


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


def gc_content(base_counts, lunghezza):
    """Frazione di basi G+C sul totale: il descrittore compositivo piu'
    elementare, un rapporto puro (non dipende dalla lunghezza assoluta della
    sequenza ne' dal metodo di sequenziamento che l'ha prodotta)."""
    g, c = base_counts.get("G", 0), base_counts.get("C", 0)
    return (g + c) / lunghezza if lunghezza > 0 else 0.0


def gc_at_skew(base_counts):
    """Skew di composizione (G-C)/(G+C) e (A-T)/(A+T) da un Counter di basi.

    Cattura l'asimmetria tra le due basi di ciascuna coppia complementare
    (bias di filamento), un'informazione che gc_content da solo non vede:
    due sequenze con la stessa % di GC possono avere composizione G/C molto
    diversa tra loro.
    """
    g, c = base_counts.get("G", 0), base_counts.get("C", 0)
    a, t = base_counts.get("A", 0), base_counts.get("T", 0)
    gc_skew = (g - c) / (g + c) if (g + c) > 0 else 0.0
    at_skew = (a - t) / (a + t) if (a + t) > 0 else 0.0
    return gc_skew, at_skew


def purine_pyrimidine_skew(base_counts):
    """Skew purine/pirimidine (A+G-C-T)/(A+G+C+T): asimmetria di composizione
    sull'asse purina/pirimidina, complementare a gc_skew/at_skew (che
    guardano l'asse delle coppie complementari). Formulato come skew
    (differenza su somma, in [-1, 1]) invece che come rapporto grezzo per
    restare limitato anche quando pirimidine=0.
    """
    purine = base_counts.get("A", 0) + base_counts.get("G", 0)
    pirimidine = base_counts.get("C", 0) + base_counts.get("T", 0)
    totale = purine + pirimidine
    return (purine - pirimidine) / totale if totale > 0 else 0.0


def cpg_observed_over_expected(sequence, base_counts):
    """Rapporto CpG osservato/atteso: proxy di metilazione/pressione selettiva.

    Numeratore e denominatore scalano entrambi con la lunghezza della
    sequenza (sono frequenze, non conteggi grezzi), quindi il rapporto resta
    indipendente dalla lunghezza assoluta.
    """
    lunghezza = len(sequence)
    freq_c = base_counts.get("C", 0) / lunghezza
    freq_g = base_counts.get("G", 0) / lunghezza
    if freq_c == 0 or freq_g == 0:
        return 0.0

    n_cg = sum(1 for i in range(lunghezza - 1) if sequence[i:i + 2] == "CG")
    freq_cg_osservata = n_cg / (lunghezza - 1)
    return freq_cg_osservata / (freq_c * freq_g)


def dinucleotide_signature_distance(base_counts, di_counts, di_total, lunghezza):
    """'Firma genomica' di Karlin: media di |rho_xy - 1| sui 16 dinucleotidi,
    dove rho_xy = f(xy) / (f(x)*f(y)). Generalizza cpg_oe a tutte le coppie
    di basi: quantifica quanto la composizione a coppie si discosta da
    quella attesa per indipendenza (bias globale, non solo CpG).
    """
    freq_base = {b: base_counts.get(b, 0) / lunghezza for b in BASES}
    scarti = []
    for x in BASES:
        for y in BASES:
            fx, fy = freq_base[x], freq_base[y]
            if fx == 0 or fy == 0:
                continue
            f_xy = di_counts.get(x + y, 0) / di_total
            rho = f_xy / (fx * fy)
            scarti.append(abs(rho - 1))
    return sum(scarti) / len(scarti) if scarti else 0.0


def tandem_repeat_fraction(sequence, max_unit=TANDEM_REPEAT_MAX_UNIT, min_copies=TANDEM_REPEAT_MIN_COPIES):
    """Frazione della sequenza coperta da ripetizioni in tandem semplici
    (unita' di 1..max_unit basi, ripetuta almeno min_copies volte di fila).

    La formazione di eccDNA e' spesso mediata da ripetizioni genomiche
    (dirette o invertite): questo descrittore cattura un aspetto strutturale
    diverso da composizione/entropia - non "quanto e' prevedibile la
    sequenza" ma "quanto e' letteralmente occupata da un motivo ripetuto".
    E' una frazione (copertura/lunghezza totale), non un conteggio grezzo:
    resta confrontabile tra sequenze di lunghezza diversa. Scansione greedy:
    a ogni posizione si cerca la ripetizione piu' lunga (su tutte le
    lunghezze di unita' provate), poi si salta oltre la regione coperta.
    O(n * max_unit).
    """
    n = len(sequence)
    if n == 0:
        return 0.0
    covered = 0
    i = 0
    while i < n:
        miglior_lunghezza = 0
        for unit in range(1, max_unit + 1):
            if i + unit * min_copies > n:
                continue
            u = sequence[i:i + unit]
            copie = 1
            j = i + unit
            while j + unit <= n and sequence[j:j + unit] == u:
                copie += 1
                j += unit
            if copie >= min_copies:
                lunghezza = copie * unit
                if lunghezza > miglior_lunghezza:
                    miglior_lunghezza = lunghezza
        if miglior_lunghezza > 0:
            covered += miglior_lunghezza
            i += miglior_lunghezza
        else:
            i += 1
    return covered / n


def nearest_neighbor_stability(sequence):
    """nn_stability_mean, nn_stability_std: stabilita' termodinamica del duplex
    di DNA dal modello nearest-neighbor (energia libera ΔG, tabella NN_DELTA_G).

    A differenza di gc_content, che guarda solo QUANTE G/C ci sono, questo
    guarda QUALI basi sono adiacenti: la forza con cui i due filamenti legano
    dipende dalla coppia di passi (es. un passo GC lega molto piu' di un passo
    TA anche a parita' di contenuto GC complessivo). E' una proprieta' fisica
    misurata sperimentalmente, non una statistica di conteggio.

    Si scorre la sequenza a coppie consecutive, si somma il ΔG di ogni passo e
    si ritorna (media, deviazione standard) dei ΔG per passo:
    - media: stabilita' complessiva del duplex (piu' negativa = piu' stabile).
      E' una media per passo, quindi indipendente dalla lunghezza.
    - std: quanto la stabilita' e' omogenea o "a chiazze" lungo la molecola,
      un asse di eterogeneita' fisica basato sull'energia di legame e non sulla
      sola composizione delle basi.
    Ritorna (0.0, 0.0) per sequenze troppo corte per almeno un passo.
    """
    valori = [NN_DELTA_G[sequence[i:i + 2]] for i in range(len(sequence) - 1)
              if sequence[i:i + 2] in NN_DELTA_G]
    if not valori:
        return 0.0, 0.0
    media = sum(valori) / len(valori)
    std = statistics.pstdev(valori) if len(valori) >= 2 else 0.0
    return media, std


def _autocorrelation_at_lag(signal, lag):
    """Autocorrelazione (coefficiente di Pearson) di un segnale numerico con
    se stesso spostato di 'lag' posizioni. Misura quanto la sequenza si
    'assomiglia' a distanza fissa: valori alti = periodicita' a quel passo."""
    n = len(signal)
    if n <= lag + 1:
        return 0.0
    x = signal[:n - lag]
    y = signal[lag:]
    mx, my = x.mean(), y.mean()
    denom = math.sqrt(((x - mx) ** 2).sum() * ((y - my) ** 2).sum())
    if denom == 0:
        return 0.0
    return float(((x - mx) * (y - my)).sum() / denom)


def sequence_periodicity(sequence):
    """periodicity_3bp, periodicity_10bp: forza della periodicita' della
    sequenza a passo 3 e a passo 10 basi, via autocorrelazione.

    Biologia: le regioni codificanti hanno una periodicita' di 3 basi (i
    codoni); il DNA avvolto sui nucleosomi mostra una periodicita' di ~10 basi
    nei dinucleotidi A/T. Sono impronte di come la sequenza e' usata nella
    cellula, non catturate da entropia/composizione (che sono 'senza scala').

    Statistica: la sequenza viene codificata come segnale binario W/S
    (A o T = 1, G o C = 0) e se ne calcola l'autocorrelazione a lag 3 e lag 10.
    E' un coefficiente in [-1, 1], indipendente dalla lunghezza. Ritorna
    (0.0, 0.0) per sequenze piu' corte di PERIODICITY_MIN_LENGTH (a lag 10 il
    segnale sarebbe troppo rumoroso per essere affidabile).
    """
    n = len(sequence)
    if n < PERIODICITY_MIN_LENGTH:
        return 0.0, 0.0
    signal = np.fromiter((1.0 if b in "AT" else 0.0 for b in sequence), dtype=np.float64, count=n)
    return _autocorrelation_at_lag(signal, 3), _autocorrelation_at_lag(signal, 10)


def compute_sequence_descriptors(sequence):
    """Calcola i 14 descrittori biologici/statistici per una sequenza.

    Ritorna un dict con le chiavi in DESCRIPTOR_NAMES. La sequenza deve
    essere gia' filtrata a monte (niente 'N', lunghezza minima >= 3): qui
    non viene rifatto quel controllo per evitare di duplicare la logica di
    filtro tra descriptor_extractor.py e descriptor_understanding*.py.

    Tutti i valori sono rapporti/frazioni/differenze normalizzate/medie per
    passo (mai conteggi grezzi), cosi' da non dipendere dalla lunghezza
    assoluta della sequenza ne' dal metodo di sequenziamento che l'ha prodotta
    - vedi descriptor_understanding_by_disease.py per la verifica empirica di
    questa proprieta' (AUC length-matched/method-matched/method+length-matched).
    """
    lunghezza = len(sequence)
    base_counts = Counter(sequence)
    di_counts, di_total = _kmer_counts(sequence, 2)
    tri_counts, tri_total = _kmer_counts(sequence, 3)

    h1 = _entropy_bits_from_counts(base_counts, lunghezza)
    h2 = _entropy_bits_from_counts(di_counts, di_total)
    h3 = _entropy_bits_from_counts(tri_counts, tri_total)

    gc_skew, at_skew = gc_at_skew(base_counts)
    nn_stability_mean, nn_stability_std = nearest_neighbor_stability(sequence)
    periodicity_3bp, periodicity_10bp = sequence_periodicity(sequence)

    return {
        "gc_content": gc_content(base_counts, lunghezza),
        "gc_skew": gc_skew,
        "at_skew": at_skew,
        "purine_pyrimidine_skew": purine_pyrimidine_skew(base_counts),
        "cpg_oe": cpg_observed_over_expected(sequence, base_counts),
        "dinuc_signature_dist": dinucleotide_signature_distance(base_counts, di_counts, di_total, lunghezza),
        "tandem_repeat_fraction": tandem_repeat_fraction(sequence),
        "entropy_tri": h3 / 6,
        "cond_entropy_1": (h2 - h1) / 2,
        "cond_entropy_2": (h3 - h2) / 2,
        "nn_stability_mean": nn_stability_mean,
        "nn_stability_std": nn_stability_std,
        "periodicity_3bp": periodicity_3bp,
        "periodicity_10bp": periodicity_10bp,
    }


if __name__ == "__main__":
    # Piccolo self-test manuale (non fa parte della pipeline): calcola i
    # descrittori su una sequenza demo e verifica che le chiavi combacino.
    demo = "ACGTACGTACGTACGTACGTGGGGCCCCATATATAT" * 3
    d = compute_sequence_descriptors(demo)
    assert sorted(d) == sorted(DESCRIPTOR_NAMES), "chiavi descrittori incoerenti"
    print(f"OK: {len(d)} descrittori calcolati sulla sequenza demo")
    for nome in DESCRIPTOR_NAMES:
        print(f"  {nome:24s} {d[nome]:.4f}")
