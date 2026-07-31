"""Funzioni condivise dalla pipeline eccDNA.

- read_fasta_stream: lettura streaming del FASTA, RAM-safe
- compute_sequence_descriptors e le funzioni di supporto (composizione/skew,
  CpG e firma dinucleotidica, ripetizioni, entropia): usate da
  descriptor_extractor.py e descriptor_understanding_by_disease.py
"""

import math
from collections import Counter

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
# compositivi (rapporti/frazioni, indipendenti da lunghezza e metodo di
# sequenziamento - vedi i docstring delle singole funzioni), organizzati in
# famiglie biologico/statistiche:
# - composizione/skew: gc_content, gc_skew, at_skew, purine_pyrimidine_skew
# - bias a coppie di basi: cpg_oe, dinuc_signature_dist
# - ripetizioni: tandem_repeat_fraction
# - entropia/complessita': entropy_tri, cond_entropy_1, cond_entropy_2
#
# Questi 10 sono il set FINALE. Il percorso di selezione:
# 1. da 18 candidati validati con descriptor_understanding_by_disease.py
#    (importanza RF + AUC univariata sulle 17 malattie robuste, blocco
#    method+length-matched) sono stati scartati 4 descrittori deboli:
#    g4_fraction (G-quadruplex), gc_window_std, entropy_tri_window_std
#    (eterogeneita' a finestre: AUC alta grezza ma ~0.02 method+length-matched,
#    catturavano la lunghezza) e palindrome_density -> 14 descrittori.
# 2. un'ablation successiva ha rimosso anche termodinamica (nn_stability_mean/std)
#    e periodicita' (periodicity_3bp/10bp): nn_stability_mean era correlato 0.99
#    con gc_content (GC travestito) e togliendo tutte e 4 l'AUC del modello
#    calava solo di ~0.003-0.011 (entro il rumore) -> 10 descrittori, piu'
#    essenziali a parita' di prestazioni. Coerente con "pochi ma forti".
DESCRIPTOR_NAMES = [
    "gc_content", "gc_skew", "at_skew", "purine_pyrimidine_skew",
    "cpg_oe", "dinuc_signature_dist",
    "tandem_repeat_fraction",
    "entropy_tri", "cond_entropy_1", "cond_entropy_2",
]
TANDEM_REPEAT_MAX_UNIT = 6  # lunghezza massima dell'unita' ripetuta cercata (1..6 bp)
TANDEM_REPEAT_MIN_COPIES = 3  # minimo di copie consecutive per contare come ripetizione


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


def compute_sequence_descriptors(sequence):
    """Calcola i 10 descrittori biologici/statistici per una sequenza.

    Ritorna un dict con le chiavi in DESCRIPTOR_NAMES. La sequenza deve
    essere gia' filtrata a monte (niente 'N', lunghezza minima >= 3): qui
    non viene rifatto quel controllo per evitare di duplicare la logica di
    filtro tra descriptor_extractor.py e descriptor_understanding*.py.

    Tutti i valori sono rapporti/frazioni/differenze normalizzate (mai conteggi
    grezzi), cosi' da non dipendere dalla lunghezza assoluta della sequenza ne'
    dal metodo di sequenziamento che l'ha prodotta - vedi
    descriptor_understanding_by_disease.py per la verifica empirica di questa
    proprieta' (AUC length-matched/method-matched/method+length-matched).
    """
    lunghezza = len(sequence)
    base_counts = Counter(sequence)
    di_counts, di_total = _kmer_counts(sequence, 2)
    tri_counts, tri_total = _kmer_counts(sequence, 3)

    h1 = _entropy_bits_from_counts(base_counts, lunghezza)
    h2 = _entropy_bits_from_counts(di_counts, di_total)
    h3 = _entropy_bits_from_counts(tri_counts, tri_total)

    gc_skew, at_skew = gc_at_skew(base_counts)

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
