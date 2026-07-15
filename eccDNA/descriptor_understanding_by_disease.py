"""Descrittori biologici/statistici PER SOTTOTIPO di malattia, non aggregati.

In descriptor_understanding.py la classifica di importanza su
disease_binary_label (sano vs TUTTI i malati insieme) era piatta: nessun
segnale evidente. Possibile causa: il segnale esiste solo per alcuni
sottotipi specifici e si "annacqua" nella media fatta su malattie molto
diverse tra loro (es. gastric cancer vs lung cancer hanno probabilmente
biologia di sequenza diversa).

Questo script ripete l'analisi separatamente per ciascun sottotipo (sottotipo
vs un pool di sequenze sane costruito su misura per quel sottotipo - vedi
sotto), cosi' da isolare un eventuale segnale specifico che l'analisi
aggregata nasconde. Per ogni sottotipo con abbastanza sequenze stampa: media
+/- std dei descrittori, ROC-AUC di una RandomForest 'spia' su train/test
split, e (in modalita' --diseases) il dettaglio di quale/i descrittore/i
guida/no la separazione.

CAMPIONAMENTO SANO BILANCIATO PER METODO (per sottotipo): una prima versione
di questo script usava UN SOLO pool sano condiviso, campionato a caso, per
tutti i sottotipi. Il problema: 'method' (il metodo di sequenziamento, es.
Circle-seq/ATAC-seq/WGS) e' quasi perfettamente confuso con l'etichetta
malattia per molti sottotipi (es. Fetal Growth Restriction ed Esophageal
Cancer sono ~100% WGS), e un pool sano campionato a caso dal totale (dove
WGS e' ~0.0004%) perdeva quasi certamente i pochi sani con quel metodo anche
quando esistevano in numero non trascurabile altrove nel dataset (es. ci
sono 11.312 sani sequenziati con ATAC-seq su 445.138 sani totali - tanti,
ma invisibili a un campione casuale di 1000).

Ora, per OGNI sottotipo, il pool sano viene costruito su misura:
1. si campiona il malato del sottotipo
2. si guarda la distribuzione di 'method' in quel campione malato
3. per ciascun metodo presente, si prende dal pool sano GLOBALE (non da un
   sottoinsieme pre-campionato) fino a min(quanti malati hanno quel metodo,
   quanti sani con quel metodo esistono in TUTTO il dataset) sani - il piu'
   vicino possibile a un confronto 1:1 per metodo
4. se un metodo e' quasi assente nel sano (es. WGS: solo 2 sequenze sane in
   tutto il dataset), il campione sano per quel metodo restera' onestamente
   piccolo/vuoto: non e' un problema di campionamento, e' un vero limite dei
   dati sorgente, e viene riportato esplicitamente (tabella per metodo).

CONTROLLO CONFONDENTE - LUNGHEZZA: per ogni sottotipo lo script stampa anche
media lunghezza sano vs malato, la correlazione lunghezza<->entropy_tri, e
un AUC "length-matched" (si bilanciano le due classi per bin di lunghezza
quantili prima di rifare train/test+RF) - un'ulteriore verifica anche dopo
il bilanciamento per metodo, perche' la lunghezza puo' variare ancora
all'interno di uno stesso metodo.

CONTROLLO CONFONDENTE - METODO: oltre al campionamento su misura, viene
comunque calcolato un AUC "method-matched" a valle (stessa logica del
length-matched ma per categoria di metodo) come verifica indipendente che
il ricampionamento abbia funzionato.

CONTROLLO DOPPIO - METODO + LUNGHEZZA INSIEME: length-matched e
method-matched, presi separatamente, possono restare alti anche quando il
vero motore e' la lunghezza *all'interno* di uno stesso metodo (caso reale
trovato per Melanoma: method-matched alto, ma lunghezza molto diversa tra
sano e malato anche dentro lo stesso metodo ATAC-seq). L'AUC
"method+length-matched" bilancia sano/malato per la combinazione
(metodo, bin di lunghezza calcolato separatamente per ciascun metodo): e'
il controllo piu' severo, risponde a "il segnale sopravvive controllando
ENTRAMBI i confondenti insieme, non uno alla volta?".

MODALITA' DEEP-DIVE (--diseases): quando si passa un elenco esplicito di
sottotipi, lo script stampa in piu' per ciascuno la classifica di importanza
RandomForest e l'AUC univariata per descrittore (grezza, length-matched,
method-matched).

Script puramente diagnostico: non produce file di output.

Uso:
    python descriptor_understanding_by_disease.py --n-per-class 2000
    python descriptor_understanding_by_disease.py --diseases "Caid syndrome,Melanoma" --n-per-class 2000
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from eccdna_utils import DESCRIPTOR_NAMES, compute_sequence_descriptors, read_fasta_stream

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LENGTH_MATCH_BINS = 10
DOUBLE_MATCH_LENGTH_BINS = 5  # meno bin dello standalone: qui la lunghezza e' gia' stratificata anche per metodo
MATCH_MIN_TOTAL = 40
MIN_SANO_PER_FIT = 20  # sotto questa soglia il campione sano e' troppo piccolo per un RF/AUC attendibile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection.body.fa"),
    )
    parser.add_argument(
        "--metadata-clean-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_metadata_CLEAN.tsv"),
    )
    parser.add_argument(
        "--metadata-raw-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection_metadata.tsv"),
        help="TSV grezzo (con la colonna 'method') usato per il campionamento bilanciato per metodo",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=2000,
        help="sequenze malate campionate per ciascun sottotipo (il sano viene costruito su misura, "
             "vedi docstring); puoi alzarlo liberamente, non e' piu' un pool fisso condiviso",
    )
    parser.add_argument(
        "--min-per-class", type=int, default=100,
        help="sottotipi con meno sequenze MALATE disponibili vengono saltati",
    )
    parser.add_argument(
        "--diseases", type=str, default=None,
        help="lista di sottotipi separati da virgola da analizzare in dettaglio "
             "(es. 'Melanoma,Liver cancer'); se omesso analizza tutti i sottotipi validi",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fit_rf_auc(X, y, seed):
    """Split train/test 70/30 stratificato, allena la RF 'spia', ritorna (auc, rf)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    return auc, rf


def univariate_aucs(X, y):
    """AUC di ciascun descrittore preso da solo (nessuna combinazione con gli altri).

    Lontano da 0.5 in una direzione o nell'altra = descrittore isolato
    informativo; vicino a 0.5 = da solo non discrimina (puo' comunque
    contribuire in combinazione con altri, ma qui isoliamo l'effetto singolo).
    """
    risultato = {}
    for col in X.columns:
        try:
            risultato[col] = roc_auc_score(y, X[col])
        except ValueError:
            risultato[col] = float("nan")
    return pd.Series(risultato).sort_values(key=lambda s: (s - 0.5).abs(), ascending=False)


def print_importance_and_univariate(label, rf, X, y):
    importanza = pd.Series(rf.feature_importances_, index=DESCRIPTOR_NAMES).sort_values(ascending=False)
    auc_uni = univariate_aucs(X, y)
    print(f"    [{label}] importanza RF (multivariata) e AUC univariata per descrittore:")
    for nome in importanza.index:
        print(f"        {nome:24s} importanza={importanza[nome] * 100:5.1f}%   AUC da solo={auc_uni[nome]:.3f}")


def _matched_auc_by_groups(desc_sani, desc_malati, groups_sani, groups_malati, seed):
    """Nucleo comune: bilancia sano/malato per gruppo (bin di lunghezza o categoria
    di metodo) campionando min(n_sano_gruppo, n_malato_gruppo) da ciascun lato,
    poi rifa' train/test+RF sul campione bilanciato. Ritorna
    (auc, n_totale, X_matched, y_matched, rf_matched) oppure
    (None, n_disponibile, None, None, None) se non ci sono abbastanza dati
    (es. nessuna sovrapposizione tra i gruppi delle due classi).
    """
    rng = np.random.RandomState(seed)
    matched_sani_ids, matched_malati_ids = [], []
    for gruppo in set(groups_sani.unique().tolist()) | set(groups_malati.unique().tolist()):
        idx_s = groups_sani[groups_sani == gruppo].index.to_numpy()
        idx_m = groups_malati[groups_malati == gruppo].index.to_numpy()
        n = min(len(idx_s), len(idx_m))
        if n == 0:
            continue
        matched_sani_ids.extend(rng.choice(idx_s, size=n, replace=False))
        matched_malati_ids.extend(rng.choice(idx_m, size=n, replace=False))

    n_totale = len(matched_sani_ids) + len(matched_malati_ids)
    if len(matched_sani_ids) < MATCH_MIN_TOTAL // 2 or len(matched_malati_ids) < MATCH_MIN_TOTAL // 2:
        return None, n_totale, None, None, None

    X = pd.concat([desc_sani.loc[matched_sani_ids], desc_malati.loc[matched_malati_ids]])
    y = [0] * len(matched_sani_ids) + [1] * len(matched_malati_ids)
    auc, rf = fit_rf_auc(X, y, seed)
    return auc, n_totale, X, y, rf


def length_matched_auc(desc_sani, desc_malati, length_sani, length_malati, seed):
    """Bilancia sano/malato per bin di lunghezza (quantili) prima di rifare train/test+RF."""
    lengths_combined = pd.concat([length_sani, length_malati])
    try:
        bins = pd.qcut(lengths_combined, q=LENGTH_MATCH_BINS, duplicates="drop")
    except ValueError:
        return None, 0, None, None, None
    return _matched_auc_by_groups(desc_sani, desc_malati, bins.loc[length_sani.index], bins.loc[length_malati.index], seed)


def method_matched_auc(desc_sani, desc_malati, method_sani, method_malati, seed):
    """Bilancia sano/malato per CATEGORIA di metodo, come ulteriore verifica indipendente
    che il campionamento su misura (FASE 2) abbia gia' risolto il confondente."""
    return _matched_auc_by_groups(desc_sani, desc_malati, method_sani, method_malati, seed)


def method_and_length_matched_auc(desc_sani, desc_malati, method_sani, method_malati,
                                   length_sani, length_malati, seed, n_bins=DOUBLE_MATCH_LENGTH_BINS):
    """Controllo DOPPIO: bilancia sano/malato per la combinazione (metodo, bin di
    lunghezza), non uno alla volta. Length-matched e method-matched, presi
    separatamente, possono restare alti anche quando il segnale e' in realta'
    spiegato dalla lunghezza *all'interno* di uno stesso metodo (es. melanoma:
    method-matched alto ma lunghezza molto diversa tra sano e malato anche
    dentro lo stesso metodo ATAC-seq). Questo test risponde alla domanda:
    il segnale sopravvive controllando ENTRAMBI i confondenti insieme?

    I quantili di lunghezza vengono calcolati SEPARATAMENTE per ciascun
    metodo (non su scala globale), cosi' un metodo con sequenze tipicamente
    piu' corte/lunghe di un altro non finisce tutto nello stesso bin globale.
    """
    gruppi_sani = pd.Series(index=method_sani.index, dtype=object)
    gruppi_malati = pd.Series(index=method_malati.index, dtype=object)

    metodi_comuni = set(method_sani.unique()) | set(method_malati.unique())
    for metodo in metodi_comuni:
        idx_s = method_sani[method_sani == metodo].index
        idx_m = method_malati[method_malati == metodo].index
        lunghezze_metodo = pd.concat([length_sani.loc[idx_s], length_malati.loc[idx_m]])

        n_bins_effettivi = min(n_bins, lunghezze_metodo.nunique())
        if n_bins_effettivi < 2:
            etichette = pd.Series(f"{metodo}__bin0", index=lunghezze_metodo.index)
        else:
            try:
                bins = pd.qcut(lunghezze_metodo, q=n_bins_effettivi, duplicates="drop")
                etichette = metodo + "__" + bins.astype(str)
            except ValueError:
                etichette = pd.Series(f"{metodo}__bin0", index=lunghezze_metodo.index)

        gruppi_sani.loc[idx_s] = etichette.reindex(idx_s)
        gruppi_malati.loc[idx_m] = etichette.reindex(idx_m)

    return _matched_auc_by_groups(desc_sani, desc_malati, gruppi_sani, gruppi_malati, seed)


def load_method_series_full(raw_path, chunk_size=250000):
    """Carica {id: method} per OGNI riga del TSV grezzo (RAM-safe, a blocchi).

    A differenza di una versione precedente che filtrava per un set di id gia'
    scelto, qui serve la mappa COMPLETA: per campionare il sano in modo
    bilanciato per metodo dobbiamo prima sapere quanti sani esistono per
    ciascun metodo in TUTTO il dataset (non solo in un sottoinsieme
    pre-campionato), altrimenti torneremmo al problema originale.
    """
    parti = []
    for chunk in pd.read_csv(raw_path, sep="\t", usecols=["id", "method"], chunksize=chunk_size, low_memory=False):
        chunk["method"] = chunk["method"].fillna("Unknown").astype("category")
        parti.append(chunk)
    full = pd.concat(parti, ignore_index=True)
    full["id"] = full["id"].astype(str)
    return full.set_index("id")["method"]


def build_method_balanced_healthy_sample(malato_sample, df_healthy, seed):
    """Costruisce un pool sano su misura per un sottotipo, bilanciato per METODO
    di sequenziamento rispetto al campione malato di quel sottotipo.

    Per ciascun metodo presente nel malato, prende dal pool sano GLOBALE
    fino a min(quanti malati hanno quel metodo, quanti sani con quel metodo
    esistono in TUTTO il dataset) sani - il piu' vicino possibile a un
    confronto 1:1 per metodo. Se un metodo e' quasi assente nel sano (es.
    WGS), il numero ottenuto rispecchia onestamente quel limite: non e' un
    problema di campionamento, e' un vero buco nei dati sorgente.

    Ritorna (healthy_ids: set, report_df: DataFrame per-metodo).
    """
    metodo_malato_counts = malato_sample["method"].value_counts()

    righe_report = []
    healthy_ids = set()
    for metodo, n_malato_metodo in metodo_malato_counts.items():
        pool_metodo = df_healthy[df_healthy["method"] == metodo]
        disponibili = len(pool_metodo)
        n_prendere = min(int(n_malato_metodo), disponibili)
        if n_prendere > 0:
            campione = pool_metodo.sample(n=n_prendere, random_state=seed)
            healthy_ids.update(campione["id"].tolist())
        righe_report.append({
            "metodo": metodo,
            "malato": int(n_malato_metodo),
            "sano_disponibili_nel_dataset": disponibili,
            "sano_campionati": n_prendere,
        })

    report_df = pd.DataFrame(righe_report).sort_values("malato", ascending=False)
    return healthy_ids, report_df


def main():
    args = parse_args()
    seed = args.seed
    detailed = args.diseases is not None

    print("--- FASE 1: CONTEGGIO SOTTOTIPI DI MALATTIA ---")
    df_clean = pd.read_csv(args.metadata_clean_path, sep="\t", usecols=["id", "disease_binary_label", "disease"])
    df_clean["id"] = df_clean["id"].astype(str)

    print("Caricamento del metodo di sequenziamento per TUTTO il dataset "
          "(necessario per campionare il sano in modo equo per ciascun sottotipo)...")
    method_series = load_method_series_full(args.metadata_raw_path)
    df_clean["method"] = df_clean["id"].map(method_series).astype(object).fillna("Unknown")

    df_healthy = df_clean[df_clean["disease_binary_label"] == 0]
    df_disease = df_clean[df_clean["disease_binary_label"] == 1]
    print(f"Sani totali nel dataset: {len(df_healthy)}. Distribuzione per metodo:")
    print(df_healthy["method"].value_counts().to_string())

    conteggi = df_disease["disease"].value_counts()
    print()
    print(conteggi.to_string())

    sottotipi_validi = conteggi[conteggi >= args.min_per_class].index.tolist()
    if args.diseases is not None:
        richiesti = [d.strip() for d in args.diseases.split(",")]
        non_trovati = [d for d in richiesti if d not in conteggi.index]
        if non_trovati:
            print(f"\nATTENZIONE: non trovati nel dataset: {non_trovati}")
        sottotipi_validi = [d for d in richiesti if d in sottotipi_validi]
    print(f"\nSottotipi analizzabili (>= {args.min_per_class} sequenze malate disponibili): "
          f"{len(sottotipi_validi)}/{len(conteggi)}")

    print("\n--- FASE 2: CAMPIONAMENTO SU MISURA PER SOTTOTIPO (malato + sano bilanciato per metodo) ---")
    subtype_malato_ids = {}
    subtype_healthy_ids = {}
    all_wanted_ids = set()

    for subtype in sottotipi_validi:
        pool = df_disease[df_disease["disease"] == subtype]
        malato_sample = pool.sample(n=min(args.n_per_class, len(pool)), random_state=seed)
        subtype_malato_ids[subtype] = set(malato_sample["id"])

        healthy_ids_subtype, report_df = build_method_balanced_healthy_sample(malato_sample, df_healthy, seed)
        subtype_healthy_ids[subtype] = healthy_ids_subtype

        print(f"\n> {subtype}: malato={len(malato_sample)}, sano campionato su misura={len(healthy_ids_subtype)}")
        print(report_df.to_string(index=False))

        all_wanted_ids |= subtype_malato_ids[subtype]
        all_wanted_ids |= healthy_ids_subtype

    print(f"\nTotale sequenze uniche da leggere dal FASTA (tutti i sottotipi + i loro sani su misura): "
          f"{len(all_wanted_ids)}")

    print("\n--- FASE 3: CALCOLO DESCRITTORI + LUNGHEZZA (singola scansione del FASTA, in memoria) ---")
    righe = []
    for seq_id, sequence in read_fasta_stream(args.fasta_path, wanted_ids=all_wanted_ids):
        if "N" not in sequence and len(sequence) >= 3:
            descrittori = compute_sequence_descriptors(sequence)
            descrittori["id"] = seq_id
            descrittori["length"] = len(sequence)  # SOLO diagnostico: non e' un feature del modello
            righe.append(descrittori)
    df_desc = pd.DataFrame(righe).set_index("id")
    df_desc = df_desc.dropna(subset=DESCRIPTOR_NAMES)  # lz_complexity e' None sulle sequenze troppo lunghe
    df_desc["method"] = df_desc.index.map(method_series).astype(object).fillna("Unknown")
    print(f"Descrittori calcolati per {len(df_desc)}/{len(all_wanted_ids)} sequenze richieste "
          f"(esclude 'N'/lunghezza insufficiente e sequenze troppo lunghe per lz_complexity).")

    print("\n--- FASE 4: ANALISI PER SOTTOTIPO (media +/- std, lunghezza/metodo, AUC grezzo vs matched) ---")
    risultati = []

    for subtype in sottotipi_validi:
        ids_malati = [i for i in subtype_malato_ids[subtype] if i in df_desc.index]
        ids_sani = [i for i in subtype_healthy_ids[subtype] if i in df_desc.index]

        if len(ids_malati) < args.min_per_class:
            print(f"\n> {subtype}: saltato (malati insufficienti dopo il filtro FASTA).")
            continue
        if len(ids_sani) < MIN_SANO_PER_FIT:
            print(f"\n> {subtype}: NON ANALIZZABILE - solo {len(ids_sani)} sequenze sane con metodo compatibile "
                  f"esistono in tutto il dataset (soglia minima: {MIN_SANO_PER_FIT}). Non e' un limite del "
                  f"campionamento: quel metodo di sequenziamento e' quasi assente nel sano a monte, nei dati grezzi.")
            continue

        X_malati = df_desc.loc[ids_malati, DESCRIPTOR_NAMES]
        X_sani = df_desc.loc[ids_sani, DESCRIPTOR_NAMES]
        len_malati = df_desc.loc[ids_malati, "length"]
        len_sani = df_desc.loc[ids_sani, "length"]
        method_malati = df_desc.loc[ids_malati, "method"]
        method_sani = df_desc.loc[ids_sani, "method"]
        X = pd.concat([X_sani, X_malati])
        y = [0] * len(X_sani) + [1] * len(X_malati)

        print(f"\n> {subtype} (malato={len(X_malati)}, sano={len(X_sani)}):")
        for nome in DESCRIPTOR_NAMES:
            m_s, s_s = X_sani[nome].mean(), X_sani[nome].std()
            m_m, s_m = X_malati[nome].mean(), X_malati[nome].std()
            print(f"    {nome:24s} sano={m_s:.4f}+/-{s_s:.4f}   malato={m_m:.4f}+/-{s_m:.4f}")

        length_diff = len_malati.mean() - len_sani.mean()
        length_combined = pd.concat([len_sani, len_malati])
        entropy_tri_combined = pd.concat([X_sani["entropy_tri"], X_malati["entropy_tri"]])
        corr_len_entropy_tri = length_combined.corr(entropy_tri_combined)
        print(f"    lunghezza      sano={len_sani.mean():.1f}+/-{len_sani.std():.1f}   "
              f"malato={len_malati.mean():.1f}+/-{len_malati.std():.1f}   (diff={length_diff:+.1f})")
        print(f"    corr(lunghezza, entropy_tri) sul campione di questo confronto: {corr_len_entropy_tri:+.3f}")

        metodi_malato_top = method_malati.value_counts(normalize=True).head(3)
        metodi_sano_top = method_sani.value_counts(normalize=True).head(3)
        metodi_comuni = set(method_malati.unique()) & set(method_sani.unique())
        overlap_malato = method_malati.isin(metodi_comuni).mean()
        print(f"    metodo malato (top): {dict((metodi_malato_top * 100).round(1))}")
        print(f"    metodo sano   (top): {dict((metodi_sano_top * 100).round(1))}")
        print(f"    frazione di 'malato' con un metodo presente anche in 'sano': {overlap_malato * 100:.1f}%")

        auc, rf = fit_rf_auc(X, y, seed)
        auc_matched, n_matched, X_matched, y_matched, rf_matched = length_matched_auc(
            X_sani, X_malati, len_sani, len_malati, seed
        )
        auc_matched_str = f"{auc_matched:.3f} (n={n_matched})" if auc_matched is not None else f"N/A (n={n_matched} insufficiente)"

        auc_method, n_method, X_method, y_method, rf_method = method_matched_auc(
            X_sani, X_malati, method_sani, method_malati, seed
        )
        auc_method_str = f"{auc_method:.3f} (n={n_method})" if auc_method is not None else f"N/A (n={n_method}, sovrapposizione insufficiente)"

        auc_double, n_double, X_double, y_double, rf_double = method_and_length_matched_auc(
            X_sani, X_malati, method_sani, method_malati, len_sani, len_malati, seed
        )
        auc_double_str = f"{auc_double:.3f} (n={n_double})" if auc_double is not None else f"N/A (n={n_double}, sovrapposizione insufficiente)"

        print(f"    ROC-AUC grezzo (test 30%): {auc:.3f}   |   length-matched: {auc_matched_str}   |   "
              f"method-matched: {auc_method_str}   |   method+length-matched: {auc_double_str}")

        if detailed:
            print_importance_and_univariate("GREZZO", rf, X, y)
            if rf_matched is not None:
                print_importance_and_univariate("LENGTH-MATCHED", rf_matched, X_matched, y_matched)
            else:
                print("    [LENGTH-MATCHED] non disponibile (dati insufficienti per il match).")
            if rf_method is not None:
                print_importance_and_univariate("METHOD-MATCHED", rf_method, X_method, y_method)
            else:
                print("    [METHOD-MATCHED] non disponibile (sovrapposizione di metodo insufficiente).")
            if rf_double is not None:
                print_importance_and_univariate("METHOD+LENGTH-MATCHED", rf_double, X_double, y_double)
            else:
                print("    [METHOD+LENGTH-MATCHED] non disponibile (sovrapposizione insufficiente).")

        risultati.append({
            "disease": subtype, "n_malati": len(X_malati), "n_sani": len(X_sani),
            "auc_grezzo": auc, "auc_length_matched": auc_matched, "auc_method_matched": auc_method,
            "auc_method_length_matched": auc_double,
            "method_overlap_pct": overlap_malato * 100,
            "length_diff": length_diff, "corr_len_entropy_tri": corr_len_entropy_tri,
        })

    print("\n--- FASE 5: RIEPILOGO - SOTTOTIPI ORDINATI PER ROC-AUC GREZZO ---")
    if risultati:
        df_riepilogo = pd.DataFrame(risultati).sort_values("auc_grezzo", ascending=False)
        print(df_riepilogo.to_string(index=False))
        print("\nCon il campionamento su misura, 'auc_grezzo' e' gia' costruito per essere il piu' possibile "
              "bilanciato per metodo di sequenziamento (non solo per lunghezza). 'auc_length_matched' e "
              "'auc_method_matched' restano comunque utili come verifica indipendente a valle. "
              "'auc_method_length_matched' e' il controllo piu' severo: bilancia ENTRAMBI i confondenti insieme "
              "(per la stessa combinazione metodo+bin di lunghezza) - se resta alto anche qui, il segnale non e' "
              "spiegabile ne' dal metodo ne' dalla lunghezza, nemmeno combinati.")
    else:
        print("Nessun sottotipo con dati sufficienti.")

    print("\n--- ANALISI COMPLETATA ---")


if __name__ == "__main__":
    main()
