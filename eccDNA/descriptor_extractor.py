"""Estrazione su larga scala dei descrittori biologici/statistici eccDNA.

Stadio di PRODUZIONE della pipeline: calcola i 6 descrittori definiti in
eccdna_utils.compute_sequence_descriptors (DESCRIPTOR_NAMES) e li scrive su
TSV, con lettura FASTA streaming (RAM-safe) e calcolo parallelizzato su piu'
processi (il calcolo per sequenza e' CPU-bound puro Python - non vettorizzabile,
lz_complexity in particolare e' O(n^2) nel caso peggiore).

Da lanciare SOLO dopo aver stabilito, con descriptor_understanding_by_disease.py,
quali descrittori sono davvero informativi e quali sottotipi hanno un segnale
robusto (confermato contro i due confondenti length/method).

SELEZIONE MIRATA (non tutto il dataset): una prima versione estraeva i
descrittori per OGNI sequenza valida del dataset (~3,75M) - ma la stragrande
maggioranza appartiene a malattie che non useremo mai (es. Gastric cancer da
sola e' il 68% del dataset) o non ha superato i controlli di confondimento.
Ora --diseases limita l'estrazione alle sole malattie indicate: per ciascuna,
si campiona il malato (fino a --max-malato-per-disease, tutto se ce n'e' meno)
e si costruisce un pool sano su misura bilanciato per metodo di sequenziamento
(stessa logica, riusata da import, di descriptor_understanding_by_disease.py:
build_method_balanced_healthy_sample/load_method_series_full). Il default sono
le 12 malattie con segnale robusto confermato anche dopo il controllo
combinato lunghezza+metodo (vedi README).

ABBINAMENTO MALATTIA<->SANO (--pairing-output-path): il pool sano NON e'
disgiunto tra malattie - una stessa sequenza sana campionata con Circle_seq
puo' comparire nel sano sia di Gastric cancer sia di Cataract. --output-path
contiene i descrittori una sola volta per id (deduplicati), ma per allenare
un classificatore PER MALATTIA serve sapere quali sane sono state abbinate
proprio a quella malattia - prendere "tutto il sano del file" contro il
malato di una sola malattia reintrodurrebbe il confondente di metodo gia'
controllato (es. il sano di Lung cancer e' ATAC-seq, ma la maggioranza del
sano nel file e' Circle_seq/Circle-seq, campionato per Gastric/Colorectal
cancer). eccdna_disease_pairing.tsv risolve questo: una riga per (disease,
id, role), da filtrare per malattia prima di unire i descrittori. Usare
--pairing-only per rigenerare solo questo file (es. se serve ispezionarlo)
senza rifare l'estrazione, gia' completata, dei descrittori.

RIPRESA: se --output-path esiste gia', gli id gia' calcolati vengono letti e
saltati invece di essere ricalcolati da zero. Se l'ultima riga del file
precedente risulta troncata (scrittura interrotta a meta'), viene scartata e
il file troncato a quel punto, cosi' quella sequenza viene ricalcolata per
intero (nessuna riga corrotta nel file finale). Usare --no-resume per forzare
una riestrazione completa da zero (necessario se cambia l'elenco --diseases:
la ripresa saprebbe solo saltare id gia' presenti, non ri-filtrare quelli
ormai fuori target).

NOTA: la lunghezza della sequenza NON viene inclusa tra i descrittori.
E' stata deliberatamente esclusa da dataCleaning.py perche' identificata
come feature leaking (vedi dataUnderstanding.py) e non va reintrodotta senza
rifare quell'analisi.

Uso:
    python descriptor_extractor.py
    python descriptor_extractor.py --diseases "Gastric cancer,Melanoma" --max-malato-per-disease 5000
    python descriptor_extractor.py --no-resume
"""

import argparse
import multiprocessing as mp
import os

from eccdna_utils import DESCRIPTOR_NAMES, compute_sequence_descriptors, read_fasta_stream
from descriptor_understanding_by_disease import build_method_balanced_healthy_sample, load_method_series_full
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WRITE_BUFFER_SIZE = 5000
CHUNK_SIZE = 500  # sequenze per task inviato a un worker: ammortizza l'overhead di IPC tra processi
HEADER = ["id"] + DESCRIPTOR_NAMES

# Le 12 malattie con segnale robusto confermato anche dopo il controllo
# combinato lunghezza+metodo (vedi descriptor_understanding_by_disease.py e README).
DEFAULT_DISEASES = [
    "Gastric cancer", "Colorectal cancer", "Stomach cancer", "Chronic kidney disease",
    "Primary pulmonary hypertension", "Cataract", "Dilated cardiomyopathy",
    "Glioblastoma cancer", "Coronary artery disease", "Caid syndrome",
    "Lung cancer", "Systemic lupus erythematosus",
]
DEFAULT_MAX_MALATO_PER_DISEASE = 20000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection.body.fa"),
        help="percorso del FASTA con le sequenze eccDNA",
    )
    parser.add_argument(
        "--metadata-clean-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_metadata_CLEAN.tsv"),
        help="percorso del TSV pulito (disease/disease_binary_label)",
    )
    parser.add_argument(
        "--metadata-raw-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_detection_metadata.tsv"),
        help="TSV grezzo (con la colonna 'method') per il campionamento sano bilanciato per metodo",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_descriptor_features.tsv"),
        help="percorso del TSV di output",
    )
    parser.add_argument(
        "--pairing-output-path",
        default=os.path.join(SCRIPT_DIR, "data/processed/eccdna_disease_pairing.tsv"),
        help="TSV che registra, per ciascuna malattia, quali id sono il suo malato e quali il suo "
             "sano abbinato - necessario perche' il pool sano e' condiviso tra malattie in --output-path",
    )
    parser.add_argument(
        "--pairing-only", action="store_true",
        help="calcola e salva solo il file di abbinamento (FASE 1), senza rifare l'estrazione dei descrittori",
    )
    parser.add_argument(
        "--diseases", type=str, default=",".join(DEFAULT_DISEASES),
        help="lista di sottotipi malati separati da virgola da includere nell'estrazione",
    )
    parser.add_argument(
        "--max-malato-per-disease", type=int, default=DEFAULT_MAX_MALATO_PER_DISEASE,
        help="tetto massimo di sequenze malate campionate per ciascuna malattia (tutte se ce n'e' meno)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-workers", type=int, default=max(1, os.cpu_count() or 1),
        help="numero di processi paralleli per il calcolo dei descrittori (default: tutti i core disponibili)",
    )
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="salta gli id gia' presenti in --output-path invece di ricalcolarli (default)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="ignora un eventuale output precedente e riestrae tutto da zero",
    )
    return parser.parse_args()


def select_target_ids(metadata_clean_path, metadata_raw_path, diseases, max_malato_per_disease, seed):
    """Sceglie gli id da estrarre: solo le malattie indicate (malato, con
    tetto) + un pool sano su misura bilanciato per metodo per ciascuna.

    Riusa la stessa logica di descriptor_understanding_by_disease.py invece
    di duplicarla: e' la stessa identica strategia di campionamento gia'
    validata li', qui applicata su un numero di sequenze piu' grande (fino
    al tetto scelto) per costruire il dataset di produzione.

    Ritorna (id_scelti, pairing_df). pairing_df ha una riga per (malattia, id,
    ruolo): serve perche' il pool sano NON e' disgiunto tra malattie (una
    stessa sequenza sana puo' essere stata campionata per piu' malattie con lo
    stesso metodo) - senza questa mappa, un training che prendesse "tutto il
    sano del file" contro il malato di UNA sola malattia perderebbe il
    bilanciamento per metodo e reintrodurrebbe il confondente gia' controllato.
    """
    df_clean = pd.read_csv(metadata_clean_path, sep="\t", usecols=["id", "disease_binary_label", "disease"])
    df_clean["id"] = df_clean["id"].astype(str)

    print("Caricamento del metodo di sequenziamento per tutto il dataset "
          "(necessario per il campionamento sano bilanciato)...")
    method_series = load_method_series_full(metadata_raw_path)
    df_clean["method"] = df_clean["id"].map(method_series).astype(object).fillna("Unknown")

    df_healthy = df_clean[df_clean["disease_binary_label"] == 0]

    id_scelti = set()
    righe_pairing = []
    for malattia in diseases:
        df_malato = df_clean[df_clean["disease"] == malattia]
        if df_malato.empty:
            print(f"  ATTENZIONE: nessuna sequenza trovata per '{malattia}', la salto.")
            continue

        n_prendere = min(len(df_malato), max_malato_per_disease)
        malato_sample = df_malato.sample(n=n_prendere, random_state=seed)
        id_scelti.update(malato_sample["id"].tolist())
        righe_pairing.extend(
            {"disease": malattia, "id": _id, "role": "malato"} for _id in malato_sample["id"]
        )

        healthy_ids, report_df = build_method_balanced_healthy_sample(malato_sample, df_healthy, seed)
        id_scelti.update(healthy_ids)
        righe_pairing.extend(
            {"disease": malattia, "id": _id, "role": "sano"} for _id in healthy_ids
        )

        print(f"\n> {malattia}: malato={len(malato_sample)} (su {len(df_malato)} disponibili), "
              f"sano campionato su misura={len(healthy_ids)}")
        print(report_df.to_string(index=False))

    pairing_df = pd.DataFrame(righe_pairing, columns=["disease", "id", "role"])
    return id_scelti, pairing_df


def _format_row(seq_id, descrittori):
    valori = [str(descrittori[nome]) if descrittori[nome] is not None else "nan" for nome in DESCRIPTOR_NAMES]
    return "\t".join([seq_id] + valori)


def _process_chunk(chunk):
    """Calcola i descrittori per un blocco di (id, sequenza). Eseguita nei processi worker."""
    righe = []
    for seq_id, sequence in chunk:
        if "N" not in sequence and len(sequence) >= 3:
            descrittori = compute_sequence_descriptors(sequence)
            righe.append(_format_row(seq_id, descrittori))
    return righe


def _chunked(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _resume_from_previous_output(output_path):
    """Legge gli id gia' calcolati in un output precedente, troncando via
    un'eventuale ultima riga incompleta (scrittura interrotta a meta').

    Ritorna il set di id gia' calcolati. Il file viene troncato subito dopo
    l'ultima riga valida, cosi' le nuove righe si possono accodare in sicurezza.
    """
    n_colonne_attese = len(HEADER)
    gia_fatti = set()
    with open(output_path, "rb") as f:
        f.readline()  # header
        bytes_validi = f.tell()
        while True:
            riga = f.readline()
            if not riga:
                break
            if riga.endswith(b"\n") and len(riga.split(b"\t")) == n_colonne_attese:
                gia_fatti.add(riga.split(b"\t", 1)[0].decode())
                bytes_validi = f.tell()
            else:
                break  # riga troncata/incompleta: si scarta, si tronca qui e si ricalcola
    with open(output_path, "r+b") as f:
        f.truncate(bytes_validi)
    return gia_fatti


def main():
    args = parse_args()
    diseases = [d.strip() for d in args.diseases.split(",") if d.strip()]

    print("--- FASE 1: SELEZIONE MIRATA DEGLI ID (malato per malattia + sano bilanciato per metodo) ---")
    print(f"Malattie incluse ({len(diseases)}): {', '.join(diseases)}")
    id_validi, pairing_df = select_target_ids(
        args.metadata_clean_path, args.metadata_raw_path, diseases, args.max_malato_per_disease, args.seed,
    )
    print(f"\nTotale id da estrarre (unione malato+sano su tutte le malattie): {len(id_validi)}")

    pairing_df.to_csv(args.pairing_output_path, sep="\t", index=False)
    print(f"Abbinamento malattia<->id (malato/sano) salvato in: {args.pairing_output_path} "
          f"({len(pairing_df)} righe)")

    if args.pairing_only:
        print("\n--pairing-only: salto l'estrazione dei descrittori.")
        return

    scrivi_header = True
    if args.resume and os.path.exists(args.output_path):
        print("Ripresa: leggo gli id gia' presenti nell'output precedente...")
        gia_fatti = _resume_from_previous_output(args.output_path)
        id_validi -= gia_fatti
        scrivi_header = False
        print(f"{len(gia_fatti)} sequenze gia' calcolate in precedenza (saltate). "
              f"Restano {len(id_validi)} da calcolare.")

    print(f"\n--- FASE 2: ESTRAZIONE DESCRITTORI DAL FASTA ({args.n_workers} processi paralleli) ---")

    sequenze_salvate = 0
    ultima_stampa = 0
    buffer = []

    def sequence_generator():
        for seq_id, sequence in read_fasta_stream(args.fasta_path, wanted_ids=id_validi):
            yield seq_id, sequence

    with open(args.output_path, "a" if not scrivi_header else "w") as f_out:
        if scrivi_header:
            f_out.write("\t".join(HEADER) + "\n")

        with mp.Pool(processes=args.n_workers) as pool:
            for righe_chunk in pool.imap_unordered(_process_chunk, _chunked(sequence_generator(), CHUNK_SIZE)):
                buffer.extend(righe_chunk)
                sequenze_salvate += len(righe_chunk)

                if len(buffer) >= WRITE_BUFFER_SIZE:
                    f_out.write("\n".join(buffer) + "\n")
                    buffer.clear()

                if sequenze_salvate - ultima_stampa >= 50000:
                    ultima_stampa = sequenze_salvate
                    print(f"  Salvate {sequenze_salvate} sequenze finora...")

        if buffer:
            f_out.write("\n".join(buffer) + "\n")

    print("\n--- FASE 3: COMPLETAMENTO ---")
    print(f"Estrazione completata! {len(DESCRIPTOR_NAMES)} descrittori salvati in: {args.output_path}")


if __name__ == "__main__":
    main()
