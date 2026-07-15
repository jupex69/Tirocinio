# eccDNA — ricerca sui descrittori biologici/statistici

L'obiettivo è individuare quali descrittori biologici e statistici
(composizione, entropia/disordine, complessità, ripetizioni) si possono
estrarre dalla sequenza di eccDNA (extrachromosomal circular DNA) per
allenare una rete neurale che la classifichi come sana o associata a
malattia. Questo repo, nella sua forma attuale, è la pipeline che risponde
a quella domanda: dai metadati grezzi alla scoperta, sottotipo di malattia
per sottotipo, di quali descrittori portano segnale reale e non un
artefatto dei dati.

Per la descrizione del dataset grezzo (colonne, split, note sui valori
mancanti) vedi `../README_disease_detection.md`, scritto dal tutor.

## 1. Setup ambiente

```powershell
conda activate ecc_Dna_hotspot
pip install -r requirements.txt
```

`requirements.txt` contiene solo `pandas`, `numpy`, `scikit-learn`: tutto
cio' che serve alla pipeline attiva.

## 2. Dati necessari (non su git)

I file grezzi sono troppo grandi per essere versionati (vedi `.gitignore`).
Vanno messi in `data/processed/` prima di eseguire qualunque script:

```text
data/processed/eccdna_disease_detection_metadata.tsv
data/processed/eccdna_disease_detection.body.fa
```

`eccdna_metadata_CLEAN.tsv` ed `eccdna_descriptor_features.tsv` vengono
generati dagli script sotto.

## 3. Pipeline, in ordine

| # | Comando | Cosa produce |
|---|---|---|
| 1 | `python dataUnderstanding.py` | Solo diagnostica: EDA + rilevamento data leakage sui metadati grezzi (non genera file) |
| 2 | `python dataCleaning.py` | `data/processed/eccdna_metadata_CLEAN.tsv` (metadati puliti, senza le colonne che causano leakage) |
| 3 | `python descriptor_understanding_by_disease.py --diseases "..."` | Solo diagnostica: per i sottotipi indicati, calcola i descrittori su un campione e verifica quali sono davvero informativi, controllando i confondenti lunghezza e metodo di sequenziamento (non genera file) |
| 4 | `python descriptor_extractor.py` | `data/processed/eccdna_descriptor_features.tsv` (i 6 descrittori estratti solo per le malattie con segnale confermato, non su tutto il dataset — vedi sezione 4) + `eccdna_disease_pairing.tsv` (abbinamento malattia↔id, necessario perché il pool sano è condiviso tra malattie) |

Il passo 3 è quello su cui si è concentrata la ricerca finora: senza
argomenti analizza tutti i sottotipi con almeno `--min-per-class` sequenze;
con `--diseases "MalattiaA,MalattiaB"` fa un'analisi di dettaglio (importanza
RandomForest + AUC univariata per ciascun descrittore) solo sui sottotipi
indicati.

## 4. Cosa abbiamo scoperto finora

**6 descrittori**, definiti in `eccdna_utils.compute_sequence_descriptors`
(ridotti da un set iniziale di 15 dopo l'analisi per sottotipo — le altre 9,
di composizione e ripetizioni, davano un contributo trascurabile), divisi
in 2 famiglie:
- **Disordine/complessità**: `entropy_tri`, `cond_entropy_1`, `cond_entropy_2`, `lz_complexity`
- **Eterogeneità interna** (sequenze "a mosaico"): `gc_window_std`, `entropy_tri_window_std`

**Due confondenti nascosti nei metadati**, entrambi da controllare prima di
fidarsi di un qualunque segnale sano/malato:
- `length` (già noto, per questo escluso da `dataCleaning.py`)
- `method`/`source_db`/`library_type` (il protocollo di sequenziamento,
  scoperto in questa ricerca — es. alcune malattie sono ~100% WGS mentre il
  pool sano è quasi 0% WGS: senza controllo, un modello imparerebbe il
  protocollo, non la malattia). Confermato anche da `dataUnderstanding.py`:
  nel modello spia RandomForest, `method` è il secondo confondente più forte
  in assoluto (14.9% di importanza, dietro solo `tissue` al 61.3%).

`descriptor_understanding_by_disease.py` costruisce il pool sano su misura
per ciascun sottotipo, bilanciato per metodo di sequenziamento (non un
pool condiviso campionato a caso), e verifica comunque a valle con un AUC
"length-matched", "method-matched" e "method+length-matched" (il controllo
più severo: bilancia entrambi i confondenti insieme).

**Il quadro completo, sulle 67 malattie del dataset grezzo**: 41 hanno
abbastanza sequenze malate da testare, 31 hanno anche un sano compatibile
per metodo di sequenziamento, e di queste **17 mostrano un segnale
biologico robusto** che sopravvive al controllo combinato lunghezza+metodo
(le altre 14 sembravano promettenti grezze ma erano quasi interamente
spiegate da length/method — es. Melanoma: AUC 0.83 grezzo → 0.57 dopo il
controllo doppio, un artefatto). Delle 17 robuste, i 6 descrittori scelti
ne classificano bene **12** (es. gastric/colorectal cancer, chronic kidney
disease, primary pulmonary hypertension, dilated cardiomyopathy, systemic
lupus erythematosus), 2 in modo più debole (colorectal adenoma, stomach) e
3 restano poco solide per limiti dei dati sorgente (campione piccolo, o
confronto contro linee cellulari isolate/tessuto sano non pertinente
all'organo malato, es. hypopharynx cancer). Le rimanenti malattie (es.
fetal growth restriction, esophageal cancer) risultano indeterminabili: il
sottotipo usa un metodo di sequenziamento quasi assente nel pool sano
dell'intero dataset (es. solo 2 sequenze sane con WGS su 445.138 sane
totali) — non un limite dello script, un limite dei dati sorgente.

## 5. File principali

| File | Ruolo |
|---|---|
| `dataUnderstanding.py` | EDA + rilevamento data leakage sui metadati grezzi |
| `dataCleaning.py` | Pulizia metadati grezzi → `eccdna_metadata_CLEAN.tsv` |
| `eccdna_utils.py` | Funzioni condivise: parsing FASTA, calcolo dei 6 descrittori |
| `descriptor_understanding_by_disease.py` | Scoperta dei descrittori per sottotipo, con controllo dei confondenti lunghezza/metodo |
| `descriptor_extractor.py` | Estrazione dei descrittori per le malattie con segnale confermato (stadio di produzione) |

## 6. Bibliografia

Paper di riferimento consultati per la scelta dei descrittori e per il
contesto sull'eccDNA (i PDF restano solo in locale in `Paper/`, non su git —
vedi `.gitignore`):

- Wang C. et al., "DeepECC: a deep learning framework for genome-wide
  identification and analysis of human cancer eccDNAs", *Nucleic Acids
  Research*, 2026. DOI: [10.1093/nar/gkag198](https://doi.org/10.1093/nar/gkag198).
  Fonte della convenzione "basi non standard -> colonna zero" usata in
  `eccdna_utils.one_hot_encode_circular`.
- Chang K.-L. et al., "Short human eccDNAs are predictable from sequences"
  (DeepCircle), *Briefings in Bioinformatics* 24(3), 2023.
  DOI: [10.1093/bib/bbad147](https://doi.org/10.1093/bib/bbad147). Prima
  evidenza che le eccDNA corte sono predicibili dalla sola sequenza, pur
  con origine genomica quasi casuale — la premessa di fondo di questa ricerca.
- Li J., Liu Z., Zhang Z., "eccDNAMamba: A Pre-Trained Model for Ultra-Long
  eccDNA Sequence Analysis", arXiv:2506.18940, 2025.
  DOI: [10.48550/arXiv.2506.18940](https://doi.org/10.48550/arXiv.2506.18940).
  Fonte dell'augmentation a "tiling circolare" usata in
  `eccdna_utils.one_hot_encode_circular`.
- Fang J. et al., "Detection of primary cancer types via fragment size
  selection in circulating cell-free extrachromosomal circular DNA",
  *Genome Medicine* 18:18, 2026.
  DOI: [10.1186/s13073-025-01595-6](https://doi.org/10.1186/s13073-025-01595-6).
  Conferma indipendente che la lunghezza del frammento di eccDNA porta
  segnale biologico reale — coerente con perché va comunque trattata come
  confondente da controllare, non da ignorare, nella nostra analisi.
- "Application of Deep Learning in the Identification of Extrachromosomal
  Circular DNA (eccDNA)", ACM, 2025.
  DOI: [10.1145/3757110.3757175](https://doi.org/10.1145/3757110.3757175).
