# eccDNA — ricerca sui descrittori biologici/statistici

Prima di classificare le sequenze di eccDNA (extrachromosomal circular DNA)
come sane o associate a malattia, il tutor ha chiesto di individuare quali
descrittori biologici e statistici (composizione, entropia/disordine,
complessità, ripetizioni) si possono estrarre dalla sequenza per allenare
una rete neurale. Questo repo, nella sua forma attuale, è la pipeline che
risponde a quella domanda: dai metadati grezzi alla scoperta, sottotipo di
malattia per sottotipo, di quali descrittori portano segnale reale e non
un artefatto dei dati.

Per la descrizione del dataset grezzo (colonne, split, note sui valori
mancanti) vedi `../README_disease_detection.md`, scritto dal tutor.

## 1. Setup ambiente

```powershell
conda activate ecc_Dna_hotspot
pip install -r requirements.txt
```

`requirements.txt` contiene solo `pandas`, `numpy`, `scikit-learn`: tutto
cio' che serve alla pipeline attiva. Per rilanciare gli script archiviati in
`vecchi/` (vedi sezione 5) serve installare a parte anche `torch`,
`matplotlib` e `tqdm`.

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
| 3 | `python descriptor_understanding_by_disease.py --diseases "..."` | Solo diagnostica: per i sottotipi indicati, calcola i 15 descrittori su un campione e verifica quali sono davvero informativi, controllando i confondenti lunghezza e metodo di sequenziamento (non genera file) |
| 4 | `python descriptor_extractor.py` | `data/processed/eccdna_descriptor_features.tsv` (i 15 descrittori estratti su tutto il dataset — lanciare solo dopo aver deciso quali descrittori tenere con il passo 3) |

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

## 5. Script archiviati (`vecchi/`, fuori da git)

La cartella `vecchi/` (accanto a questo repo, ignorata da git) contiene la
pipeline di classificazione esplorata prima di questa ricerca sui
descrittori — non più collegata al lavoro attuale, tenuta solo per
riferimento: `model.py` (CNN), `train_classifier.py`,
`build_classification_dataset.py`, `baseline_kmer_model.py`,
`kmer_extractor.py`, `triplet_generator.py`, `descriptor_understanding.py`
(la prima versione, aggregata su tutte le malattie insieme — superata da
`descriptor_understanding_by_disease.py` perché l'aggregazione annacqua il
segnale specifico di ogni sottotipo).

## 6. File principali

| File | Ruolo |
|---|---|
| `dataUnderstanding.py` | EDA + rilevamento data leakage sui metadati grezzi |
| `dataCleaning.py` | Pulizia metadati grezzi → `eccdna_metadata_CLEAN.tsv` |
| `eccdna_utils.py` | Funzioni condivise: parsing FASTA, calcolo dei 15 descrittori |
| `descriptor_understanding_by_disease.py` | Scoperta dei descrittori per sottotipo, con controllo dei confondenti lunghezza/metodo |
| `descriptor_extractor.py` | Estrazione dei descrittori su tutto il dataset (stadio di produzione) |
