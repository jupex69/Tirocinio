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

**15 descrittori**, definiti in `eccdna_utils.compute_sequence_descriptors`,
divisi in 4 famiglie:
- **Composizione**: `gc_skew`, `at_skew`, `cpg_oe`, `purine_pyrimidine_ratio`, `dinuc_signature_dist`
- **Disordine statistico**: `entropy_mono/di/tri`, `cond_entropy_1/2`, `lz_complexity`
- **Eterogeneità interna** (sequenze "a mosaico"): `gc_window_std`, `entropy_tri_window_std`
- **Ripetizioni**: `tandem_repeat_fraction`, `palindrome_density`

**Due confondenti nascosti nei metadati**, entrambi da controllare prima di
fidarsi di un qualunque segnale sano/malato:
- `length` (già noto, per questo escluso da `dataCleaning.py`)
- `method`/`source_db`/`library_type` (il protocollo di sequenziamento,
  scoperto in questa ricerca — es. alcune malattie sono ~100% WGS mentre il
  pool sano è quasi 0% WGS: senza controllo, un modello imparerebbe il
  protocollo, non la malattia)

`descriptor_understanding_by_disease.py` costruisce il pool sano su misura
per ciascun sottotipo, bilanciato per metodo di sequenziamento (non un
pool condiviso campionato a caso), e verifica comunque a valle con un AUC
"length-matched" e "method-matched".

**Malattie con segnale confermato robusto** (sopravvive ai controlli su
lunghezza e metodo): systemic lupus erythematosus, chronic kidney disease,
gastric/colorectal/breast cancer, primary pulmonary hypertension, cataract,
glioblastoma cancer, dilated cardiomyopathy. Altre (es. fetal growth
restriction, esophageal cancer) risultano al momento indeterminabili: il
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
