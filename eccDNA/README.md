# eccDNA — classificazione sano / associato a malattia

Pipeline per classificare frammenti di eccDNA (extrachromosomal circular DNA)
come `healthy` o `disease_associated`, a partire dalla sola sequenza
genomica. Ispirata a DeepCircle, ECCNET, eccDNAMamba, DeepECC e ScanTecc
(vedi `../Paper/`).

Per la descrizione del dataset grezzo (colonne, split, note sui valori
mancanti) vedi `../README_disease_detection.md`, scritto dal tutor.

## 1. Setup ambiente

```powershell
conda activate ecc_Dna_hotspot
pip install -r requirements.txt
```

Nessuna GPU richiesta (`torch` è la build CPU): tutto lo script di training
gira su CPU in tempi ragionevoli grazie a un dataset campionato e bilanciato
(non l'intero dataset da 3,7M sequenze).

## 2. Dati necessari (non su git)

I file grezzi sono troppo grandi per essere versionati (vedi `.gitignore`).
Vanno messi in `data/processed/` prima di eseguire qualunque script:

```text
data/processed/eccdna_disease_detection_metadata.tsv
data/processed/eccdna_disease_detection.body.fa
```

Tutto il resto (`eccdna_metadata_CLEAN.tsv`, i file dei k-meri, i dataset
di classificazione, i modelli allenati) viene generato dagli script sotto.

## 3. Pipeline, in ordine

| # | Comando | Cosa produce |
|---|---|---|
| 1 | `python dataUnderstanding.py` | Solo diagnostica: EDA + rilevamento data leakage (facoltativo, non genera file) |
| 2 | `python dataCleaning.py` | `data/processed/eccdna_metadata_CLEAN.tsv` (metadati puliti, senza le colonne che causano leakage) |
| 3 | `python kmer_extractor.py --k 3` | `data/processed/eccdna_kmer_3_features.tsv` (frequenze di k-meri, usate solo dalla baseline) |
| 4 | `python build_classification_dataset.py` | `data/classification/{train,val,test}_ids.csv` (id bilanciati sano/malato, uniformi tra i tipi di malattia) |
| 5a | `python train_classifier.py` | CNN sulla sequenza one-hot con augmentation circolare → `data/models/` (checkpoint, metriche, grafico) |
| 5b | `python baseline_kmer_model.py` | Logistic Regression + Random Forest sui k-meri → `data/models_baseline/metrics.json` |

Il passo 3 può essere rifatto anche con `--k 4` (256 feature invece di 64);
`baseline_kmer_model.py` accetta `--kmer-features-path` per usare quel file
al posto del default k=3.

Gli step 5a e 5b sono indipendenti tra loro e possono girare in qualunque
ordine (o in parallelo, se la macchina ha abbastanza core).

## 4. Via secondaria (non attiva): triplet loss

`triplet_generator.py` genera triplette anchor/positive/negative bilanciate per un
eventuale futuro approccio a embedding con triplet loss / rete siamese.
Non è la via principale  e non è
consumato da nessun modello attuale: resta nel repo come asset già generato
(`data/triplets/*.csv`, già committati) per un possibile sviluppo futuro.

## 5. File principali

| File | Ruolo |
|---|---|
| `dataUnderstanding.py` | EDA + rilevamento data leakage sui metadati grezzi |
| `dataCleaning.py` | Pulizia metadati grezzi → `eccdna_metadata_CLEAN.tsv` |
| `kmer_extractor.py` | Estrazione frequenze di k-meri (`--k 3` o `--k 4`) |
| `eccdna_utils.py` | Funzioni condivise: parsing FASTA, codifica one-hot circolare, campionamento bilanciato |
| `build_classification_dataset.py` | Costruisce il dataset bilanciato train/val/test per il classificatore diretto |
| `model.py` | Architettura della CNN 1D (`EccDNACNN`) |
| `train_classifier.py` | Training + valutazione della CNN |
| `baseline_kmer_model.py` | Baseline Logistic Regression / Random Forest sui k-meri |
| `triplet_generator.py` | Via secondaria (triplet loss), non attiva |

## 6. Risultati (ultimo run)

Dataset bilanciato: 30.000 train / 6.000 val / 6.000 test, 50/50 sano/malato,
66 tipi di malattia diversi mescolati (non un singolo tipo di cancro, a
differenza dei paper di riferimento).

| Modello | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| CNN (sequenza, `data/models/metrics.json`) | 0.546 | 0.531 | 0.793 | 0.636 | 0.572 |
| Logistic Regression (k-meri) | 0.542 | 0.536 | 0.627 | 0.578 | 0.562 |
| Random Forest (k-meri, `data/models_baseline/metrics.json`) | 0.662 | 0.653 | 0.692 | 0.672 | 0.712 |

Il Random Forest sui k-meri è al momento il modello migliore.
