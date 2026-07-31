# eccDNA — descrittori di sequenza e classificazione sano/malato

L'obiettivo è capire se una sequenza di eccDNA (extrachromosomal circular DNA)
possa essere classificata come **sana o associata a malattia** a partire solo da
descrittori biologici/statistici estraibili dalla sequenza stessa — senza usare
come feature confondenti tecnici (lunghezza, metodo di sequenziamento) e
indipendentemente dalla malattia specifica. La pipeline va dai metadati grezzi
alla scoperta, malattia per malattia, di quali descrittori portano segnale reale,
fino al confronto di sei modelli di classificazione.

Per la descrizione del dataset grezzo (colonne, split, note sui valori mancanti)
vedi `../README_disease_detection.md`, scritto dal tutor.

## 1. Setup ambiente

```powershell
conda activate ecc_Dna_hotspot
pip install -r requirements.txt
# torch va installato dall'indice CPU dedicato (niente CUDA su questa macchina):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `torch` (CPU).

## 2. Dati necessari (non su git)

I file grezzi sono troppo grandi per essere versionati (vedi `.gitignore`).
Vanno messi in `data/processed/` prima di eseguire qualunque script:

```text
data/processed/eccdna_disease_detection_metadata.tsv
data/processed/eccdna_disease_detection.body.fa
```

`eccdna_metadata_CLEAN.tsv`, `eccdna_descriptor_features.tsv`,
`eccdna_disease_pairing.tsv` e i TSV dei risultati vengono generati dagli script.

## 3. Pipeline, in ordine

**Flusso dei dati** (dove e quando la sequenza diventa 10 numeri):

```text
SEQUENZE GREZZE (lettere ACGT)
  eccdna_disease_detection.body.fa
        │
        ▼   descriptor_extractor.py  →  chiama  eccdna_utils.compute_sequence_descriptors()
CALCOLO DEI 10 DESCRITTORI          (qui, UNA VOLTA SOLA, prima di ogni addestramento)
        │
        ▼
NUMERI GIA' PRONTI SU FILE
  eccdna_descriptor_features.tsv   (una riga per sequenza: id + 10 numeri)
        │
        ▼   training_data.build_balanced_splits()
DATASET BILANCIATO  =  10 colonne X  +  y (0/1)
        │
        ▼   train_models.py
ADDESTRAMENTO E CONFRONTO DEI MODELLI
  (il modello legge solo i numeri, mai le lettere ACGT)
```

Da qui in poi la sequenza originale non serve più: i modelli lavorano solo sui
10 numeri gia' calcolati e salvati.

| # | Comando | Cosa produce |
|---|---|---|
| 1 | `python dataUnderstanding.py` | Solo diagnostica: EDA + rilevamento data leakage sui metadati grezzi (non genera file) |
| 2 | `python dataCleaning.py` | `eccdna_metadata_CLEAN.tsv` (metadati puliti, senza le colonne che causano leakage) |
| 3 | `python descriptor_understanding_by_disease.py --diseases "..."` | Solo diagnostica: per i sottotipi indicati, calcola i descrittori su un campione e verifica quali sono informativi controllando i confondenti lunghezza e metodo (importanza RandomForest + AUC univariata per descrittore); non genera file |
| 4 | `python descriptor_extractor.py` | `eccdna_descriptor_features.tsv` (i 10 descrittori estratti per le 17 malattie con segnale confermato) + `eccdna_disease_pairing.tsv` (abbinamento malattia↔id, necessario perché il pool sano è condiviso tra malattie) |
| 5 | `python train_models.py` | Costruisce il dataset bilanciato (`training_data.py`), allena e confronta i 6 modelli (`models_pytorch.py`), salva `model_comparison_results.tsv` e `model_comparison_per_disease.tsv` |

## 4. I 10 descrittori

Definiti in `eccdna_utils.compute_sequence_descriptors`. Sono tutti rapporti o
frazioni (mai conteggi grezzi), quindi indipendenti per costruzione dalla
lunghezza assoluta. Quattro famiglie:

- **Composizione / skew**: `gc_content`, `gc_skew`, `at_skew`, `purine_pyrimidine_skew`
- **Bias a coppie di basi**: `cpg_oe`, `dinuc_signature_dist`
- **Ripetizioni**: `tandem_repeat_fraction`
- **Entropia / complessità**: `entropy_tri`, `cond_entropy_1`, `cond_entropy_2`

**Percorso di selezione** (da 18 candidati a 10):
1. Da 18 candidati, il controllo combinato lunghezza+metodo scarta 4 descrittori
   deboli: `lz_complexity` (dipendeva dal metodo via cutoff di lunghezza),
   `gc_window_std` e `entropy_tri_window_std` (AUC univariata alta grezza ma ~0.02
   method+length-matched: catturavano la lunghezza), `palindrome_density` e
   `g4_fraction` (G-quadruplex) → **14 descrittori**.
2. Un'ablation rimuove anche **termodinamica** (`nn_stability_mean/std`) e
   **periodicità** (`periodicity_3bp/10bp`): `nn_stability_mean` era correlato
   **0.99 con `gc_content`** (contenuto di GC travestito), e togliendo tutte e 4
   l'AUC del modello calava solo di ~0.004–0.011 (entro il rumore) → **10
   descrittori**, più essenziali a parità di prestazioni.

**Due confondenti** da controllare prima di fidarsi di qualunque segnale:
- `length` (già noto, per questo escluso da `dataCleaning.py`)
- `method` / `source_db` / `library_type` (il protocollo di sequenziamento: alcune
  malattie sono ~100% WGS mentre il pool sano è quasi 0% WGS; senza controllo un
  modello imparerebbe il protocollo, non la malattia).

`descriptor_understanding_by_disease.py` costruisce il pool sano su misura per
ciascun sottotipo, bilanciato per metodo, e verifica a valle con AUC
"length-matched", "method-matched" e "method+length-matched" (il più severo).
Sulle 67 malattie del dataset, 41 hanno abbastanza malati da testare, 31 hanno un
sano compatibile per metodo, e **17 mostrano un segnale robusto** che sopravvive
al controllo combinato (le altre sono quasi interamente spiegate da length/method
— es. Melanoma: AUC 0.88 grezzo → 0.58 dopo il controllo doppio, un artefatto).

## 5. Dataset bilanciato per il training

`training_data.build_balanced_splits` ricostruisce train/val/test dal file di
abbinamento in modo **equo e method-safe**:
- **50/50 sano/malato** esatto, per ogni malattia (non un 50/50 globale casuale);
- **method-matching preservato**: per ogni malattia i sani hanno la stessa
  distribuzione di metodo dei malati (entro ~1%), perché bilanciare "a caso"
  reintrodurrebbe il confondente (tipo-malattia e metodo sono correlati);
- **dominio dei tumori GI attenuato** con un tetto di 3.000 sequenze per malattia.

Risultato: train 62.090, val 21.712, test 10.696 sequenze, tutti 50/50, su 17
malattie. Split per `split_cluster` (cluster genomico), non casuale, per non
sovrastimare le performance. Standardizzazione con media/std calcolate solo sul
train. La colonna `disease` non è mai una feature del modello.

## 6. Risultati dei modelli

Sei modelli (2 ML classico, 4 reti neurali PyTorch) sul test set bilanciato
(10.696 sequenze), ordinati per ROC-AUC:

| Modello | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| MLP con attenzione | **0.779** | 0.706 | 0.675 | 0.795 | 0.730 |
| MLP profondo | 0.778 | 0.704 | 0.669 | 0.809 | 0.732 |
| Siamese (loss combinata) | 0.776 | 0.706 | 0.680 | 0.777 | 0.725 |
| Gradient Boosting | 0.769 | 0.699 | 0.667 | 0.794 | 0.725 |
| Random Forest | 0.758 | 0.680 | 0.632 | 0.859 | 0.728 |
| Siamese (metric learning) | 0.545 | 0.527 | 0.517 | 0.824 | 0.636 |

- I 4 approcci discriminativi (2 MLP, GBM, RF) più la siamese a loss combinata si
  raggruppano entro ~2 punti di AUC: con poche feature tabellari la complessità del
  modello conta poco, il tetto lo danno i descrittori.
- La siamese a **metric learning puro** fallisce (0.545, quasi caso): il paradigma
  contrastivo non si adatta a poche feature scalari con classi molto sovrapposte.
  La variante a loss combinata arriva a ~0.78 solo perché, col peso della triplet
  ridotto a 0.05, è di fatto un MLP.
- L'AUC ~0.78 è più bassa di un ipotetico ~0.9 perché è **onesta**: sopravvive al
  controllo dei confondenti e allo split per cluster genomico.

Il riepilogo completo è anche nella docstring di `train_models.py`.

## 7. File principali

| File | Ruolo |
|---|---|
| `dataUnderstanding.py` | EDA + rilevamento data leakage sui metadati grezzi |
| `dataCleaning.py` | Pulizia metadati grezzi → `eccdna_metadata_CLEAN.tsv` |
| `eccdna_utils.py` | Funzioni condivise: parsing FASTA, calcolo dei 10 descrittori |
| `descriptor_understanding_by_disease.py` | Scoperta dei descrittori per sottotipo, con controllo dei confondenti lunghezza/metodo |
| `descriptor_extractor.py` | Estrazione dei 10 descrittori per le malattie con segnale confermato (produzione) |
| `training_data.py` | Assemblaggio del dataset bilanciato (50/50, method-matched per malattia) |
| `models_pytorch.py` | Architetture PyTorch: MLP, MLP con attenzione, siamese (metric learning e loss combinata) |
| `train_models.py` | Allena e confronta i 6 modelli, salva i risultati |

## 8. Bibliografia

Paper di riferimento consultati per la scelta dei descrittori e per il contesto
sull'eccDNA (i PDF restano solo in locale in `Paper/`, non su git):

- Wang C. et al., "DeepECC: a deep learning framework for genome-wide
  identification and analysis of human cancer eccDNAs", *Nucleic Acids Research*,
  2026. DOI: [10.1093/nar/gkag198](https://doi.org/10.1093/nar/gkag198).
- Chang K.-L. et al., "Short human eccDNAs are predictable from sequences"
  (DeepCircle), *Briefings in Bioinformatics* 24(3), 2023.
  DOI: [10.1093/bib/bbad147](https://doi.org/10.1093/bib/bbad147). Prima evidenza
  che le eccDNA corte sono predicibili dalla sola sequenza — la premessa di fondo
  di questa ricerca.
- Li J., Liu Z., Zhang Z., "eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA
  Sequence Analysis", arXiv:2506.18940, 2025.
  DOI: [10.48550/arXiv.2506.18940](https://doi.org/10.48550/arXiv.2506.18940).
- Fang J. et al., "Detection of primary cancer types via fragment size selection
  in circulating cell-free extrachromosomal circular DNA", *Genome Medicine*
  18:18, 2026. DOI: [10.1186/s13073-025-01595-6](https://doi.org/10.1186/s13073-025-01595-6).
  Conferma indipendente che la lunghezza del frammento porta segnale biologico
  reale — coerente con il trattarla come confondente da controllare, non da ignorare.
- "Application of Deep Learning in the Identification of Extrachromosomal Circular
  DNA (eccDNA)", ACM, 2025. DOI: [10.1145/3757110.3757175](https://doi.org/10.1145/3757110.3757175).
