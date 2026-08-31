# eccDNA — identificare la malattia dalla sequenza

L'obiettivo è capire se, a partire **esclusivamente dalla sequenza** di una
molecola di eccDNA (extrachromosomal circular DNA), sia possibile identificare la
malattia. La domanda è affrontata in due fasi:

1. **Classificazione binaria** sano/malato — con dieci descrittori biologici della
   sequenza, scelti per fondatezza biologica e validati contro i confondenti;
2. **Classificazione multiclasse** del tipo di malattia (tessuto), condotta sotto
   controllo severo dei confondenti.

Il filo conduttore metodologico è il **controllo dei confondenti** (lunghezza,
metodo di sequenziamento, studio di origine): nel dataset grezzo l'etichetta di
malattia è quasi allineata a queste variabili tecniche, e separare le malattie
senza controlli significa in gran parte riconoscere il *batch*, non la biologia.

Il documento di tesi (`tesi_descrittori_validazione.tex` → PDF, ignorato da git)
raccoglie l'intero lavoro: abstract → descrittori → confondenti → binario →
multiclasse → conclusioni.

Per la descrizione del dataset grezzo (colonne, split, valori mancanti) vedi
`../README_disease_detection.md`, scritto dal tutor.

## 1. Setup ambiente

```powershell
conda activate ecc_Dna_hotspot
pip install -r requirements.txt
# torch dall'indice CPU (niente CUDA su questa macchina):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `scipy`, `torch` (CPU).

## 2. Dati necessari (non su git)

I file grezzi sono troppo grandi per essere versionati. Vanno messi in
`data/processed/` prima di eseguire qualunque script:

```text
data/processed/eccdna_disease_detection_metadata.tsv   (metadati completi, ~1.3 GB)
data/processed/eccdna_disease_detection.body.fa        (sequenze, ~6.5 GB)
```

`eccdna_metadata_CLEAN.tsv`, `eccdna_descriptor_features.tsv`,
`eccdna_disease_pairing.tsv` e i TSV dei risultati vengono generati dagli script.

## 3. Flusso dei dati (dove la sequenza diventa numeri)

```text
SEQUENZE GREZZE (lettere ACGT)  →  eccdna_disease_detection.body.fa
        │  descriptor_extractor.py → eccdna_utils.compute_sequence_descriptors()
        ▼
10 DESCRITTORI su file  →  eccdna_descriptor_features.tsv  (id + 10 numeri per sequenza)
        │  training_data.build_balanced_splits()   /   gold_standard_data (multiclasse)
        ▼
DATASET (X = numeri, y = etichetta)  →  train_models.py (binario) / experiment_goldstandard.py (multiclasse)
```

Il modello legge solo i numeri, mai le lettere ACGT. La malattia specifica non è
mai una feature: è mascherata in addestramento e usata solo per l'analisi.

## 4. I 10 descrittori

Definiti in `eccdna_utils.compute_sequence_descriptors`. Tutti rapporti o frazioni
(mai conteggi grezzi), quindi indipendenti per costruzione dalla lunghezza. Quattro
famiglie:

- **Composizione / skew**: `gc_content`, `gc_skew`, `at_skew`, `purine_pyrimidine_skew`
- **Bias a coppie di basi**: `cpg_oe`, `dinuc_signature_dist`
- **Ripetizioni**: `tandem_repeat_fraction`
- **Entropia / complessità**: `entropy_tri`, `cond_entropy_1`, `cond_entropy_2`

**Selezione (da 18 candidati a 10):** il controllo combinato lunghezza+metodo
scarta i descrittori deboli o confondenti (es. `lz_complexity`, `g4_fraction`,
`palindrome_density`, le eterogeneità a finestre); una successiva ablation rimuove
termodinamica e periodicità (`nn_stability_mean` correlava 0.99 con `gc_content`).
Restano i 10 essenziali, a parità di prestazioni.

## 5. Il controllo dei confondenti

È il fondamento di tutto il lavoro. Due script diagnosticano il dataset completo:

| Comando | Cosa mostra |
|---|---|
| `python diagnose_dataset.py` | Diagnostica completa sui ~3.75 M record: valori mancanti, bilanciamento (73 etichette, il tumore gastrico è ~77% dei malati, rapporto max/min > 10⁶), **purezza per malattia** (metodo ~0.91, database ~0.96, tessuto ~0.86 → la malattia è quasi un alias del protocollo/studio), confondente lunghezza, verifica leakage (assente) |
| `python explore_full_dataset.py` | Profilo per malattia: metodo/tessuto/studio dominante e relativa purezza; salva `full_dataset_disease_profile.tsv` |

A livello di **descrittore**, `descriptor_understanding_by_disease.py` costruisce
per ogni sottotipo un pool sano appaiato per metodo e verifica il segnale con AUC
"length-matched", "method-matched" e "method+length-matched" (il più severo): solo
i descrittori (e le malattie) il cui segnale sopravvive vengono tenuti.

## 6. Task binario: sano/malato

**Dataset** (`training_data.build_balanced_splits`): 50/50 sano/malato per ogni
malattia, con i sani appaiati per metodo ai malati (bilanciare "a caso"
reintrodurrebbe il confondente), tetto di 3.000 sequenze per malattia. Split per
cluster genomico (non casuale). Test: 10.696 sequenze, 17 malattie robuste.

**Risultati** (`train_models.py`, test bilanciato, ordinati per ROC-AUC):

| Modello | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| MLP con attenzione | **0.779** | 0.706 | 0.675 | 0.795 | 0.730 |
| MLP profondo | 0.778 | 0.704 | 0.669 | 0.809 | 0.732 |
| Siamese (loss combinata) | 0.776 | 0.706 | 0.680 | 0.777 | 0.725 |
| Gradient Boosting | 0.769 | 0.699 | 0.667 | 0.794 | 0.725 |
| Random Forest | 0.758 | 0.680 | 0.632 | 0.859 | 0.728 |
| Siamese (metric learning) | 0.545 | 0.527 | 0.517 | 0.824 | 0.636 |

Con poche feature scalari la complessità del modello conta poco: il tetto lo danno
i descrittori. La siamese a metric learning puro fallisce (quasi caso). L'AUC ~0.78
è **onesta** perché sopravvive ai controlli. Intervalli di confidenza al 95%
(bootstrap) dell'ordine di ±0.01, che escludono 0.5 (es. GBM 0.769 [0.760, 0.778])
→ significativa (vedi §8).

## 7. Task multiclasse: quale tessuto

**Idea:** classificare *quale* malattia (tessuto), non solo sano/malato. Poiché qui
i confondenti dominano, si lavora sotto controlli severi.

| File | Ruolo |
|---|---|
| `gold_standard_data.py` | Costruisce il dataset "gold-standard" confounder-controlled: **unico strato** (database CircleBaseV2 + protocollo Circle-seq → metodo e studio costanti), igiene etichette (doppioni fusi per tessuto), **appaiamento per lunghezza**, split per cluster; include la diagnostica lunghezza |
| `multiclass_data.py` | Dataset multiclasse per la variante a 17 malattie (split per cluster, con riparazione delle classi rare) |
| `train_multiclass.py` | Modelli: reti **prototipiche** (siamese) con distanza euclidea/coseno/triplet + baseline **softmax**, con addestramento e selezione bilanciati |
| `train_siamese_multiclass.py` | Modello siamese consolidato su rappresentazione ricca (spettro 3-mer + descrittori) |
| `experiment_goldstandard.py` | Binario e multiclasse dentro lo strato pulito, con baseline solo-lunghezza |

**Risultati** (gold-standard, 5 tessuti, length-matched; caso = 0.20):

| Modello | Accuracy | Acc. bilanciata | macro-F1 |
|---|---|---|---|
| Softmax bilanciato | 0.416 | 0.406 | 0.409 |
| Siamese (prototipica) | 0.400 | 0.394 | 0.381 |
| Baseline solo-lunghezza | 0.232 | — | — |

Prestazioni **per tessuto** (siamese): si riconoscono bene i tessuti più distinti —
Colon-retto (F1 ≈ 0.51) e Cataratta (F1 ≈ 0.48, unica condizione non tumorale) —
mentre i tumori epiteliali affini (Ipofaringe, Prostata, Stomaco) restano deboli
(F1 ≈ 0.22–0.37). Il segnale è **reale ma modesto** (~2× il caso), la siamese non
supera il softmax, e senza controlli l'accuratezza apparente sarebbe più alta ma
in gran parte *batch*.

## 8. Robustezza statistica

`python statistical_robustness.py` verifica che i risultati non siano frutto del
caso:

- **Binario** — bootstrap sul test (2000 ricampionamenti): ROC-AUC con IC 95%
  dell'ordine di ±0.01, che esclude 0.5 (es. GBM 0.769 [0.760, 0.778]) → segnale
  significativo.
- **Multiclasse** — 10 ricampionamenti indipendenti: accuratezza 0.40–0.42 (IC 95%
  [0.39, 0.43]), stabile. **Test di permutazione**: rimescolando le etichette il
  modello scende a 0.20 (il caso), contro 0.40 con le etichette vere; **p = 0.001**
  → altamente significativo rispetto al caso.

## 9. File del progetto

**Dati e descrittori:** `eccdna_utils.py` (calcolo dei 10 descrittori, parsing
FASTA), `dataUnderstanding.py` (EDA/leakage grezzo), `dataCleaning.py` (metadati
puliti), `descriptor_understanding_by_disease.py` (validazione descrittori per
sottotipo), `descriptor_extractor.py` (estrazione descrittori), `training_data.py`
(dataset binario bilanciato).

**Confondenti (dataset completo):** `diagnose_dataset.py`, `explore_full_dataset.py`.

**Binario:** `models_pytorch.py` (architetture), `train_models.py` (6 modelli).

**Multiclasse:** `gold_standard_data.py`, `multiclass_data.py`, `train_multiclass.py`,
`train_siamese_multiclass.py`, `experiment_goldstandard.py`.

**Robustezza:** `statistical_robustness.py` (IC 95% + test di permutazione).

**Tesi:** `tesi_descrittori_validazione.tex` (+ PDF, ignorati da git).

## 10. Bibliografia

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
