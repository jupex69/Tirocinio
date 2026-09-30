# Modelli pronti all'uso

Due classificatori **RandomForest** addestrati sui **10 descrittori interpretabili**
della sequenza di eccDNA (nessuna GPU, nessun k-mero: solo i dieci descrittori del
set finale della tesi). I file `.joblib` sono già addestrati e pronti.

| File | Compito | Generalizza a un nuovo studio? |
|---|---|---|
| `binary_presence_rf.joblib` | **sano vs malato** (presenza) | **Sì** (ROC-AUC ≈ 0,71 cross-studio) |
| `multiclass_disease_rf.joblib` | **quale malattia** (9 classi) | **No** — utile solo entro lo stesso studio |

> **Nota onesta.** Il binario coglie un segnale reale e trasferibile. Il multiclasse,
> invece, entro lo stesso studio arriva a ~2,6× il caso ma **crolla al livello del caso
> su uno studio nuovo**: la malattia è quasi sempre confusa con lo studio/protocollo
> (vedi tesi, RQ3). Il multiclasse va usato come strumento di ricerca, **non come
> diagnosi**.

## Uso rapido

```bash
# sano vs malato
python predict.py --model binary --seq ACGT...

# quale malattia (con il caveat sopra)
python predict.py --model multiclass --seq ACGT...

# un intero FASTA (una riga di output per sequenza)
python predict.py --model binary --fasta campioni.fa
```

Come modulo Python:

```python
from predict import predict
etichetta, probabilita = predict("ACGT...", kind="binary")
```

Ogni `.joblib` contiene un dizionario con `model` (il RandomForest addestrato),
`features` (i 10 descrittori nell'ordine atteso), `classes` e `task`.

## Requisiti

`numpy`, `scikit-learn`, `joblib` (già in `../requirements.txt`). `predict.py` usa
`src/eccdna_utils.py` per calcolare i descrittori dalla sequenza, quindi va eseguito
dentro il repository (la cartella `src/` deve essere presente).

## Ri-addestrare i modelli

Con i dati in `src/data/` presenti:

```bash
python train_export.py
```

Riusa gli stessi loader e cache del progetto (`src/`) e riscrive i due `.joblib`.
RandomForest è invariante alla scala, quindi i modelli lavorano sui descrittori grezzi.
