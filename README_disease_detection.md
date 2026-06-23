# eccDNA Disease Detection Dataset

Questo dataset collega sequenze eccDNA a metadati biologici utili per modelli di classificazione,
metric learning e reti siamesi.

L'idea principale del progetto e':

```text
sequenza eccDNA -> disease / classe biologica associata
```

La versione gia pronta espone anche un target binario piu semplice:

```text
sequenza eccDNA -> healthy oppure disease_associated
```

La label binaria e':

```text
0 = healthy
1 = disease_associated
```

## File da usare

Usare questi due file insieme:

```text
data/processed/eccdna_disease_detection_metadata.tsv
data/processed/eccdna_disease_detection.body.fa
```

Il TSV contiene metadati, coordinate cromosomiche, label e split.
Il FASTA contiene le sequenze associate alle stesse regioni eccDNA.

La chiave per collegarli e':

```text
id
```

Esempio:

Nel TSV:

```text
id = eccDNABase_000124402
disease = Prostate Cancer
disease_binary_label = 1
chrom = chr1
start = 11080
end = 11289
```

Nel FASTA:

```text
>eccDNABase_000124402|label=1|chrom=chr1|start=11080|end=11289|...
SEQUENZA...
```

Quindi la parte prima del primo `|` nell'header FASTA deve corrispondere alla colonna `id` nel TSV.

## Obiettivi ML supportati

### 1. Baseline binaria

Usare:

```text
disease_binary_label
```

Interpretazione:

```text
0 = disease uguale a Health/Healthy
1 = tutte le altre disease
```

Questo e' il primo esperimento consigliato per controllare che la pipeline di training funzioni.

### 2. Classificazione disease

Usare:

```text
disease
```

come classe di malattia. Prima del training multiclass conviene creare una colonna pulita, per
esempio `disease_class`, normalizzando maiuscole/minuscole e sinonimi:

```text
Healthy / Health -> healthy
Prostate Cancer / prostate cancer -> prostate_cancer
Gastric cancer / Gastric Cancer -> gastric_cancer
```

Per un dataset solido, non usare tutte le disease indistintamente: tenere solo classi con abbastanza
sequenze, cluster genomici e diversita' di tissue/source.

### 3. Rete siamese o triplet loss

Per una rete siamese, generare coppie dopo aver applicato lo split:

```text
id_a
id_b
same_disease
disease_a
disease_b
split
```

Coppia positiva:

```text
same_disease = 1
disease_a == disease_b
```

Coppia negativa:

```text
same_disease = 0
disease_a != disease_b
```

Per triplet loss:

```text
anchor_id
positive_id
negative_id
disease_anchor
split
```

Regola importante: creare coppie/triplette solo dentro lo stesso split. Non creare coppie prima dello
split, altrimenti sequenze correlate possono finire indirettamente sia in train sia in test.

## Colonne principali del TSV

```text
id
disease_binary_label
disease_binary_name
disease
chrom
start
end
length
gc
n_fraction
source_db
sample
tissue
cell_line
method
library_type
cluster_id
role
sequence_kind
sequence_fasta
sequence_body_id
sequence_circular_id
sequence_breakpoint_id
split_cluster
split_deepcircle_external
```

Le colonne piu importanti per il modello sono:

```text
id
disease_binary_label
disease
chrom
start
end
length
gc
tissue
cell_line
source_db
split_cluster
```

## Ruolo di disease, cell_line e tissue

`disease` e' il target principale per la classificazione disease-aware.

`cell_line` e `tissue` non sono necessariamente la classe finale. Sono campi utili per:

```text
filtrare esempi poco affidabili
bilanciare classi e batch
costruire test piu difficili
controllare confondenti
```

Esempi di stress test:

```text
train su alcune cell_line, test su cell_line mai viste
train su alcuni tissue, test su tissue diversi
coppie positive con stessa disease ma cell_line diversa
coppie negative con disease diversa ma stesso tissue
coppie negative con disease diversa ma lunghezza/GC simili
```

Questo evita che la rete impari scorciatoie troppo facili, per esempio riconoscere una cell line
invece della disease.

## Valori mancanti

Questo dataset contiene solo eccDNA positivi con `disease` compilato. Invece `cell_line`, `tissue` e
`sample` possono essere incompleti perche le sorgenti non hanno tutte la stessa ricchezza di
metadati.

Considerare come mancanti:

```text
campo vuoto
NA
NaN
nan
None
null
```

Nella versione corrente del dataset:

```text
rows = 3,755,079
healthy = 445,138
disease_associated = 3,309,941
```

Le sorgenti principali presenti sono:

```text
CircleBaseV2
eccDNABase
```

`eccDNABase` ha molti record con `tissue`, ma in genere non ha `cell_line`.
`CircleBaseV2` ha `disease` e spesso `tissue`, ma `cell_line` e' molto piu sparsa.

Controllo rapido consigliato prima del training:

```python
import pandas as pd

missing_values = ["", "NA", "NaN", "nan", "None", "none", "NULL", "null"]

df = pd.read_csv(
    "data/processed/eccdna_disease_detection_metadata.tsv",
    sep="\t",
    low_memory=False,
    keep_default_na=False,
)

for col in ["disease", "tissue", "cell_line", "source_db", "split_cluster"]:
    missing = df[col].astype(str).str.strip().isin(missing_values).sum()
    print(col, missing, "missing /", len(df))

print(df["disease_binary_label"].value_counts(dropna=False))
print(df["source_db"].value_counts(dropna=False))
```

## Sequenze

Il file:

```text
data/processed/eccdna_disease_detection.body.fa
```

contiene la sequenza genomica dell'intervallo eccDNA:

```text
chrom:start-end
```

Questa e' la sequenza consigliata per il primo esperimento ML.

Se vuoi usare invece il contesto del breakpoint:

```powershell
python scripts/make_disease_detection_dataset.py --sequence-kind breakpoint_context --source-db CircleBaseV2 eccDNABase eccDNA_xlsx
```

Output:

```text
data/processed/eccdna_disease_detection.breakpoint_context.fa
```

## Split consigliato

Per training/validation/test usare:

```text
split_cluster
```

Motivo: riduce il rischio che regioni genomiche molto simili finiscano sia in train sia in test.

Non usare `split_random` come risultato principale, perche puo dare performance troppo ottimistiche.

Per stress test piu severi, creare split aggiuntivi usando:

```text
cell_line
tissue
source_db
```

Esempi:

```text
cell_line_holdout
tissue_holdout
source_holdout
```

## Generazione dei file

Dalla root del progetto:

```powershell
conda activate ecc_Dna_hotspot
python scripts/make_disease_detection_dataset.py --sequence-kind body --source-db CircleBaseV2 eccDNABase eccDNA_xlsx
```

Output:

```text
data/processed/eccdna_disease_detection_metadata.tsv
data/processed/eccdna_disease_detection.body.fa
```

Nota: se una sorgente non appare nell'output finale, controllare che nel manifest di partenza abbia
`disease` compilato e che i metadati siano stati propagati correttamente.

## Esempio di caricamento Python

```python
from pathlib import Path

import pandas as pd


metadata = pd.read_csv(
    "data/processed/eccdna_disease_detection_metadata.tsv",
    sep="\t",
    low_memory=False,
)

train = metadata[metadata["split_cluster"] == "train"]
val = metadata[metadata["split_cluster"] == "val"]
test = metadata[metadata["split_cluster"] == "test"]

y_train_binary = train["disease_binary_label"]
y_train_disease = train["disease"]
```

Per leggere il FASTA in un dizionario semplice:

```python
def read_fasta(path):
    records = {}
    current_id = None
    chunks = []

    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(chunks)
                current_id = line[1:].split("|", 1)[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        records[current_id] = "".join(chunks)

    return records


sequences = read_fasta("data/processed/eccdna_disease_detection.body.fa")

example_id = train.iloc[0]["id"]
example_sequence = sequences[example_id]
example_binary_label = train.iloc[0]["disease_binary_label"]
example_disease = train.iloc[0]["disease"]
```

## Note importanti

- Questo dataset contiene solo esempi positivi eccDNA con `disease` compilato.
- Non e' il task eccDNA vs non-eccDNA: per quello serve anche `label=0` dal manifest completo.
- `DeepCircle` non viene incluso nel comando consigliato perche ha metadati disease meno ricchi ed e'
  meglio usarlo come benchmark esterno.
- Il TSV non contiene direttamente la sequenza per evitare un file enorme e scomodo: le sequenze
  stanno nel FASTA.
- Per reti siamesi e triplet loss, generare coppie/triplette dopo lo split e salvare gli ID originali
  in modo da poter risalire sempre a `disease`, `cell_line`, `tissue` e `source_db`.
