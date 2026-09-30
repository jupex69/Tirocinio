# Risultati (tabelle `.tsv`)

Output degli esperimenti, in formato tab-separated. Ogni file è prodotto dallo script
`src/<stesso-nome>.py` (o dall'esperimento indicato). Sono raggruppati per domanda di
ricerca; i file `*_perclass` / `*_per_class` / `*_per_tissue` sono i dettagli per classe.

## RQ1 — presenza (sano vs malato)
- `full_classifier_metrics.tsv`, `full_classifier_per_class.tsv` — sei modelli, entro studio
- `binary_crossstudy.tsv` — binario in validazione su un nuovo studio (leave-one-database-out)

## RQ2 — identità (quale malattia, entro lo stesso studio)
- `disease_multiclass.tsv`, `disease_multiclass_perclass.tsv` — multiclasse a due database
- `disease_multiclass_controlled*.tsv` — con metodo **e** database tenuti costanti
- `final_comparison.tsv` — confronto RandomForest vs 3 loss siamesi (one-shot/few-shot)
- `multiclass_loss_comparison.tsv`, `multiclass_fewshot.tsv`, `oneshot_*.tsv`, `fewshot_*.tsv`

## RQ3 — generalizzazione a un nuovo studio e limite dei dati
- `classify_which.tsv`, `multiclass_confounder_profile.tsv` — perché il segnale non trasferisce
- `cross_source_transfer.tsv`, `cross_source_per_tissue.tsv` — trasferimento tra sorgenti
- `honest_multiclass.tsv`, `honest_all_siamese.tsv` — valutazioni oneste (LOSO)

## Descrittori, k-meri, pre-addestramento (ablation)
- `richer_kmers.tsv`, `richer_kmers_fewshot.tsv`, `richer_kmers_siamese.tsv` — k-meri di ordine crescente
- `new_descriptors_multiclass_rf.tsv`, `new_descriptors_multiclass_siamese.tsv` — descrittori nuovi ortogonali
- `richer_descriptors_cardiac.tsv` — descrittori fisici sul task cardiaco
- `pretrain_fewshot.tsv` — effetto del pre-addestramento self-supervised

## Controlli e robustezza
- `leakage_check.tsv`, `leakage_check_seq.tsv` — assenza di leakage
- `merged_random_vs_studysplit.tsv` — split casuale vs per studio
- `method_search_grid.tsv`, `certified_detectable.tsv`

## Altri esperimenti esplorativi
- `cnn_multiclass.tsv`, `adversarial_*.tsv`, `disease_vs_tissue*.tsv`,
  `confusion_tissue.tsv`, `disease_state_colorectal.tsv`
