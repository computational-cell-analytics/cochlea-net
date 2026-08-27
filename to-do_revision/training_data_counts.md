# Training- and test-data counts for SGN v2, IHC v11, and synapse v3

The strict optimizer-training counts are:

| Network | Training crops | Cells / puncta | Validation | Total development pool |
|---|---:|---:|---:|---:|
| SGN v2 | 40 | 1,706 SGNs | 8 crops / 896 SGNs | 48 crops / 2,602 SGNs |
| IHC v11 | 75 | 1,430 IHCs | 14 crops / 263 IHCs | 89 crops / 1,693 IHCs |
| Synapse v3 | 12 | 3,236 puncta | 3 crops / 760 puncta | 15 crops / 3,996 puncta |

For a methods section that counts all training and validation annotations, report **48 crops / 2,602 SGNs**, **89 crops / 1,693 IHCs**, and **15 crops / 3,996 synaptic puncta**, respectively.

## Consensus-annotated test sets

| Target | Test data | Consensus annotations |
|---|---:|---:|
| SGNs | 12 image slices | 2,565 SGNs |
| IHCs | 12 image slices | 286 IHCs |
| Synapses | 6 image crops | 1,149 synaptic puncta |

These are the independent test sets used by the consensus-based model evaluations. The totals count
one consensus CSV per slice or crop. For IHCs and synapses, only rows whose `annotator` value is
`consensus` are included; unmatched annotations retained from individual annotators are excluded.

## Counting details

- Cell counts use the project's minimum-size convention: connected components with at least 1,000 pixels are counted as cells. Smaller components are treated as annotation artifacts and are not included in the cell totals.
- The `Dataset` column in the SGN and IHC inventory tables describes their storage folders. The training code pooled these crops and created its own 85:15 training-validation split. Consequently, the strict checkpoint splits differ from the `train` and `val` totals reported directly from that column.
- The synapse v3 training pool was explicitly divided into 12 training and 3 validation crops.
- The older `test` partition recorded in `doc/data/synapses_v3.tsv` contains 7 crops and 1,644
  single-annotator puncta. It is not the current 6-crop consensus-annotated model-evaluation set,
  and neither test set is included in the development-pool total above.

## Sources

- `doc/data/SGN_v2.tsv`
- `doc/data/IHC_v11.tsv`
- `doc/data/synapses_v3.tsv`
- IHC v11 checkpoint `split.json`
- SGN v2 checkpoint loader paths and the historical training script
- Synapse v3 reproducible training-validation split
- SGN consensus annotations in `AnnotatedImageCrops/F1ValidationSGNs/final_annotations/final_consensus_annotations`
- IHC consensus annotations in `AnnotatedImageCrops/F1ValidationIHCs/consensus_annotation`
- Synapse consensus annotations in `AnnotatedImageCrops/Synapses_2026-04/consensus_annotation`
