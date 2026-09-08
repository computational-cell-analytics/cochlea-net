# Revision experiment hand-off

This branch collects the analyses and production fixes made for the revision. The large
intermediate data live in `/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools`; the
repository contains the scripts, provenance and durable accuracy tables.

## Main results

- **Model variance (Figure 2):** I ran the four `SGN_v2` seed variants on all five validation
  cochleae. Their F1 scores span 0.880-0.893 (mean 0.885, SD 0.005), similar to the IHC seed
  experiment. See [sgns.md](sgns.md) and [the executable workflow](scripts_sgn_variance/README.md).
  I also trained/evaluated v3-style synapse seed variants on the same split. Test F1 varies much
  more (0.782-0.872), and minimum voxelwise validation loss does not select the best detector.
  Figure 2 currently shows the SD of the five best-scoring v3-style models (0.0155); this is a
  selection-conditioned display, while the SD over all eight evaluated models is 0.0313. See the
  [training-dynamics report](scripts_synapses/v3_training_dynamics_report.md).

- **Wild-type gerbils (Figure 5):** I completed the missing G301L segmentation and the G301L/G302R
  synapse runs. The earlier G302R prediction was truncated; the replacement uses a resumable
  three-stage Slurm workflow with an explicit coverage check. The remaining low counts in parts of
  G301L/G301R track poor CtBP2 contrast, not skipped prediction blocks. Block-local affine
  normalization and reducing the detection threshold did not recover the signal. Figure 5 now uses
  all four cochleae for SGN/IHC counts, but excludes staining-limited G301L from the synapse panel
  and excludes zero-detection IHCs for the other three. This conditional reporting rule must be
  stated in the legend/Methods. See [the diagnosis and manuscript decision](gerbil-synapses.md),
  [final counts](gerbil-state.md), and [production workflow](scripts_syn_gerbil/README.md).

- **M29 Type-II rescue (Figure 3):** I implemented a local, non-destructive rescue that merges
  PV-derived SGNs into the CR/Ntng1 segmentation for M29L/R and recomputes subtype assignments.
  In the fixed cohort, Type II changes from 0.41% to 1.74% for M29L and from 3.08% to 2.52% for
  M29R. The generated segmentations/tables are intentionally ignored and have not been promoted
  to MoBIE or S3; see [the rescue README](scripts_sgn_typeii_rescue/README.md). However, we
  decided **not to include this in the paper** and to stay with the 2 cochleae we had before for
  this figure, so `plot_fig3.py` no longer reads the rescued tables at all.

- **Supporting revision work:** I recorded consistent training/test annotation counts in
  [training_data_counts.md](training_data_counts.md), added OTOF annotation/export and random-forest
  evaluation utilities, updated the manuscript plotting scripts, and expanded the gallery export
  code (ROI/view-state handling, marker/subtype exports and tests). The napari plugin can now show
  intermediate model outputs.

There are several other minor changes to figure scripts etc.

## Open decisions / next steps

1. **Do not start another synapse seed comparison with the current training path.** The input is
   not normalized consistently, validation patches change on every evaluation, and the training
   volumes contain many plausible but unlabeled CtBP2 spots that are penalized as background.
   Before retraining, add identical train/inference normalization, fixed detection-level
   validation, and an ignore mask outside exhaustively annotated/IHC-valid regions. The evidence
   and proposed ablations are in [synapses.md](synapses.md). The v6/flow comparison machinery is
   present, but those scores are not in the committed accuracy table.

For transferred inputs and remaining storage locations, see [data_transfer.md](data_transfer.md).
