# Synapse training runs and the v3 seed experiment

Training runs behind the synapse numbers in the revision, plus the two single-job production
predictions that preceded the resumable chain. The question the folder answers is how much of the
reported synapse accuracy is the model and how much is the training seed. Models are written to
`/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/networks/synapses`; the durable
product is [`v3_training_dynamics_report.md`](v3_training_dynamics_report.md) and the standard
deviation it feeds into Figure 2c.

**There is no seed argument.** Nothing in `train_synapse_detection.py`, `torch_em` or the czii
trainer sets a global seed, so weight initialization and patch order differ between runs on their
own — that is the variability being measured. `--random_state 42` is passed explicitly to every
run so that the train/val split stays *fixed*; without it the default derives from the model
suffix and would resample the split, confounding seed noise with split noise.

## The runs

`v3-1` … `v3-4` are byte-identical except for `-m`: four seed replicates at the full 100,000
iterations on an A100. `v3-5` … `v3-7` repeat that at 10,000 iterations on cheaper queues
(`v3-5` on an interactive `3g.40gb` MIG slice, `v3-6`/`v3-7` on a whole A100 under the 2 h QoS),
to sample seed behaviour without spending 32 h each.

The last two fill the empty cells of a data-versus-recipe comparison:

```
              15-crop v3 data      29-crop v5/v6 data
   heatmap    v3, v3-1..v3-7       v6-1
   flow       v3-flow-1            v5
```

`v6-1` changes only the dataset (`-v v6`); `v3-flow-1` changes only the recipe (`--use_flow`,
which switches to 5 output channels, the combined heatmap+flow loss and `MinPointSampler`). At
production settings v6-1 loses 0.163 recall against v3 while v5 gains 0.191 against v6-1, so the
two effects nearly cancel and neither run separates them alone.

## What the seed experiment found

Full numbers in [`v3_training_dynamics_report.md`](v3_training_dynamics_report.md), on the six
consensus test crops at threshold 0.5:

- **Test F1 spans 0.782 to 0.872 across seeds** — a range of 0.090 for one unchanged recipe.
- **Validation loss does not select the best detector.** The association is if anything positive
  (Spearman rho 0.39, p = 0.38, n = 7). The lowest-loss run, `v3-2`, scored 0.792; the best
  detector, `v3-4`, had a higher loss. The cause is a metric mismatch: a voxelwise heatmap loss
  against point-detection F1 after peak thresholding and IHC filtering, on a 3-crop validation
  set.
- The Figure 2c error bar uses the five best-scoring v3-style models and is **0.0155**. That is
  selection-conditioned: over all eight evaluated models the SD is 0.0313, about twice as large.

## The two production scripts

`synapse_process_GLR000301R.sbatch` and `synapse_process_GLR000302R.sbatch` run prediction, peak
detection and IHC matching for one cochlea in a single job. They are the predecessors of the
three-stage, resumable chain in [`../scripts_syn_gerbil/`](../scripts_syn_gerbil/README.md); use
that chain for new work. Note the trap it documents: `marker_detection` skips the prediction
outright when `synapse_detection.tsv` already exists, which is how a truncated `G_LR_000302_R`
result passed as finished. Both scripts start with a GPU check that exits before the long
prediction, because a silent CPU fallback is about 50x slower and never finishes in the limit.

## Files

| file | purpose |
|---|---|
| `train_synapse_v3-1.sbatch` … `v3-4` | Seed replicates of the shipped v3 model, 100k iterations |
| `train_synapse_v3-5.sbatch` … `v3-7` | The same at 10k iterations, on cheaper queues |
| `train_synapse_v3-flow-1.sbatch` | v3 data with the v5 recipe (`--use_flow`) |
| `train_synapse_v6-1.sbatch` | v6 data with the v3 recipe |
| `synapse_detect_v5-variation_F1val.sbatch` | Run the four v5 fold models over the six test crops |
| `synapse_process_GLR000301R.sbatch` | Single-job prediction + detection + IHC matching, G301R |
| `synapse_process_GLR000302R.sbatch` | The same for G302R |
| `distance_unet_checkpoint.py` | Load only the inference model from a torch-em checkpoint |
| `v3_training_dynamics_report.md` | The seed experiment: scores, dynamics and what to conclude |
