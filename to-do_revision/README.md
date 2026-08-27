Workspace location:
/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools

Open TODOs:
- Export gerbil IHC / Syn Det. for Lennart.
  Not the same thing as `<workspace>/figures-lennart/`, which is his figure drafts pulled *from*
  the archive by `figures-lennart/transfer-fig.sh`, not an export going the other way.
- Synapses: check validation for bugs, compare v3 and v6, train folds for best, re-eval if needed.
  In progress: `v3-1` and `v6-1` are trained, `v3-2` .. `v3-4` were still training on 2026-08-24
  (jobs 15464598-15464600). Scripts in `scripts_synapses/`, instructions in `synapses.md`.
- Wild-type gerbil synapse detection, for the four cochleae of figure 5. All prediction chains
  have finished. The complete `G_LR_000302_R` result has replaced the truncated S3/MoBIE and
  `ihc_counts_v11` tables. `G_LR_000301_L` is computationally complete, but its CTBP2 staining is
  biologically partial over about one third of the cochlea; do not interpret it as a complete
  synapse-per-IHC measurement. See `scripts_syn_gerbil/README.md` for the validation evidence.

Resolved:
- Implement the mask dilation for synapse prediction.
  Already in the library: `build_ihc_mask` in `flamingo_tools/segmentation/synapse_detection.py`
  binarizes the IHC segmentation at `s4` and dilates it with a 9^3 structure element, so roughly
  55 um around the IHCs. Added in PR #132 (commit 8cd974e). The caveat in `synapses.md` about the
  mask being "limited to the extension of the IHC segmentation" predates it.
- Copy over Vglut3 for new gerbil.
  `<workspace>/G301L/GLR_301L_Vglut3_fused.n5` (111 GB), transferred as described in
  `data_transfer.md`. It is the input of the IHC_v11 segmentation of `G_LR_000301_L`, see
  `scripts_gerbil/2026-08-22_sbatch_mask_IHC-v11_G301L.sbatch` and its `apply` counterpart.
