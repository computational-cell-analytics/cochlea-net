# M29 SGN Type-II rescue

This directory contains an isolated, read-only-with-respect-to-project-data rescue analysis for
`M_AMD_N190_L` (M29L) and `M_AMD_N190_R` (M29R). The analysis merges the existing PV-derived
`SGN_v2` detections into the current `CR_Ntng1_SGN_v2` segmentation and repeats the CR/Ntng1
subtype assignment.

All generated data are written below the ignored `output/` directory. The MoBIE project, the
prediction directories, and S3 are never modified.

For `M_AMD_N190_R`, the `SGN_v2` and `CR_SGN_v2` prediction arrays are stored under one another's
biological names. The current CR+Ntng1 label range and direct table/array geometry checks show that
the biological PV prediction is the local/S3 `CR_SGN_v2` array; its matching geometry table is the
local `SGN_v2/default.tsv`. The script records this exception in provenance and validates the
table/array pairing before merging. It also normalizes M29R's full-volume encoded background label
6734 to zero in the new output only.

The merge follows the existing `scripts/measurements/merge_sgn_segmentation.py` convention:

1. Compute overlaps at OME-Zarr scale `s2`.
2. Select PV objects whose summed IoU with non-background CR+Ntng1 objects is below 0.25.
3. At full resolution, insert selected PV pixels only where CR+Ntng1 is background.
4. Offset added PV label IDs by the maximum CR+Ntng1 label ID.

Only the median CR and Ntng1 intensities are needed by the current absolute-intensity subtype
thresholds. Existing measurements are reused for unchanged CR+Ntng1 objects; medians are computed
from the S3 image volumes for every newly required object. A deterministic sample of unchanged
objects is remeasured to verify exact agreement.

Run both cochleae:

```bash
python -u to-do_revision/scripts_sgn_typeii_rescue/rescue.py all
```

Important outputs:

```text
output/<cochlea>/CR_Ntng1_PV_SGN_v2/segmentation.zarr
output/<cochlea>/CR_Ntng1_PV_SGN_v2/default.tsv
output/<cochlea>/CR_Ntng1_PV_SGN_v2/CR_object-measures.tsv
output/<cochlea>/CR_Ntng1_PV_SGN_v2/Ntng1_object-measures.tsv
output/<cochlea>/selection.json
output/<cochlea>/provenance.json
output/subtype_comparison.tsv
output/subtype_comparison.json
output/report.md
```

The comparison contains two after-rescue cohorts:

- `after_fixed_cohort`: the original CR+Ntng1 component 1 plus rescued PV objects within 30 µm
  of it. This is the primary comparison because it isolates the effect of adding PV detections.
- `after_recomputed_component`: component 1 after recomputing the standard 30 µm SGN graph on
  the complete merged table. This is a sensitivity analysis for changes in component topology.

## Figure 3 no longer reads these tables

The rescue is not part of the paper. `scripts/figures/plot_fig3.py` reads every cochlea from its
S3 `SGN_v2` table, M29L/R included, so nothing here is on the figure path any more. The rescued
tables, the `fixed_cohort` selection and the `FLAMINGO_M29_RESCUE_OUTPUT` override were removed
from the plotter together with the panels named after the rescue.

This folder stays as the record of what the rescue did and what it would have changed.
