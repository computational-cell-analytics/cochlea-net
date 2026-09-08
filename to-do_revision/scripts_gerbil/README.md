# G301L segmentation, tables and MoBIE export

Segmentation of the one gerbil cochlea `G_LR_000301_L` that was still missing for the revision,
from raw channel to a MoBIE source. Two channels, both at key `setup0/timepoint0/s0` in the
workspace `/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools`: `GLR_301L_PV_fused.n5`
gives SGNs (model `SGN_v2`) and `GLR_301L_Vglut3_fused.n5` gives IHCs (model `IHC_v11`). Every
output lands under `prediction/G301L/{SGN_v2, IHC_v11, IHC_v11_dilated_mask1}`. The workspace
expires; the durable products are the counts in [`../gerbil-state.md`](../gerbil-state.md) and
the MoBIE source this folder writes.

The IHC run needed a second pass. The first mask was too tight, so `prepare_dilated_ihc_mask.py`
grows it by one iteration and only the newly covered blocks are predicted, on top of a copy of
the first prediction. This is why the IHC chain appears twice below.

## Order

```bash
# Pass 1 — SGN and IHC are independent and can run at the same time.
sbatch 2026-08-22_sbatch_mask_SGN-v2_G301L.sbatch        # mask.zarr + mean_std.json
sbatch 2026-08-22_sbatch_apply_SGN-v2_G301L.sbatch       # predictions.zarr, GPU array 0-9
sbatch 2026-08-22_sbatch_watershed_SGN-v2_G301L.sbatch   # seeds.zarr + segmentation.zarr
sbatch 2026-08-22_sbatch_mask_IHC-v11_G301L.sbatch
sbatch 2026-08-22_sbatch_apply_IHC-v11_G301L.sbatch
sbatch 2026-08-22_sbatch_watershed_IHC-v11_G301L.sbatch
sbatch 2026-08-22_sbatch_table_G301L.sbatch              # array 0 = SGN, 1 = IHC
sbatch 2026-08-23_sbatch_pyramid_G301L.sbatch            # view it, pick the components

# Pass 2 — IHC only, after the dilated mask was decided on.
#   Both of these steps are manual; nothing in this folder submits them.
python prepare_dilated_ihc_mask.py \
    --source-mask .../IHC_v11/mask.zarr \
    --output-folder .../IHC_v11_dilated_mask1 --iterations 1
cp -r .../IHC_v11/predictions.zarr .../IHC_v11/mean_std.json .../IHC_v11_dilated_mask1/

sbatch 2026-08-24_sbatch_apply_IHC-v11-dilated_G301L.sbatch      # delta blocks only
sbatch 2026-08-24_sbatch_watershed_IHC-v11-dilated_G301L.sbatch  # flips mask.zarr to the union
sbatch 2026-08-24_sbatch_table_IHC-v11-dilated_G301L.sbatch
sbatch 2026-08-25_sbatch_pyramid_IHC-v11-dilated_G301L.sbatch
sbatch 2026-08-25_sbatch_copy_IHC-v11_G301L_to_mobie.sbatch
```

Order matters. Each sbatch checks that its inputs exist and exits before doing any work if they
do not, so a step run too early fails fast rather than producing a partial result.

Two gaps to be aware of, both listed as manual above: `prepare_dilated_ihc_mask.py` has no submit
script, and the apply job for the dilated pass only *asserts* that the previous
`predictions.zarr` has been copied in — it never performs the copy.

## What the tables contain

`create_table_and_components.py` is the piece other folders depend on
([`../scripts_sgn_variance/README.md`](../scripts_sgn_variance/README.md) and
`flamingo_tools/validation.py` both expect its output). It writes two files next to the
segmentation:

- `default.tsv` — MoBIE-style morphology: `label_id`, `anchor_{x,y,z}`, `bb_min_{x,y,z}`,
  `bb_max_{x,y,z}`, `n_pixels`. Positions are micrometer and `bb_max` is exclusive. Written only
  if absent; pass `-f` to recompute.
- `default_components.tsv` — the same table plus `component_labels`, where 0 is
  background/filtered and the components are numbered from 1 in decreasing size. Always
  rewritten, because the component parameters are what gets tuned.

## Verification

The block counts are hard-coded as gates in the sbatch scripts, so they double as the expected
numbers: SGN 8,562 mask blocks → 34,248 prediction chunks per channel; IHC 157 → 628; the dilated
IHC union → 7,140. The MoBIE copy asserts 1,182 objects, of which 946 IHCs have
`component_labels > 0`.

## Files

| file | purpose |
|---|---|
| `2026-08-22_sbatch_mask_SGN-v2_G301L.sbatch` | Foreground mask + normalization stats, PV channel |
| `2026-08-22_sbatch_mask_IHC-v11_G301L.sbatch` | The same for the Vglut3 channel |
| `2026-08-22_sbatch_apply_SGN-v2_G301L.sbatch` | SGN network, 10-task A100 array |
| `2026-08-22_sbatch_apply_IHC-v11_G301L.sbatch` | IHC network, 10-task A100 array |
| `2026-08-22_sbatch_watershed_SGN-v2_G301L.sbatch` | Seeded watershed to SGN instances |
| `2026-08-22_sbatch_watershed_IHC-v11_G301L.sbatch` | Seeded watershed to IHC instances |
| `2026-08-22_sbatch_table_G301L.sbatch` | Morphology table + components for both channels |
| `2026-08-23_sbatch_pyramid_G301L.sbatch` | OME-Zarr pyramid for viewing both segmentations |
| `prepare_dilated_ihc_mask.py` | Dilate the IHC mask, split it into original / delta / union |
| `2026-08-24_sbatch_apply_IHC-v11-dilated_G301L.sbatch` | Predict only the delta blocks, H100 array |
| `2026-08-24_sbatch_watershed_IHC-v11-dilated_G301L.sbatch` | Switch to the union mask, then watershed |
| `2026-08-24_sbatch_table_IHC-v11-dilated_G301L.sbatch` | Table + components for the dilated IHCs |
| `2026-08-25_sbatch_pyramid_IHC-v11-dilated_G301L.sbatch` | Pyramid for the dilated IHCs |
| `2026-08-25_sbatch_copy_IHC-v11_G301L_to_mobie.sbatch` | Add the result to MoBIE, keep the curated table |
| `create_table_and_components.py` | Block-wise MoBIE table + component labeling |
| `create_multiscale_pyramid.py` | OME-Zarr v0.5 pyramid, levels s0-s5, zstd, sharded |
