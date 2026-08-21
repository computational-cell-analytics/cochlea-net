# Synapses

## Network training

Since PR `#133` the synapse detection network is trained without flow per default. The flow option has to be selected by supplying a `--use_flow` flag.
The training data is located in separate `images` and `labels` directories, so the training/validation split has to be reproducible. The random state `--random_state 42` has to be passed for training the networks to test the variation to ensure the same split.
An example script for training `synapses_v3-1` is `train_synapse_v3-1.sbatch`.

A new synapse network v6 or its variation could be trained by substituting the function call with
```bash
# train synapse network v6
python $SCRIPT_DIR/train_synapse_detection.py -v v6 --random_state 42
# train synapse network v6-1
python $SCRIPT_DIR/train_synapse_detection.py -v v6 -m v6-1 --random_state 42
```
The training data is already prepared in `/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/training_data/v6` and is identical to the `v5` one.

## Network application
Potentially relevant for a new synapse network.
An example script for cochlea `G_LR_000302_R` is `synapse_process_GLR000302R.sbatch`.
The volume can be masked based on an IHC segmentation, which can be local or on the S3 bucket.
The mask may cut off potential synapses because its size is currently limited to the extension of the IHC segmentation.
Future updates may improve this by dilating the mask before applying the network.


## Post-processing

* transfer synapse detection to MoBIE
* transfer to S3 bucket
* check content of `flamingo_tools/postprocessing./synapse_per_ihc_utils.py` (probably already updated)

### Calculate synapses near IHC components
This script reads the information of `synapse_per_ihc_utils.py`:
```bash
OUT_DIR=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses/ihc_counts_v11/
python ~/flamingo-tools/scripts/measurements/measure_synapses.py -c <cochlea1> <cochlea2> ... -o "$OUT_DIR"
```

### Add `syn_per_IHC` to segmentation table
The column `syn_per_IHC` is read by some plot functions for figure 3. It has to be added to the segmentation table and needs to be uploaded to the S3 bucket.
```bash
python ~/flamingo-tools/scripts/synapse_marker_detection/add_synapse_per_ihc.py -c <cochlea> -o .
```
will produce a segmentation table in the current directory. It can be transferred to S3 with:
```bash
COCHLEA=G_LR_000302_R
rclone copyto "$COCHLEA"_v11_syn-per-ihc.tsv cochlea-lightsheet:cochlea-lightsheet/"$COCHLEA"/tables/IHC_v11/default.tsv
```

## Network variation

The script `synapse_detect_v5-variation_F1val.sbatch` was used to apply the synapse network `v5` for the validation.
The script has to be adapted once the variation scripts for v3 have been trained.
Afterwards, the accuracy can be calculated using

```bash
python scripts/validation/synapses/run_evaluation.py -v v3-1 -o ~/flamingo-tools/reproducibility/model_accuracy/
python scripts/validation/synapses/run_evaluation.py -v v3-2 -o ~/flamingo-tools/reproducibility/model_accuracy/
python scripts/validation/synapses/run_evaluation.py -v v3-3 -o ~/flamingo-tools/reproducibility/model_accuracy/
python scripts/validation/synapses/run_evaluation.py -v v3-4 -o ~/flamingo-tools/reproducibility/model_accuracy/
```
The accuracy values will be written into `reproducibility/model_accuracy/synapses.json`.
From there they can be read by `plot_fig2.py`.
