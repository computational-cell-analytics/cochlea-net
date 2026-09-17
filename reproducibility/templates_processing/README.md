# Segmentation and detection workflows

Implements workflows to segment SGNs or IHCs, and to detect ribbon synapses in slurm.

For SGN segmentation run:
- mean_std_SGN_template.sbatch
- apply_unet_SGN_template.sbatch
- segment_unet_SGN_template.sbatch

For IHC segmentation run:
- mean_std_IHC_template.sbatch
- apply_unet_IHC_template.sbatch
- segment_unet_IHC_template.sbatch

After this, run the following to add segmentation to MoBIE, create component labels and upload to S3:
- templates_transfer/mobie_segmentation_template.sbatch
- templates_transfer/s3_seg_template.sh
- `flamingo_tools.label_components`, with the parameters recorded in processing/
- templates_transfer/s3_seg_template.sh

For ribbon synapse detection run:
- mean_std_synapse_template.sbatch
- apply_synapse_template.sbatch
- detect_synapse_peaks_template.sbatch

Pass the IHC segmentation to the first step, which restricts the prediction to the dilated IHC
region, and to the third step, which matches the detections to the IHCs. This is the standard
case: on a full cochlea the mask removes about 97 % of the prediction blocks. Without a
segmentation the prediction falls back to the full volume and the detections are not matched.

To run all three steps as a single job instead, without splitting up the prediction:
- detect_synapse_template.sbatch, with the IHC segmentation as its fourth argument

After this, run the following to add detections to MoBIE and upload to S3:
- templates_transfer/mobie_spots_template.sbatch
- templates_transfer/s3_synapse_template.sh

## Model training

To train a ribbon synapse detection model run:
- train_synapse_template.sbatch

The template trains the heatmap only, which reproduces synapse_detection_v3. Pass `--use_flow` to
also train the 4 stereographic flow channels, as for synapse_detection_v5.
