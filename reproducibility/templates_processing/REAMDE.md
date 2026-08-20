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
- templates_transfer/sync_mobie.py
- label_components/repro_label_components.py
- templates_transfer/sync_mobie.py

For ribbon synapse detection run:
- mean_std_synapse_template.sbatch
- apply_synapse_template.sbatch
- detect_synapse_peaks_template.sbatch

Pass an IHC segmentation to the first step to restrict the prediction to the region around the
IHCs, and to the third step to also write the detections matched to the IHCs.

To run the detection as a single job instead, without splitting up the prediction:
- detect_synapse_template.sbatch, without an associated IHC segmentation
- detect_synapse_marker_template.sbatch, with an associated IHC segmentation

After this, run the following to add detections to MoBIE and upload to S3:
- templates_transfer/mobie_spots_template.sbatch
- templates_transfer/sync_mobie.py
