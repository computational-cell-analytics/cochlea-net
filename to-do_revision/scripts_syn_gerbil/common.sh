# Shared configuration for the wild-type gerbil ribbon synapse detection.
# Sourced by every sbatch script in this folder; not meant to be executed on its own.

SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
SYN_GERBIL_DIR=$SCRIPT_REPO/to-do_revision/scripts_syn_gerbil

# Everything this pipeline writes lives in the workspace. Nothing is written into the
# existing vast predictions folders, so the current (broken) G_LR_000302_R result stays
# untouched until the new one has been checked against it.
WS=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools
OUT_ROOT=$WS/synapses-v3
DATA_ROOT=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet
MODEL=$DATA_ROOT/trained_models/Synapses/synapse_detection_model_v3.pt

# The two wild-type gerbils that still need a synapse detection. G_EK_000233_L and
# G_LR_000301_R are finished and verified, see the README.
COCHLEAE=(G_LR_000301_L G_LR_000302_R)
# Five tasks, not ten: only 2.8 % of the blocks are in the mask, so ten tasks were ~5 min of
# work each and mostly slurm overhead. Five gives 227 to 269 in-mask blocks per task, about
# 11 to 12.5 min. Changing this invalidates any existing prediction, because the block-to-task
# assignment is a permutation split into exactly this many parts.
PREDICTION_INSTANCES=5

# MIG slice for the preemptible queue. Measured, not guessed: peak device occupancy is
# 18.10 GiB, so a 1g.10gb slice (9.5 GiB visible) dies with a CUDA OOM part way through the
# first in-mask block, while 1g.20gb (19.5 GiB) fits with 1.40 GiB spare. 3g.40gb has more
# room but there are only eight of them and they are the scarcest thing on the partition, so
# the wait costs more than the head room is worth. See the README.
PREEMPTIBLE_PARTITION=grete:preemptible
PREEMPTIBLE_SLICE=1g.20gb

# Passed to the apply job as PYTORCH_ALLOC_CONF when non-empty. Deliberately empty:
# expandable_segments is worth 1.6x on the prediction (1.76 against 2.78 s per block) but takes
# device occupancy from 18.10 to 19.42 GiB, which is 100 % of the 1g.20gb slice above with
# 0.08 GiB spare. The speed is not worth running that close to the edge. See the README.
SYN_ALLOC_CONF=

# Key of the IHC segmentation used to build the prediction mask (low scale, held in memory
# and dilated) and to match the detections to the IHCs (full resolution).
MASK_INPUT_KEY=s4
MASK_FULL_KEY=s0

# Detection parameters. VOXEL_SIZE and THRESHOLD are the library defaults. MAX_DISTANCE=8
# reproduces the three finished cochleae, whose synapse_detection_filtered.tsv tops out at
# 7.99 um. measure_synapses.py re-filters to 3 um when it counts, so keeping 8 here leaves
# that cutoff free to change without re-running the detection.
VOXEL_SIZE="0.38 0.38 0.38"
THRESHOLD=0.5
MAX_DISTANCE=8

# The CTBP2 channel. Both copies live in the workspace: G_LR_000301_L was transferred from
# the UKON archive as a fused n5, G_LR_000302_R was copied back from S3 because its raw data
# is no longer on vast.
raw_path() {
	case "$1" in
		G_LR_000301_L) echo "$WS/G301L/GLR_301L_CTBP2_fused.n5" ;;
		G_LR_000302_R) echo "$WS/G_LR_000302_R/CTBP2.ome.zarr" ;;
		*) echo "unknown cochlea: $1" >&2; return 1 ;;
	esac
}

raw_key() {
	case "$1" in
		G_LR_000301_L) echo "setup0/timepoint0/s0" ;;
		G_LR_000302_R) echo "s0" ;;
		*) echo "unknown cochlea: $1" >&2; return 1 ;;
	esac
}

# The finalized IHC_v11 segmentation, as a multiscale ome-zarr. G_LR_000301_L uses the
# dilated-mask re-prediction; G_LR_000302_R's pyramid only existed on S3 and was copied in
# (52 MiB).
mask_path() {
	case "$1" in
		G_LR_000301_L) echo "$WS/prediction/G301L/IHC_v11_dilated_mask1/segmentation.ome.zarr" ;;
		G_LR_000302_R) echo "$WS/G_LR_000302_R/IHC_v11.ome.zarr" ;;
		*) echo "unknown cochlea: $1" >&2; return 1 ;;
	esac
}

# Folder holding mask.zarr, mean_std.json, predictions.zarr and the detection tables.
# OUTPUT_FOLDER_OVERRIDE lets a deliberately versioned rerun coexist with an earlier result.
# It is exported explicitly by the rerun submission script; ordinary runs keep the historical
# OUT_ROOT/<cochlea> layout.
output_folder() {
	if [[ -n "${OUTPUT_FOLDER_OVERRIDE:-}" ]]; then
		echo "$OUTPUT_FOLDER_OVERRIDE"
	else
		echo "$OUT_ROOT/$1"
	fi
}

activate_env() {
	source ~/.bashrc
	micromamba activate new-stack || exit 1
	export PYTHONPATH="$SCRIPT_REPO${PYTHONPATH:+:$PYTHONPATH}"
}

require_paths() {
	local path
	for path in "$@"; do
		if [[ ! -e "$path" ]]; then
			echo "Required path does not exist: $path" >&2
			exit 1
		fi
	done
}

# Fail on a cochlea name that is not in COCHLEAE, so a typo does not silently create a new
# output folder with an empty configuration.
require_known_cochlea() {
	local candidate
	for candidate in "${COCHLEAE[@]}"; do
		[[ "$1" == "$candidate" ]] && return 0
	done
	echo "Unknown cochlea: '$1'. Expected one of:" >&2
	printf '  %s\n' "${COCHLEAE[@]}" >&2
	exit 1
}
