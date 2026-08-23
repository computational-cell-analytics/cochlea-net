# Shared configuration for the SGN v2 seed-variance experiment.
# Sourced by every sbatch script in this folder; not meant to be executed on its own.

SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
SGN_VARIANCE_DIR=$SCRIPT_REPO/to-do_revision/scripts_sgn_variance
# Holds distance_unet_checkpoint.py, needed to load these checkpoints in the new-stack env.
CHECKPOINT_HELPER_DIR=$SCRIPT_REPO/to-do_revision/scripts_synapses
GERBIL_DIR=$SCRIPT_REPO/to-do_revision/scripts_gerbil

# Everything this experiment writes lives in the workspace.
WS=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/SGN-v2-variance
# The raw data, the models and the predictions of the earlier runs stay read-only on vast.
DATA_ROOT=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet
MODEL_ROOT=$DATA_ROOT/trained_models/SGN
OLD_PREDICTION_ROOT=$DATA_ROOT/predictions

COCHLEAE=(M_AMD_000058_L M_LR_000169_R M_LR_000226_L M_LR_000227_L M_LR_000227_R)
VERSIONS=(1 2 3 4)
PREDICTION_INSTANCES=10

# The cochlea whose predictions already exist and are converted instead of recomputed.
CONVERTED_COCHLEA=M_LR_000169_R

# In-mask 128^3 block counts of the masks the existing predictions were computed with. The staging
# step fails if a copied mask does not reproduce these, which would mean the inputs changed.
declare -A EXPECT_BLOCKS=(
	[M_AMD_000058_L]=11380
	[M_LR_000169_R]=16327
	[M_LR_000226_L]=9627
	[M_LR_000227_L]=12064
	[M_LR_000227_R]=12309
)

model_path() {
	echo "$MODEL_ROOT/v2-$1_cochlea_distance_unet_SGN_supervised"
}

raw_path() {
	echo "$DATA_ROOT/$1/PV.ome.zarr"
}

# Folder holding mask.zarr, mean_std.json and shard_manifest.json for one cochlea.
cochlea_folder() {
	echo "$WS/$1"
}

# Folder holding predictions.zarr / segmentation.zarr / the tables for one cochlea and version.
version_folder() {
	echo "$WS/$1/SGN_v2-$2"
}

activate_env() {
	source ~/.bashrc
	micromamba activate new-stack || exit 1
	export PYTHONPATH="$SGN_VARIANCE_DIR:$CHECKPOINT_HELPER_DIR:$SCRIPT_REPO${PYTHONPATH:+:$PYTHONPATH}"
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
