#!/bin/bash
#SBATCH --job-name=eval-SGNv2var
#SBATCH --time=06:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Evaluate the four seed variants against the consensus annotations, reading everything from the
# workspace. Run this once all 20 tables exist.
#
# Deliberately not an array job: json_util.update_json reads the file, updates the top-level keys
# and rewrites the whole thing, so four concurrent writers would keep only one result.
#
# One cache folder per version is mandatory: the cache file name is '<cochlea>_<slice>.tif' with no
# version in it, and a cache hit short-circuits before the segmentation is read, so a shared folder
# would silently score the first version four times.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
activate_env
set -euo pipefail

SCRIPT=$SCRIPT_REPO/scripts/validation/SGNs/run_evaluation.py
# The accuracy JSON is the durable product of the experiment and is committed, so it stays in the
# repository rather than the workspace, which expires.
OUTPUT_DIR=$SCRIPT_REPO/reproducibility/model_accuracy

require_paths "$SCRIPT" "$OUTPUT_DIR"

for version in "${VERSIONS[@]}"; do
	for cochlea in "${COCHLEAE[@]}"; do
		require_paths "$(version_folder "$cochlea" "$version")/default_components.tsv"
	done
done

for version in "${VERSIONS[@]}"; do
	cache=$WS/eval_cache/SGN_v2-$version
	mkdir -p "$cache"
	echo "=== SGN_v2-$version ==="
	python "$SCRIPT" \
		--segmentation_name "SGN_v2-$version" \
		--local_root "$WS" \
		--cache_folder "$cache" \
		-o "$OUTPUT_DIR"
done

python "$SGN_VARIANCE_DIR"/check_results.py --accuracy_file "$OUTPUT_DIR/SGN_3D.json"
