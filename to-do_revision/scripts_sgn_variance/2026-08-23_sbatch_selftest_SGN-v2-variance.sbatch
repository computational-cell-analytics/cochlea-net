#!/bin/bash
#SBATCH --job-name=selftest-SGNv2var
#SBATCH --time=00:30:00
#SBATCH --partition=grete:shared
#SBATCH --account=nim00007
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Check that predicting one shard at a time reproduces a whole-volume run exactly. Usage:
#   sbatch 2026-08-23_sbatch_selftest_SGN-v2-variance.sbatch [cochlea] [version]
#
# Run this once, after staging and before the first apply job: it is the evidence that the sharded
# per-shard prediction is numerically the same thing as the unsharded whole-volume prediction the
# reference SGN_v2 accuracy was measured on.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
activate_env
set -euo pipefail

COCHLEA=${1:-M_LR_000226_L}
VERSION=${2:-1}

COCHLEA_FOLDER=$(cochlea_folder "$COCHLEA")
require_paths "$COCHLEA_FOLDER/shard_manifest.json" "$(raw_path "$COCHLEA")" \
	"$(model_path "$VERSION")"

echo "Cochlea: $COCHLEA, version: $VERSION"

python -m sgn_variance selftest \
	--input "$(raw_path "$COCHLEA")" --input_key s0 \
	--cochlea_folder "$COCHLEA_FOLDER" \
	--model "$(model_path "$VERSION")"
