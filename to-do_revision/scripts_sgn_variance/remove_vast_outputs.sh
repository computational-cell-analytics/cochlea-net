#!/bin/bash
# Remove the old SGN_v2-1..4 output folders on vast, once the experiment is complete in the
# workspace. This is the last step and it is deliberately manual.
#
#   bash remove_vast_outputs.sh            # checks the gates and prints what it would delete
#   bash remove_vast_outputs.sh --delete
#
# Two things to know before running it:
#   - those folders hold the only copies of mask.zarr and mean_std.json, so the workspace copies
#     must exist first. Gate 1 covers that indirectly by requiring the full set of results.
#   - they are owned by u15000 with group write on HPC_nim00007, so deletion is possible but should
#     be agreed with the owner. That is the main reason this is not automatic.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
activate_env
set -euo pipefail

DELETE=0
if [[ "${1:-}" == "--delete" ]]; then
	DELETE=1
fi

ACCURACY_FILE=$SCRIPT_REPO/reproducibility/model_accuracy/SGN_3D.json

# Gate 1: every result the experiment is supposed to produce exists in the workspace.
missing=0
for cochlea in "${COCHLEAE[@]}"; do
	for version in "${VERSIONS[@]}"; do
		table=$(version_folder "$cochlea" "$version")/default_components.tsv
		if [[ ! -s "$table" ]]; then
			echo "Missing or empty: $table" >&2
			missing=1
		fi
	done
done
if [[ "$missing" -eq 1 ]]; then
	echo "Refusing to continue: the workspace results are incomplete." >&2
	exit 1
fi
echo "Gate 1 passed: all 20 component tables exist."

# Gates 2 and 3: the accuracy file holds all four variants, they are in the expected range, and
# they are not all identical (which would mean the same segmentation was scored four times).
require_paths "$ACCURACY_FILE"
python "$SGN_VARIANCE_DIR"/check_results.py --accuracy_file "$ACCURACY_FILE"
echo "Gates 2 and 3 passed."

# The masks in the workspace are what makes the vast copies expendable.
for cochlea in "${COCHLEAE[@]}"; do
	require_paths "$(cochlea_folder "$cochlea")/mask.zarr" \
		"$(cochlea_folder "$cochlea")/mean_std.json"
done
echo "Gate 4 passed: mask.zarr and mean_std.json are staged for all cochleae."

echo
for cochlea in "${COCHLEAE[@]}"; do
	for version in "${VERSIONS[@]}"; do
		folder=$OLD_PREDICTION_ROOT/$cochlea/SGN_v2-$version
		[[ -e "$folder" ]] || continue
		size=$(du -sh "$folder" 2>/dev/null | cut -f1)
		if [[ "$DELETE" -eq 1 ]]; then
			echo "Deleting $folder ($size)"
			rm -rf "$folder"
		else
			echo "would delete: $folder ($size)"
		fi
	done
done

if [[ "$DELETE" -eq 0 ]]; then
	echo
	echo "Dry run. Re-run with --delete to actually remove these."
fi
