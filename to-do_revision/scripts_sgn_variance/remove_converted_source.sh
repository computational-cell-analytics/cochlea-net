#!/bin/bash
# Delete the unsharded predictions.zarr on vast for a cochlea whose prediction has been converted
# into the workspace. Verifies the conversion first and does nothing unless it passes.
#
#   bash remove_converted_source.sh                             # dry run, M_LR_000169_R
#   bash remove_converted_source.sh --n_samples 60              # deeper check, still a dry run
#   bash remove_converted_source.sh --delete
#
# This is narrower than remove_vast_outputs.sh: it removes only predictions.zarr, keeping mask.zarr
# and mean_std.json on vast, and it is gated on the conversion of this one cochlea rather than on
# the whole experiment being finished. Use it to get the 1.3 TB back before the rest of the
# experiment has run.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
activate_env
set -euo pipefail

DELETE=0
N_SAMPLES=20
while [[ $# -gt 0 ]]; do
	case "$1" in
		--delete) DELETE=1; shift ;;
		--n_samples) N_SAMPLES=$2; shift 2 ;;
		*) break ;;
	esac
done
COCHLEA=${1:-$CONVERTED_COCHLEA}

COCHLEA_FOLDER=$(cochlea_folder "$COCHLEA")
require_paths "$COCHLEA_FOLDER/shard_manifest.json"

failed=0
for version in "${VERSIONS[@]}"; do
	target=$(version_folder "$COCHLEA" "$version")
	source_prediction=$OLD_PREDICTION_ROOT/$COCHLEA/SGN_v2-$version/predictions.zarr

	if [[ ! -e "$source_prediction" ]]; then
		echo "SGN_v2-$version: already removed from vast."
		continue
	fi

	echo "=== SGN_v2-$version ==="
	# Checks that every shard the manifest expects exists, and that a sample of whole shards is
	# bit-for-bit equal to the original. Raise --n_samples for a stricter check; each sample reads
	# about 1.6 GB from either side.
	if ! python -m sgn_variance verify \
		--cochlea_folder "$COCHLEA_FOLDER" \
		--output_folder "$target" \
		--reference "$source_prediction" \
		--n_samples "$N_SAMPLES"; then
		echo "SGN_v2-$version: verification failed, keeping the original." >&2
		failed=1
		continue
	fi

	size=$(du -sh "$source_prediction" | cut -f1)
	if [[ "$DELETE" -eq 1 ]]; then
		echo "SGN_v2-$version: deleting $source_prediction ($size)"
		rm -rf "$source_prediction"
	else
		echo "SGN_v2-$version: would delete $source_prediction ($size)"
	fi
done

if [[ "$DELETE" -eq 0 ]]; then
	echo
	echo "Dry run. Re-run with --delete to actually remove these."
fi
exit "$failed"
