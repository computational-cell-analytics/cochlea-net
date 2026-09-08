#!/bin/bash
# Delete predictions.zarr for the versions of a cochlea whose tables are complete.
#
#   bash cleanup_predictions.sh M_LR_000226_L            # dry run, prints what it would delete
#   bash cleanup_predictions.sh --delete M_LR_000226_L
#
# The predictions are 190-330 GB each and are a pure intermediate: the evaluation reads only
# segmentation.zarr and default_components.tsv. Re-tuning the watershed thresholds would mean
# re-predicting, which is the accepted cost of not keeping 4.9 TB around.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
activate_env
set -euo pipefail

DELETE=0
if [[ "${1:-}" == "--delete" ]]; then
	DELETE=1
	shift
fi

if [[ $# -eq 0 ]]; then
	echo "Usage: bash cleanup_predictions.sh [--delete] <cochlea> [<cochlea> ...]" >&2
	exit 1
fi

for COCHLEA in "$@"; do
	for version in "${VERSIONS[@]}"; do
		folder=$(version_folder "$COCHLEA" "$version")
		prediction=$folder/predictions.zarr
		label="$COCHLEA SGN_v2-$version"

		if [[ ! -e "$prediction" ]]; then
			echo "$label: no predictions.zarr, nothing to do."
			continue
		fi

		# A corrupt or truncated table must not authorise deleting a few hundred GB, so check the
		# content of the components table and not just that the file is there.
		if ! python - "$folder" <<'PYEOF'
import sys
import os
import pandas as pd

folder = sys.argv[1]
if not os.path.exists(os.path.join(folder, "segmentation.zarr", "segmentation", "zarr.json")):
    raise SystemExit("no segmentation.zarr/segmentation")
if not os.path.getsize(os.path.join(folder, "default.tsv")):
    raise SystemExit("default.tsv is empty")
table = pd.read_csv(os.path.join(folder, "default_components.tsv"), sep="\t")
if "component_labels" not in table.columns:
    raise SystemExit("default_components.tsv has no component_labels column")
if len(table) < 1000:
    raise SystemExit(f"default_components.tsv has only {len(table)} rows")
# Gate on the size of component 1, which is the helix and the only thing the evaluation uses, not
# on its share of all objects. The four seed variants of M_LR_000169_R put 72% to 92% of their
# objects in component 1 while agreeing on its size to within 2%: they differ in how many strays
# they produce away from the helix, and those are exactly what the component step discards. A
# fraction test would have rejected the 72% run. The 25% floor is only here to catch a spiral that
# got shattered into many comparable pieces, which is a real failure mode.
in_component_1 = int((table.component_labels == 1).sum())
fraction = (table.component_labels == 1).mean()
if in_component_1 < 1000:
    raise SystemExit(f"only {in_component_1} objects in component 1")
if fraction < 0.25:
    raise SystemExit(f"only {fraction:.1%} of the objects are in component 1")
print(f"{len(table)} objects, {in_component_1} in component 1 ({fraction:.1%})")
PYEOF
		then
			echo "$label: NOT safe to delete (see above)." >&2
			continue
		fi

		size=$(du -sh "$prediction" | cut -f1)
		if [[ "$DELETE" -eq 1 ]]; then
			echo "$label: deleting $prediction ($size)"
			# rm -rf follows a trailing slash into the target of a symlink, so unlink links.
			if [[ -L "$prediction" ]]; then rm "$prediction"; else rm -rf "$prediction"; fi
			# seeds.zarr is a pure watershed intermediate as well.
			rm -rf "$folder/seeds.zarr"
		else
			echo "$label: would delete $prediction ($size) and $folder/seeds.zarr"
		fi
	done
done

if [[ "$DELETE" -eq 0 ]]; then
	echo
	echo "Dry run. Re-run with --delete to actually remove these."
fi
