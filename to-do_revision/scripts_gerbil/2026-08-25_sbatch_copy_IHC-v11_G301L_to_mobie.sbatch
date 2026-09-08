#!/bin/bash
#SBATCH --job-name=mobie-IHCv11-G301L
#SBATCH --time=02:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G

set -euo pipefail

usage() {
    echo "Usage: $0 [--dry-run]"
}

dry_run=0
if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
elif [[ $# -eq 1 ]]; then
    case "$1" in
        --dry-run)
            dry_run=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
fi

# Final GLR301L IHC segmentation and its curated component selection.
source_dir="/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/prediction/G301L/IHC_v11_dilated_mask1"
source_segmentation="${source_dir}/segmentation.zarr"
source_key="segmentation"
source_table="${source_dir}/default_components.tsv"

mobie_root="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"
cochlea="G_LR_000301_L"
source_name="IHC_v11"
resolution="[0.38,0.38,0.38]"
scale_factors="[[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]"
chunks="[64,64,64]"

target_dataset="${mobie_root}/${cochlea}"
target_image="${target_dataset}/images/ome-zarr/${source_name}.ome.zarr"
target_table_dir="${target_dataset}/tables/${source_name}"
target_table="${target_table_dir}/default.tsv"
generated_table="${target_table_dir}/default_mobie-generated.tsv"

expected_objects=1182
expected_selected=946

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -f "${source_segmentation}/zarr.json" ]] || fail "Missing source Zarr metadata: ${source_segmentation}/zarr.json"
[[ -f "${source_segmentation}/${source_key}/zarr.json" ]] || fail "Missing source array: ${source_segmentation}/${source_key}"
[[ -f "${source_table}" ]] || fail "Missing finalized component table: ${source_table}"
[[ -f "${mobie_root}/project.json" ]] || fail "Not a MoBIE project: ${mobie_root}"

# This is deliberately create-only. Never overwrite an existing dataset or a
# partial prior attempt without inspecting it first.
[[ ! -e "${target_dataset}" ]] || fail "Target dataset already exists: ${target_dataset}"

IFS=$'\t' read -r -a table_columns < "${source_table}"
component_column=0
for i in "${!table_columns[@]}"; do
    if [[ "${table_columns[$i]}" == "component_labels" ]]; then
        component_column=$((i + 1))
        break
    fi
done
[[ ${component_column} -gt 0 ]] || fail "The source table has no component_labels column"

object_count=$(awk 'END { print NR - 1 }' "${source_table}")
selected_count=$(awk -F $'\t' -v column="${component_column}" 'NR > 1 && $column + 0 > 0 { count++ } END { print count + 0 }' "${source_table}")
[[ "${object_count}" -eq "${expected_objects}" ]] || fail "Expected ${expected_objects} objects, found ${object_count}"
[[ "${selected_count}" -eq "${expected_selected}" ]] || fail "Expected ${expected_selected} selected IHCs, found ${selected_count}"

mobie_command=(
    mobie.add_segmentation
    --input_path "${source_segmentation}"
    --input_key "${source_key}"
    --root "${mobie_root}"
    --dataset_name "${cochlea}"
    --name "${source_name}"
    --resolution "${resolution}"
    --scale_factors "${scale_factors}"
    --chunks "${chunks}"
)

echo "Validated source: ${object_count} objects, ${selected_count} selected IHCs"
echo "Target: ${target_dataset}"

if [[ ${dry_run} -eq 1 ]]; then
    echo "DRY RUN: no files will be copied and no MoBIE metadata will be changed."
    printf 'Would run:'
    printf ' %q' "${mobie_command[@]}"
    printf '\n'
    echo "Would preserve MoBIE's generated table as: ${generated_table}"
    echo "Would install the finalized table as: ${target_table}"
    exit 0
fi

# The activation is intentionally after the dry-run exit, so a dry run cannot
# call MoBIE or mutate the project even if the environment is configured badly.
source ~/.bashrc
mobie_env="${MOBIE_ENV:-mobie}"
micromamba activate "${mobie_env}"
command -v mobie.add_segmentation >/dev/null || fail "mobie.add_segmentation is unavailable in environment ${mobie_env}"

"${mobie_command[@]}"

[[ -d "${target_image}" ]] || fail "MoBIE did not create the expected image: ${target_image}"
[[ -f "${target_table}" ]] || fail "MoBIE did not create the expected table: ${target_table}"

generated_count=$(awk 'END { print NR - 1 }' "${target_table}")
[[ "${generated_count}" -eq "${expected_objects}" ]] || fail "Generated MoBIE table has ${generated_count} rows; expected ${expected_objects}"

# Keep the automatically generated table for provenance, then make the curated
# component labels the default table visible in MoBIE.
install -m 0660 "${target_table}" "${generated_table}"
finalized_tmp="${target_table}.tmp.${SLURM_JOB_ID:-$$}"
install -m 0660 "${source_table}" "${finalized_tmp}"
mv "${finalized_tmp}" "${target_table}"

echo "Added ${cochlea}/${source_name} to the local MoBIE project."
echo "The finalized default table selects ${selected_count} IHCs."
echo "Inspect the local dataset before syncing it to object storage."
