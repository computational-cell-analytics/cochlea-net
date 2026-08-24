#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/../export_data/export_lower_resolution_marker.py" \
    --cochlea M_AMD_OTOF27_L \
    --scale 1 \
    --output_folder "$SCRIPT_DIR/otof27l-editing" \
    --channels IHC_v11 \
    --table "$SCRIPT_DIR/M_AMD_OTOF27_L_IHC_v11_edited.tsv" \
    --overwrite
