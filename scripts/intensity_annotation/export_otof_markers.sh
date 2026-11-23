#!/usr/bin/env bash
#SBATCH --constraint=inet

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Slurm executes a copy of this file from /var/spool/slurmd. Use the
    # directory from which sbatch was invoked to find the repository files.
    SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi

TABLE_PATH="$SCRIPT_DIR/M_AMD_OTOF27_L_IHC_v11_edited.tsv"
PYTHON_BIN="${PYTHON_BIN:-/mnt/vast-nhr/home/pape41/u12086/Work/software/micromamba/envs/envs/new-stack/bin/python}"

[[ -f "$TABLE_PATH" ]] || { echo "Missing marker table: $TABLE_PATH" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python interpreter: $PYTHON_BIN" >&2; exit 1; }

export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -m flamingo_tools.export_data.cli marker \
    --cochlea M_AMD_OTOF27_L \
    --s3 \
    --scale 1 \
    --output_folder "$SCRIPT_DIR/otof27l-editing" \
    --channels IHC_v11 \
    --table "$TABLE_PATH" \
    --force
