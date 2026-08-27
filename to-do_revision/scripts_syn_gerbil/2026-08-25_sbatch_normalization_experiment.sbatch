#!/bin/bash
#SBATCH --partition=grete:interactive
#SBATCH --gpus=1g.20gb:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=inet
#SBATCH --output=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/synapse-normalization-v3/slurm-%j.out

# Run one or more cochleae serially on a single inexpensive A100 MIG slice. Two instances
# of this script cover all four cochleae, keeping the experiment to two distinct jobs.

source /mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_syn_gerbil/common.sh
activate_env
set -euo pipefail

MANIFEST=/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_syn_gerbil/normalization_blocks.json
OUTPUT_ROOT=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/synapse-normalization-v3
RUNNER=/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_syn_gerbil/normalization_experiment.py

if (( $# == 0 )); then
    echo "Pass at least one cochlea name." >&2
    exit 2
fi

echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not visible}"
echo "Cochleae: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "$RUNNER" \
    --manifest "$MANIFEST" \
    --output-root "$OUTPUT_ROOT" \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --cochleae "$@"
