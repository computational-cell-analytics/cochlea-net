#!/bin/bash
#SBATCH --job-name=bench-syn-slice
#SBATCH --time=00:40:00
#SBATCH --partition=grete:preemptible
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Measure whether a synapse prediction task fits on a MIG slice, and how fast it is there.
# The slice is not set here: pass it on the command line so the same script can be run against
# each candidate.
#
#   sbatch --gpus=1g.10gb:1 -J bench-10gb 2026-08-24_sbatch_benchmark_slice.sbatch
#   sbatch --gpus=1g.20gb:1 -J bench-20gb 2026-08-24_sbatch_benchmark_slice.sbatch
#   sbatch -p grete:shared --gpus=A100:1 -J bench-a100 2026-08-24_sbatch_benchmark_slice.sbatch
#
# A second argument is passed through to --instances, to make the test longer or shorter.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_syn_gerbil/common.sh
activate_env
set -euo pipefail

COCHLEA=${1:-G_LR_000301_L}
INSTANCES=${2:-50}

cd "$SYN_GERBIL_DIR"
echo "Node: ${SLURM_NODELIST}, gpus: ${SLURM_GPUS:-unset}"
python benchmark_slice.py --cochlea "$COCHLEA" --instances "$INSTANCES" \
	--scratch "$WS/mig-benchmark/${SLURM_JOB_ID}"
