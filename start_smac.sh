#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -J "hydra_launcher"
#SBATCH -p normal
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8

module reset
module load lang/Miniforge3/24.1.2-0
source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate carps

export PYTHONPATH=/scratch/hpc-prf-intexml/daphne/hpi_parego:$PYTHONPATH


# Initialize an empty array to store Hydra overrides
HYDRA_OVERRIDES=()
MULTIRUN_FLAG=""
CFG_FLAG=""
RESOLVE_FLAG=""

# Parse all command-line arguments as Hydra overrides or check for --multirun flag
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--multirun" ]]; then
        MULTIRUN_FLAG="--multirun"
    elif [[ "$1" == "--cfg=all" ]]; then
        CFG_FLAG="--cfg=all"
    elif [[ "$1" == "--resolve" ]]; then
        RESOLVE_FLAG="--resolve"
    else
        HYDRA_OVERRIDES+=("$1")
    fi
    shift
done

# Construct the Hydra command
echo $CWD
HYDRA_CMD="python -m carps.run '+task/subselection/multiobjective/dev=glob(*)' +optimizer/smac20=multiobjective_rf 'seed=range(0,3)'"

# Add all Hydra overrides
for override in "${HYDRA_OVERRIDES[@]}"; do
    HYDRA_CMD+=" $override"
done

# Prepare final command with commit hash and multirun flag if specified
FINAL_CMD=$HYDRA_CMD
FINAL_CMD+=" -m"

export HYDRA_FULL_ERROR=1
echo "Executing: $FINAL_CMD"
eval "$FINAL_CMD"

wait