#!/bin/bash
#SBATCH --job-name=hpi_parego
#SBATCH --output=hpi_parego%j.out
#SBATCH --error=hpi_parego%j.err

#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB

commit_hash=$(git log -1 --pretty=format:"%h")

module load Miniforge3
echo 'activating conda'

conda activate /bigwork/$USER/.conda/carps

export PYTHONPATH=/bigwork/$USER/hpi_parego/package_hpi_parego
echo "PYTHONPATH: $PYTHONPATH"



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
HYDRA_CMD="python -m carps.run hydra.searchpath=[file:///bigwork/$USER/hpi_parego/package_hpi_parego/hpi_parego/configs] +optimizer/smac20=multiobjective +customoptimizer=hpi_parego '+problem/YAHPO/MO=cfg_iaml_super_1489' 'seed=0'"
#HYDRA_CMD="python -m carps.run hydra.searchpath=[file:///bigwork/$USER/hpi_parego/package_hpi_parego/hpi_parego/configs] +optimizer/smac20=multiobjective +customoptimizer=hpi_parego '+problem/YAHPO/MO=glob(*)' 'seed=range(0,5)'"

# Add all Hydra overrides
for override in "${HYDRA_OVERRIDES[@]}"; do
    HYDRA_CMD+=" $override"
done

# Prepare final command with commit hash and multirun flag if specified
FINAL_CMD=$HYDRA_CMD
FINAL_CMD+=" -m"

# if [[ -n "$MULTIRUN_FLAG" ]]; then
#     FINAL_CMD+=" $MULTIRUN_FLAG"
# fi

# if [[ -n "$CFG_FLAG" ]]; then
#     FINAL_CMD+=" $CFG_FLAG"
# fi

# if [[ -n "$RESOLVE_FLAG" ]]; then
#     FINAL_CMD+=" $RESOLVE_FLAG"
# fi

export HYDRA_FULL_ERROR=1
echo "Executing: $FINAL_CMD"
eval "$FINAL_CMD"

wait
