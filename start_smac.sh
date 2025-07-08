#!/bin/bash
#SBATCH -t 3:00:00
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
HYDRA_CMD="python -m carps.run '+task/subselection/multiobjective/dev=subset_hpobench_multiobjective_tabular_ml_nn_3917' +optimizer/smac20=multiobjective_rf 'seed=0' baserundir=delete"
# HYDRA_CMD="python -m carps.run '+task/subselection/multiobjective/dev=glob(*)' +optimizer/randomsearch=config 'seed=range(0,3)' baserundir=results_random_search "
# task = ['subset_hpobench_multiobjective_tabular_ml_nn_3917', 
#         'subset_hpobench_multiobjective_tabular_ml_rf_168911', 
#         'subset_hpobench_multiobjective_tabular_ml_svm_168911',
#         'subset_hpobench_multiobjective_tabular_ml_xgboost_3917', 
#         'subset_Pymoo_ManyO_unconstraint_dtlz7', 
#         'subset_Pymoo_MO_unconstraint_kursawe',
#         'subset_yahpo_mo_iaml_glmnet_1489_None', 
#         'subset_yahpo_mo_iaml_ranger_1489_None', 
#         'subset_yahpo_mo_rbv2_xgboost_28_None']

# ['task/HPOBench/multiobjective/tabular/ml=glob(*)', 'task/Pymoo/MO/unconstraint=glob(*)', 'task/Pymoo/ManyO/unconstraint=glob(*)', 'task/YAHPO/MO=glob(*)']
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