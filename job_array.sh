#!/bin/bash
#SBATCH --array 0%1
#SBATCH --job-name hpi_parego
#SBATCH --partition H100
#SBATCH --gpus 1
#BATCH ----cpus-per-gpu 8
#SBATCH --time=1-00:00
#SBATCH --mem=50G

seeds=("0" "0" "0" "0" "0" "1" "1" "1" "1" "1" "2" "2" "2" "2" "2" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4")

problems=(
  "subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_rf_167119"
)

#problems=('subselection/multiobjective/dev=subset_Pymoo_MO_omnitest' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_rf_167119' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_rf_31' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_svm_146212'
# 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_svm_146818' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_svm_31' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_xgboost_146822' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_xgboost_14965'
#  'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_xgboost_167120' 'subselection/multiobjective/dev=subset_yahpo_mo_rbv2_rpart_40499_None')

# Get the command suffix for the current array index
problem=${problems[$SLURM_ARRAY_TASK_ID - 1]}
seed=${seeds[$SLURM_ARRAY_TASK_ID - 1]}
echo ${problem}
echo ${seed}

srun \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=/netscratch/theodorakopoulos:/netscratch/theodorakopoulos,"`pwd`":"`pwd`" \
  --task-prolog="`pwd`"/package_hpi_parego/install.sh \
   python -m carps.run +optimizer/smac20=multiobjective +problem/${problem} seed=${seed}

#   cmd="python -m carps.analysis.gather_data runs"
