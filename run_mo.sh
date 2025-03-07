#!/bin/bash

problems=('Pymoo/MO=zdt1' 'Pymoo/MO=zdt2' 'Pymoo/MO=zdt3' 'Pymoo/MO=zdt4' 'Pymoo/MO=zdt5' 'Pymoo/MO=zdt6' 'Pymoo/MO=kursawe' 'Pymoo/MO=omnitest' 'Pymoo/MO=sympart' 'Pymoo/MO=sympart_rotated' 'YAHPO/MO=cfg_iaml_glmnet_1067' 'YAHPO/MO=cfg_iaml_ranger_1489' 'YAHPO/MO=cfg_iaml_super_1489' 'YAHPO/MO=cfg_iaml_xgboost_40981' 'YAHPO/MO=cfg_lcbench_189873')
#problems=('MFPBench/MO/pd1=cifar100_wideresnet_2048' 'MFPBench/MO/pd1=imagenet_resnet_512' 'MFPBench/MO/pd1=lm1b_transformer_2048' 'MFPBench/MO/pd1=translatewmt_xformer_64')

#problems=('subselection/multiobjective/dev=subset_Pymoo_MO_omnitest' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_rf_167119' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_rf_31' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_svm_146212'
# 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_svm_146818' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_svm_31' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_xgboost_146822' 'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_xgboost_14965'
#  'subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_xgboost_167120' 'subselection/multiobjective/dev=subset_yahpo_mo_rbv2_rpart_40499_None')
#problems=('subselection/multiobjective/dev=subset_hpobench_MO_tab_ml_rf_167119')
seeds=('0' '1' '2' '3' '4')
for prob in "${problems[@]}"; do
  for seed in "${seeds[@]}"; do
    cmd="python -m carps.run hydra.searchpath=[pkg://hpi_parego/configs] +optimizer/smac20=multiobjective +customoptimizer=hpi_parego +problem/$prob seed=$seed" # task.n_trials=50"
    $cmd
  done
done

#optimizers=('smac20=multiobjective', 'smac20=multiobjective_gp', 'smac20=momf', 'smac20=momf_gp', 'optuna=multiobjective', 'syntune=BO_MO_LS', 'syntune=BO_MO_RS', 'syntune=MOREA')
optimizers=('smac20=multiobjective' 'optuna=multiobjective')
for prob in "${problems[@]}"; do
  for opt in "${optimizers[@]}"; do
    for seed in "${seeds[@]}"; do
      cmd="python -m carps.run +optimizer/$opt +problem/$prob seed=$seed" # task.n_trials=50"
      $cmd
    done
  done
done

cmd="python -m carps.analysis.gather_data runs"
$cmd



python -m carps.run hydra.searchpath=[pkg://hpi_parego/configs] +optimizer/smac20=multiobjective +customoptimizer=hpi_parego '+problem/YAHPO/MO=glob(*)' 'seed=range(0,5)' -m

python -m carps.run hydra.searchpath=[pkg://hpi_parego/configs] +optimizer/smac20=multiobjective +customoptimizer=hpi_parego '+problem/Pymoo/MO=glob(*)' 'seed=range(0,5)' -m


python -m carps.run hydra.searchpath=[pkg://hpi_parego/configs] +optimizer/smac20=multiobjective +customoptimizer=hpi_parego +problem/YAHPO/MO=cfg_iaml_ranger_1489 seed=0