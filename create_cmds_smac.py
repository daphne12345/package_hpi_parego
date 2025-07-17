import pandas as pd
from itertools import product
import subprocess
import time

# tasks = ['kursawe', 'omnitest', 'sympart_rotated', 'sympart', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'] # '+task/Pymoo/MO/unconstraint={task}'
# tasks = ['convex_dtlz2', 'convex_dtlz4', 'dtlz1_inv', 'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7', 'sdtlz1', 'wfg1_10_5','wfg2_10_5', 'wfg3_10_5', 'wfg4_10_5', 'wfg5_10_5', 'wfg6_10_5', 'wfg7_10_5', 'wfg8_10_5', 'wfg9_10_5'] # '+task/Pymoo/ManyO/unconstraint={task}'
# tasks = ['subset_hpobench_multiobjective_tabular_ml_lr_53', 'subset_hpobench_multiobjective_tabular_ml_lr_9952', 'subset_hpobench_multiobjective_tabular_ml_lr_9977', 'subset_hpobench_multiobjective_tabular_ml_lr_10101', 
# 'subset_hpobench_multiobjective_tabular_ml_lr_146212', 'subset_hpobench_multiobjective_tabular_ml_lr_146606', 'subset_hpobench_multiobjective_tabular_ml_lr_146818', 'subset_hpobench_multiobjective_tabular_ml_lr_146821', 
# 'subset_hpobench_multiobjective_tabular_ml_lr_146822', 'subset_hpobench_multiobjective_tabular_ml_lr_167119', 'subset_hpobench_multiobjective_tabular_ml_lr_167120', 'subset_hpobench_multiobjective_tabular_ml_nn_3917', 
# 'subset_hpobench_multiobjective_tabular_ml_nn_10101', 'subset_hpobench_multiobjective_tabular_ml_nn_146821', 'subset_hpobench_multiobjective_tabular_ml_rf_3', 'subset_hpobench_multiobjective_tabular_ml_rf_9952', 
# 'subset_hpobench_multiobjective_tabular_ml_rf_146212', 'subset_hpobench_multiobjective_tabular_ml_rf_167119', 'subset_hpobench_multiobjective_tabular_ml_rf_167120', 'subset_hpobench_multiobjective_tabular_ml_rf_168911',
# 'subset_hpobench_multiobjective_tabular_ml_svm_3', 'subset_hpobench_multiobjective_tabular_ml_svm_31', 'subset_hpobench_multiobjective_tabular_ml_svm_53', 'subset_hpobench_multiobjective_tabular_ml_svm_3917', 
# 'subset_hpobench_multiobjective_tabular_ml_svm_9952', 'subset_hpobench_multiobjective_tabular_ml_svm_9977', 'subset_hpobench_multiobjective_tabular_ml_svm_14965', 'subset_hpobench_multiobjective_tabular_ml_svm_146822',
# 'subset_hpobench_multiobjective_tabular_ml_svm_168911', 'subset_hpobench_multiobjective_tabular_ml_xgboost_3917', 'subset_hpobench_multiobjective_tabular_ml_xgboost_7592', 'subset_hpobench_multiobjective_tabular_ml_xgboost_9952',
# 'subset_hpobench_multiobjective_tabular_ml_xgboost_146606', 'subset_hpobench_multiobjective_tabular_ml_xgboost_146822', 'subset_hpobench_multiobjective_tabular_ml_xgboost_167120', 'subset_yahpo_mo_iaml_glmnet_1489_None', 'subset_yahpo_mo_iaml_ranger_1489_None',
# 'subset_yahpo_mo_rbv2_ranger_375_None', 'subset_yahpo_mo_rbv2_xgboost_28_None', 'subset_yahpo_mo_rbv2_xgboost_182_None']
commands = []
tasks = ['time', 'fair_loss']
seeds = range(0,10)

for seed in seeds:
    for task in tasks:
        command = f"sbatch start_smac.sh '+task/my_tasks={task}' baserundir=my_tasks/ +optimizer/optuna=MO_TPE seed={seed}"

        commands.append(command)

for command in commands:
    process = subprocess.run(command, shell=True)
   
    if process.returncode != 0:
        print(f"Command failed: {command}")
    time.sleep(0.1)
print('command length', len(commands))
print('all commands executed')
exit()

print(len(commands))
