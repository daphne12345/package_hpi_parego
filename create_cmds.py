import pandas as pd
from itertools import product
import subprocess
import time

hpi_method = ['fanova']
adjust_cs = ['default', 'random', 'incumbent', 'no']
cs_proba_hpi = ['false','true']

adjust_previous_cfgs = ['no', 'true_no_retrain', 'true_retrain'] 
set_to = ['random', 'default', 'incumbent']
gt_hpi = ['false']


# thresh = [('0-up-0','[0.0,0.2,0.4,0.6,0.8,0.0]'),  ('0-075-0','[0.0,0.75,0.75,0.75,0.0]'), ('0-down-0','[0.0,0.9,0.7,0.5,0.3,0.0]'), ('down-0','[0.9,0.7,0.5,0.3,0.0]'), ('0-down','[0.0,0.9,0.7,0.5,0.3]'), 
#           ('0-down-0-fine','[0.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]'), ('075','0.75'), ('cos','[0.9,0.8780,0.8130,0.7135,0.5891,0.4500,0.3109,0.1865,0.0870,0.0220,0.9,0.8780,0.8130,0.7135,0.5891,0.4500,0.3109,0.1865,0.0870,0.0220]')]
# thresh = [('0-075','[0.0,0.75]'), ('0-down-half' ,'[0.0,0.0,0.0,0.0,0.8,0.6,0.4,0.0]')]
# thresh = [('0-down-0','[0.0,0.9,0.7,0.5,0.3,0.0]')]
# thresh = [('0-05','[0.0,0.5]'), ('0-75','[0.0,0.75]'), ('075','0.75'),('0-down-0','[0.0,0.9,0.7,0.5,0.3,0.0]'), ('0-up-0','[0.0,0.2,0.4,0.6,0.8,0.0]'))]
thresh = [('075','0.75')]

task = ['subset_hpobench_multiobjective_tabular_ml_nn_3917', 
        'subset_hpobench_multiobjective_tabular_ml_rf_168911', 
        'subset_hpobench_multiobjective_tabular_ml_svm_168911',
        'subset_hpobench_multiobjective_tabular_ml_xgboost_3917', 
        'subset_Pymoo_ManyO_unconstraint_dtlz7', 
        'subset_Pymoo_MO_unconstraint_kursawe',
        'subset_yahpo_mo_iaml_glmnet_1489_None', 
        'subset_yahpo_mo_iaml_ranger_1489_None', 
        'subset_yahpo_mo_rbv2_xgboost_28_None']

# task = ['subset_Pymoo_ManyO_unconstraint_dtlz1', 
# 'subset_Pymoo_ManyO_unconstraint_dtlz5', 
# 'subset_Pymoo_ManyO_unconstraint_dtlz6', 
# 'subset_Pymoo_ManyO_unconstraint_dtlz7', 
# 'subset_Pymoo_ManyO_unconstraint_wfg2_10_5', 
# 'subset_Pymoo_ManyO_unconstraint_wfg3_10_5', 
# 'subset_Pymoo_MO_unconstraint_kursawe', 
# 'subset_Pymoo_MO_unconstraint_zdt1', 
# 'subset_Pymoo_MO_unconstraint_zdt2', 
# 'subset_Pymoo_MO_unconstraint_zdt6']



# Generate all combinations
combinations = list(product(
    hpi_method,
    adjust_cs,
    adjust_previous_cfgs,
    cs_proba_hpi,
    set_to,
    thresh,
    gt_hpi,
    task,
))

columns = [
    'hpi_method', 'adjust_cs', 'adjust_previous_cfgs',
    'cs_proba_hpi', 'set_to',
    'thresh',
    'gt_hpi',
    'task'
]
df = pd.DataFrame(combinations, columns=columns)

def remove(row, li):
    for col in li:
        row[col] = None
    return row

df[df['adjust_cs']=='no'] = df[df['adjust_cs']=='no'].apply(lambda row: remove(row, ['cs_proba_hpi']), axis=1)
df[(df['adjust_previous_cfgs']=='no')] = df[(df['adjust_previous_cfgs']=='no')].apply(lambda row: remove(row, ['set_to']), axis=1)
df = df[~((df['adjust_cs']=='no') & (df['adjust_previous_cfgs']=='no'))]
# df = df[~((df['cs_proba_hpi']=='true')&(df['hpi_method']=='hypershap'))]

df = df.drop_duplicates()


commands = []

for _, row in df.iterrows():
    # Construct the command dynamically, skipping None values
    #command = "python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:////scratch/hpc-prf-intexml/daphne/hpi_parego/package_hpi_parego/hpi_parego/configs] +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)'"
    command = "sbatch start_create_cmds.sh "
    command += f" '+task/subselection/multiobjective/dev={row['task']}'"

    baserundir = "results_new_ablation/"
    
    command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method={row['hpi_method']}"
    baserundir += f"{row['hpi_method']}/"
    
    command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs={row['adjust_cs']}"
    baserundir += f"adjust_cs_{row['adjust_cs']}"
    
    if row['cs_proba_hpi'] is not None:
        command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi={row['cs_proba_hpi']}"
        if row['cs_proba_hpi'] == 'true':    
            baserundir += "_cs_proba_hpi"
    
    command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs={row['adjust_previous_cfgs']}"
    baserundir += f"_adjust_prev_cfgs_{row['adjust_previous_cfgs']}"
    
    if row['set_to'] is not None:
        command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.set_to={row['set_to']}"
        baserundir += f"_set_to_{row['set_to']}"
    
    command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh={row['thresh'][1]}"
    baserundir += f"_thresh_{row['thresh'][0]}"
    
    command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.gt_hpi={row['gt_hpi']}"
    if row['gt_hpi'] == 'true':    
        baserundir += "_gt_hpi"
        
        
    command += f" baserundir={baserundir}"
    command += f" optimizer_id={row['hpi_method']}_{baserundir.split('/')[-1]}"
    
    commands.append(command)

# commands.reverse()
for command in commands:
    process = subprocess.run(command, shell=True)
   
    if process.returncode != 0:
        print(f"Command failed: {command}")
    time.sleep(1)
print('command length', len(commands))
print('all commands executed')
exit()

print(len(commands))
# import subprocess
# from concurrent.futures import ThreadPoolExecutor, as_completed


# max_workers = 16

# def run_command(cmd):
#     result = subprocess.run(cmd, shell=True)
#     if result.returncode != 0:
#         print(f"Command failed: {cmd}")
#     return result.returncode


# with ThreadPoolExecutor(max_workers=max_workers) as executor:
#     futures = [executor.submit(run_command, cmd) for cmd in commands]

#     for future in as_completed(futures):
#         # Optional: check for exceptions here
#         future.result()

# print(f"[ThreadPoolExecutor] Completed {len(commands)} commands")


# python -m carps.experimenter.create_cluster_configs +optimizer/smac20=multiobjective_rf 'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' -m
# python -m carps.experimenter.create_cluster_configs +optimizer/smac20=multiobjective_gp 'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' -m
# python -m carps.experimenter.create_cluster_configs +optimizer/optuna=MO_NSGAII 'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' -m
# python -m carps.experimenter.create_cluster_configs +optimizer/optuna=MO_TPE 'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' -m

# python -m carps.experimenter.create_cluster_configs +optimizer/synetune=BO_MO_LS 'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' -m
# python -m carps.experimenter.create_cluster_configs +optimizer/synetune=BO_MO_RS 'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' -m
# python -m carps.experimenter.create_cluster_configs +optimizer/synetune=MOREA 'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' -m





# # best fanova auc: fanova_adjust_cs_default_constant_adjust_prev_cfgs_set_to_incumbent_thresh_0.75
# python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:////scratch/hpc-prf-intexml/daphne/hpi_parego/package_hpi_parego/hpi_parego/configs] 
# +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=fanova optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method=default optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi=false optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.set_to=incumbent optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=none 
# 'optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh=0.75' 
# baserundir=results/fanova/fanova_adjust_cs_default_constant_adjust_prev_cfgs_set_to_incumbent_thresh_0.75 optimizer_id=fanova_adjust_cs_default_constant_adjust_prev_cfgs_set_to_incumbent_thresh_0.75 -m

# # best fanova final: fanova_adjust_cs_incumbent_cs_proba_hpi_thresh_0.5
# python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:////scratch/hpc-prf-intexml/daphne/hpi_parego/package_hpi_parego/hpi_parego/configs] 
# +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=fanova optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method=incumbent optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant=false 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi=true optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=false 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=none 
# 'optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh=0.5' 
# baserundir=results/fanova/fanova_adjust_cs_incumbent_cs_proba_hpi_thresh_0.5 optimizer_id=fanova_adjust_cs_incumbent_cs_proba_hpi_thresh_0.5 -m

# python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:////scratch/hpc-prf-intexml/daphne/hpi_parego/package_hpi_parego/hpi_parego/configs] 
# +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=random optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method=incumbent optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant=false 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi=true optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=false 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=none 
# 'optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh=0.5' 
# baserundir=results/random/random_adjust_cs_incumbent_cs_proba_hpi_thresh_0.5 optimizer_id=random_adjust_cs_incumbent_cs_proba_hpi_thresh_0.5 -m

# # best hypershap auc: hypershap_adjust_cs_default_constant_adjust_prev_cfgs_set_to_incumbent_dynamic_decay_down
# python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:////scratch/hpc-prf-intexml/daphne/hpi_parego/package_hpi_parego/hpi_parego/configs] 
# +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=hypershap optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method=default optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi=false optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.set_to=incumbent optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=linear 
# 'optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh_list=[0.9,0.8,0.7,0.6,0.5,0.4,0.3]' 
# baserundir=results/hypershap/hypershap_adjust_cs_default_constant_adjust_prev_cfgs_set_to_incumbent_dynamic_decay_down optimizer_id=hypershap_adjust_cs_default_constant_adjust_prev_cfgs_set_to_incumbent_dynamic_decay_down -m

# python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:////scratch/hpc-prf-intexml/daphne/hpi_parego/package_hpi_parego/hpi_parego/configs] 
# +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=random optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method=default optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi=false optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.set_to=incumbent optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=linear 
# 'optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh_list=[0.9,0.8,0.7,0.6,0.5,0.4,0.3]' 
# baserundir=results/random/random_adjust_cs_default_constant_adjust_prev_cfgs_set_to_incumbent_dynamic_decay_down optimizer_id=random_adjust_cs_default_constant_adjust_prev_cfgs_set_to_incumbent_dynamic_decay_down -m

# ---
# # best hypershap final: hypershap_adjust_cs_random_adjust_prev_cfgs_set_to_random_thresh_0.75
# python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:////scratch/hpc-prf-intexml/daphne/hpi_parego/package_hpi_parego/hpi_parego/configs] 
# +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=hypershap optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method=random optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant=false 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi=false optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.set_to=random optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=none 
# 'optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh=0.75' 
# baserundir=results/hypershap/hypershap_adjust_cs_random_adjust_prev_cfgs_set_to_random_thresh_0.75 optimizer_id=hypershap_adjust_cs_random_adjust_prev_cfgs_set_to_random_thresh_0.75 -m

# python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:////scratch/hpc-prf-intexml/daphne/hpi_parego/package_hpi_parego/hpi_parego/configs] 
# +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)' '+task/subselection/multiobjective/test=glob(*)' 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=random optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method=random optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant=false 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi=false optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.set_to=random optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=none 
# 'optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh=0.75' 
# baserundir=results/random/random_adjust_cs_random_adjust_prev_cfgs_set_to_random_thresh_0.75 optimizer_id=random_adjust_cs_random_adjust_prev_cfgs_set_to_random_thresh_0.75 -m




# #fanova even better?
# python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:////scratch/hpc-prf-intexml/daphne/hpi_parego/package_hpi_parego/hpi_parego/configs] 
# +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)' '+task/subselection/multiobjective/dev=glob(*)' 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=fanova optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method=default optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi=false optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.set_to=incumbent optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=linear 
# 'optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh_list=[0,0.8,0.6,0.4,0.2,0]' 
# baserundir=results/fanova/fanova_adjust_cs_default_constant_adjust_prev_cfgs_set_to_incumbent_thresh_0_lin_0 optimizer_id=fanova_adjust_cs_default_constant_adjust_prev_cfgs_set_to_incumbent_thresh_0_lin_0 -m

# python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:////scratch/hpc-prf-intexml/daphne/hpi_parego/package_hpi_parego/hpi_parego/configs] 
# +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)' '+task/subselection/multiobjective/dev=glob(*)' 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=fanova optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method=incumbent optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant=false 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi=true optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.set_to=random optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=linear 
# 'optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh_list=[0,0.8,0.6,0.4,0.2,0]' 
# baserundir=results/fanova/fanova_adjust_cs_incumbent_cs_proba_hpi_adjust_prev_cfgs_set_to_random_thresh_0_lin_0 optimizer_id=fanova_adjust_cs_incumbent_cs_proba_hpi_adjust_prev_cfgs_set_to_random_thresh_0_lin_0 -m

# python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:////scratch/hpc-prf-intexml/daphne/hpi_parego/package_hpi_parego/hpi_parego/configs] 
# +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)' '+task/subselection/multiobjective/dev=glob(*)' 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=fanova optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method=default optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi=false optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=true 
# optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.set_to=random optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=linear 
# 'optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh_list=[0,0.8,0.6,0.4,0.2,0]' 
# baserundir=results/fanova/fanova_adjust_cs_default_constant_adjust_prev_cfgs_set_to_random_thresh_0_lin_0 optimizer_id=fanova_adjust_cs_default_constant_adjust_prev_cfgs_set_to_random_thresh_0_lin_0 -m