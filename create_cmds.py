import pandas as pd
from itertools import product
import subprocess
import time

hpi_method = ['hypershap', 'fanova', 'random']
adjust_cs = ['true', 'false']
adjust_previous_cfgs = ['true', 'false']
dynamic_decay = ['linear', 'none']

adjust_cs_method = ['default', 'random', 'incumbent']
constant = ['true', 'false']
cs_proba_hpi = ['true', 'false']

set_to = ['random', 'default', 'incumbent']

thresh_list = ['[0.9,0.8,0.7,0.6,0.5,0.4,0.3]', '[0.3,0.4,0.5,0.6,0.7,0.8,0.9]']
thresh = [0.5, 0.75]
task = ['+task/subselection/multiobjective/dev=glob(*)', '+task/subselection/multiobjective/test=glob(*)']

# Generate all combinations
combinations = list(product(
    hpi_method,
    adjust_cs,
    adjust_previous_cfgs,
    dynamic_decay,
    adjust_cs_method,
    constant,
    cs_proba_hpi,
    set_to,
    thresh_list,
    thresh,
    task
))

columns = [
    'hpi_method', 'adjust_cs', 'adjust_previous_cfgs', 'dynamic_decay',
    'adjust_cs_method', 'constant', 'cs_proba_hpi', 'set_to',
    'thresh_list', 'thresh','task'
]
df = pd.DataFrame(combinations, columns=columns)

def remove(row, li):
    for col in li:
        row[col] = None
    return row

df[df['adjust_cs']=='false'] = df[df['adjust_cs']=='false'].apply(lambda row: remove(row, ['adjust_cs_method', 'constant', 'cs_proba_hpi']), axis=1)
df[(df['adjust_cs']=='true')&(df['cs_proba_hpi']=='true')] = df[(df['adjust_cs']=='true')&(df['cs_proba_hpi']=='true')].apply(lambda row: remove(row, ['constant']), axis=1)
df[(df['adjust_previous_cfgs']=='false')] = df[(df['adjust_previous_cfgs']=='false')].apply(lambda row: remove(row, ['set_to']), axis=1)
df[(df['dynamic_decay']=='none')] = df[(df['dynamic_decay']=='none')].apply(lambda row: remove(row, ['thresh_list']), axis=1)
df[(df['dynamic_decay']=='linear')] = df[(df['dynamic_decay']=='linear')].apply(lambda row: remove(row, ['thresh']), axis=1)
df = df[~((df['cs_proba_hpi']=='true')&(df['hpi_method']=='hypershap'))]

df = df.drop_duplicates()


commands = []

for _, row in df.iterrows():
    # Construct the command dynamically, skipping None values
    command = "python -m carps.experimenter.create_cluster_configs hydra.searchpath=[file:///home/daphne/Code/hpi_parego/package_hpi_parego/hpi_parego/configs] +optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego  'seed=range(0,5)'"
    command += f" '{row['task']}'"
    baserundir = f"results/"
    
    command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method={row['hpi_method']}"
    baserundir += f"{row['hpi_method']}/"
    
    command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs={row['adjust_cs']}"
    if row['adjust_cs'] == 'true':
        baserundir += f"adjust_cs_{row['adjust_cs_method']}"
        command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method={row['adjust_cs_method']}"
    
    if row['constant'] is not None:
        command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant={row['constant']}"
        if row['constant'] == 'true':
            baserundir += f"_constant"
    if row['cs_proba_hpi'] is not None:
        command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.cs_proba_hpi={row['cs_proba_hpi']}"
        if row['cs_proba_hpi'] == 'true':    
            baserundir += f"_cs_proba_hpi"
    
    command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs={row['adjust_previous_cfgs']}"
    if row['adjust_previous_cfgs'] == 'true': 
        baserundir += f"_adjust_prev_cfgs"
    
    if row['set_to'] is not None:
        command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.set_to={row['set_to']}"
        baserundir += f"_set_to_{row['set_to']}"
    
    command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay={row['dynamic_decay']}"
    if row['dynamic_decay'] == 'linear': 
        baserundir += f"_dynamic_decay"
    
    if row['thresh'] is not None:
        command += f" optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh={row['thresh']}"
        baserundir += f"_thresh_{row['thresh']}"
        
    if row['thresh_list'] is not None:
        command += f" 'optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh_list={row['thresh_list']}'"
        if row['thresh_list'] == '[0.9,0.8,0.7,0.6,0.5,0.4,0.3]': 
            baserundir += f"_down"
        elif row['thresh_list'] == '[0.3,0.4,0.5,0.6,0.7,0.8,0.9]':
            baserundir += f"_up"
        
        
    command += f" baserundir={baserundir}"
    command += f" optimizer_id={row['hpi_method']}_{baserundir.split('/')[-1]}"
    
    command += " -m"
    
    commands.append(command)

commands.reverse()
for command in commands:
    process = subprocess.run(command, shell=True)
   
    if process.returncode != 0:
        print(f"Command failed: {command}")
    # time.sleep(3)
print('command length', len(commands))
print('all commands executed')
exit()