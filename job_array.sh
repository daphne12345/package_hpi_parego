#!/bin/bash
#SBATCH --array=0-99  # Adjust this range based on the number of different configurations (30 in this case)
#SBATCH --job-name=multi_jobs
#SBATCH --output=multi_jobs_%A_%a.out
#SBATCH --error=multi_jobs_%A_%a.err

# Define all options for the different parameters
hpi_method=("hypershap" "fanova" "random")
adjust_cs=("true" "false")
adjust_previous_cfgs=("true" "false")
dynamic_decay=("linear" "none")

# Define conditional configurations
adjust_cs_method=("default" "random")
constant=("true" "false")
set_to_default=("true" "false")
thresh_linear=(0.9 0.8 0.7 0.6 0.5 0.4 0.3)
thresh_no_decay=(0.5 0.75)

# Loop over all combinations of hpi_method, adjust_cs, adjust_previous_cfgs, and dynamic_decay
index=0
for hpi in "${hpi_method[@]}"; do
    for acs in "${adjust_cs[@]}"; do
        for adjust_prev in "${adjust_previous_cfgs[@]}"; do
            for decay in "${dynamic_decay[@]}"; do
                # Handle conditional configurations
                if [ "$adjust_cs" == "true" ]; then
                    for acs_method in "${adjust_cs_method[@]}"; do
                        for cons in "${constant[@]}"; do
                            baserundir="result_hpi_hypershap_adjust_cs_${acs}_method_${acs_method}_constant_${cons}"
                            sbatch start_hydra.sh optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=$hpi optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=$acs optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs_method=$acs_method optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.constant=$cons
                        done
                    done
                fi

                if [ "$acs" == "true" ] && [ "$adjust_prev" == "true" ]; then
                    # Specific configuration for adjust_cs and adjust_previous_cfgs being true
                    for set_default in "${set_to_default[@]}"; do
                        baserundir="result_adjust_cs_${acs}_adjust_previous_${adjust_prev}_set_default_${set_default}"
                        sbatch start_hydra.sh optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=$adjust_prev optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.set_to_default=$set_default
                    done
                fi

                if [ "$decay" == "linear" ]; then
                    # Linear decay specific configurations
                    for thresh in "${thresh_linear[@]}"; do
                        baserundir="result_decay_linear_thresh_${thresh}"
                        sbatch start_hydra.sh optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=$decay optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh=$thresh
                    done
                elif [ "$decay" == "none" ]; then
                    # No decay specific configurations
                    for thresh in "${thresh_no_decay[@]}"; do
                        baserundir="result_decay_none_thresh_${thresh}"
                        sbatch start_hydra.sh optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=$decay optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.thresh=$thresh
                    done
                fi

                # Default configurations (for other combinations that don't match the conditionals)
                if [ "$hpi" != "hypershap" ]; then
                    baserundir="result_${hpi}_adjust_cs_${acs}_adjust_prev_${adjust_prev}_decay_${decay}"
                    sbatch start_hydra.sh optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.hpi_method=$hpi optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_cs=$acs optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.adjust_previous_cfgs=$adjust_prev optimizer.smac_cfg.smac_kwargs.acquisition_maximizer.dynamic_decay=$decay
                fi

                # Increment index
                ((index++))
                if [ $index -ge 30 ]; then
                    exit 0  # Stop after 30 jobs to match the array size
                fi
            done
        done
    done
done
