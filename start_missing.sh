#!/bin/bash
#SBATCH --job-name=hpi_parego
#SBATCH --output=hpi_parego%j.out
#SBATCH --error=hpi_parego%j.err

#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

module load Miniforge3
echo 'activating conda'

conda activate /bigwork/$USER/.conda/carps

export PYTHONPATH=/bigwork/$USER/hpi_parego/package_hpi_parego
echo "PYTHONPATH: $PYTHONPATH"





export HYDRA_FULL_ERROR=1
bash $1

wait
