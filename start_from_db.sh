#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -J "hydra_launcher"
#SBATCH -p normal
#SBATCH --mem=2GB

module reset
module load lang/Miniforge3/24.1.2-0
source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate carps

export PYTHONPATH=/scratch/hpc-prf-intexml/daphne/hpi_parego:$PYTHONPATH


echo "Starting Hydra job submission..."
HYDRA_FULL_ERROR=1 python -m carps.run_from_db 'job_nr_dummy=range(1,2)' -m
