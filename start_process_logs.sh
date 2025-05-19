#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -J "hydra_launcher"
#SBATCH -p normal
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=16

module reset
module load lang/Miniforge3/24.1.2-0
source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate carps

export PYTHONPATH=/scratch/hpc-prf-intexml/daphne/hpi_parego:$PYTHONPATH


echo "Starting Hydra job submission..."
HYDRA_FULL_ERROR=1 python -m carps.experimenter.database.process_logs --results_dir ../experimenter/results
