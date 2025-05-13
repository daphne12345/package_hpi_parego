#!/bin/bash
#SBATCH --job-name=hydra_launcher
#SBATCH --output=hydra_launcher_%j.Kom
#SBATCH --error=hydra_launcher_%j.err
#SBATCH --partition=ai,taurus,amo
#SBATCH --exclude=ai-n[001-004],ai-n009
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=2GB

module load Miniforge3
conda activate /bigwork/$USER/.conda/carps

echo "Starting Hydra job submission..."
HYDRA_FULL_ERROR=1 python -m carps.run_from_db 'job_nr_dummy=range(1,200)' -m
