#!/bin/bash
#SBATCH --job-name=meta
#SBATCH --output=meta_%j.out
#SBATCH --error=meta_%j.err
#SBATCH --partition=ai,taurus,amo
#SBATCH --exclude=ai-n[001-004],ai-n009
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --mem=2GB

# Maximum number of jobs allowed in the queue
MAX_JOBS=290
# Your job submission command or script
JOB_SCRIPT="start_from_db.sh"
# Total jobs you want to submit
TOTAL_JOBS=100000


submitted=0

while [ "$submitted" -lt "$TOTAL_JOBS" ]; do
    # Count current user's jobs in the queue
    CURRENT_JOBS=$(squeue -u "$USER" | tail -n +2 | wc -l)

    if [ "$CURRENT_JOBS" -lt "$MAX_JOBS" ]; then

        sbatch "$JOB_SCRIPT"
        submitted=$((submitted + 200))
        echo "Submitted job $submitted of $TOTAL_JOBS (Current queue: $CURRENT_JOBS)"
        sleep 200
    else
        echo "Queue full ($CURRENT_JOBS jobs). Waiting..."
        sleep 200
    fi
done

echo "All $TOTAL_JOBS jobs submitted."
