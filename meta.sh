#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -J "meta"
#SBATCH -p normal
#SBATCH --mem=2GB


# Maximum number of jobs allowed in the queue
MAX_JOBS=500
# Your job submission command or script
JOB_SCRIPT="start_from_db.sh"
# Total jobs you want to submit
TOTAL_JOBS=30000


submitted=0

while [ "$submitted" -lt "$TOTAL_JOBS" ]; do
    # Count current user's jobs in the queue
    CURRENT_JOBS=$(squeue -u "$USER" -h -t R | wc -l)

    if [ "$CURRENT_JOBS" -lt "$MAX_JOBS" ]; then

        sbatch "$JOB_SCRIPT"
        submitted=$((submitted + 200))
        echo "Submitted job $submitted of $TOTAL_JOBS (Current queue: $CURRENT_JOBS)"
        sleep 300
    else
        echo "Queue full ($CURRENT_JOBS jobs). Waiting..."
        sleep 200
    fi
done

echo "All $TOTAL_JOBS jobs submitted."
