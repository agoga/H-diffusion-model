#!/bin/bash -l
#SBATCH --account=zimanyigrp
#SBATCH --partition=med2
#SBATCH --job-name=hsweep
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH -t 06:00:00
#SBATCH --array=0-1023

set -euo pipefail

# --- derive the current stdout path from the pattern %A_%a ---
LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR" sim_data
LOG="${LOG_DIR}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

CONFIG="single_sweep_config.json"
# CONFIG="single_sweep_config_trimmed.json"

# --- where to relocate logs after completion ---
OK_DIR="${LOG_DIR}/OK"
FAIL_DIR="${LOG_DIR}/FAILED"
mkdir -p "$OK_DIR" "$FAIL_DIR"

_finish() {
  ec=$?
  sleep 0.2 || true
  if [[ $ec -eq 0 ]]; then
    mv -f "$LOG" "${OK_DIR}/$(basename "$LOG")" 2>/dev/null || true
    echo "[OK] $(date -Is) job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
  else
    echo "[FAILED exit=$ec] $(date -Is) job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID}" >> "$LOG" || true
    mv -f "$LOG" "${FAIL_DIR}/$(basename "$LOG")" 2>/dev/null || true
  fi
  exit $ec
}
trap _finish EXIT

# Optional: split stderr
# exec 2> "slurm_logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# --- environment ---
module load conda
conda activate diff
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

echo "Running array task $SLURM_ARRAY_TASK_ID of ${SLURM_ARRAY_TASK_COUNT:-?}"





# Data-only sweep: params from JSON, schedules from sweeps.py helpers
# (single_sweep_config.json produced by gen_tasks_nofast.py)
srun python -u run_sweep_json.py \
  --config $CONFIG \
  --results-dir sim_data
  # If you want PETSc tweaks, add e.g.:
  # --shard-id ${SLURM_ARRAY_TASK_ID} --num-shards ${SLURM_ARRAY_TASK_COUNT}
  # (not needed on Slurm; runner reads SLURM env automatically)
