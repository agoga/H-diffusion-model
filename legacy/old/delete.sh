#!/bin/bash
#SBATCH --account=zimanyigrp
#SBATCH --partition=high2
#SBATCH --job-name=wipe
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm_logs/%j.out

set -euo pipefail

# ---- HARD-CODE YOUR TARGET DIR HERE ----
TARGET_DIR="/home/agoga/sandbox/diffusion/output/"
EMPTY_DIR="/home/agoga/sandbox/diffusion/empty_dir"
# ----------------------------------------


mkdir -p "$EMPTY_DIR"

echo "Clearing $TARGET_DIR ..."
time rsync -a --delete "$EMPTY_DIR"/ "$TARGET_DIR"/

rmdir "$EMPTY_DIR" "$TARGET_DIR"