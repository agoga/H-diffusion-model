#!/bin/bash
#SBATCH --account=zimanyigrp
#SBATCH --partition=high2
#SBATCH --job-name=zip_logs_onefile
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8           # bump for more speed
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%j.out

set -euo pipefail

# ====== USER SETTINGS ========================================================
TARGET_DIR="/home/agoga/sandbox/diffusion/sim_data"   # no trailing slash
OUT_DIR="/home/agoga/sandbox/diffusion/archives"                # must NOT be inside TARGET_DIR
LEVEL="${LEVEL:-3}"                                             # zstd level; 3 is fast & good
CPUS="${SLURM_CPUS_PER_TASK:-1}"                                # threads for zstd
ARCHIVE_BASENAME="slurm_logs_old_old_$(date +%Y%m%d_%H%M%S)"
OUT_PATH="${OUT_DIR}/${ARCHIVE_BASENAME}.tar.zst"
# ============================================================================

mkdir -p "$OUT_DIR"

# safety: don't write archive inside the directory being archived
if [[ "$OUT_DIR" == "$TARGET_DIR"* ]]; then
  echo "ERROR: OUT_DIR must not be inside TARGET_DIR." >&2
  exit 1
fi

PARENT="$(dirname "$TARGET_DIR")"
BASE="$(basename "$TARGET_DIR")"
cd "$PARENT"

echo "Creating single archive:"
echo "  source: $TARGET_DIR"
echo "  out   : $OUT_PATH"
echo "  zstd  : level=$LEVEL threads=$CPUS"

# Prefer zstd; fallback to pigz if needed
if command -v zstd >/dev/null 2>&1; then
  # One single file, multi-threaded fast compression
  # -I lets us pass custom zstd args (threads & level)
  time tar -I "zstd -T${CPUS} -${LEVEL}" -cf "$OUT_PATH" "$BASE"
elif command -v pigz >/dev/null 2>&1; then
  # Fallback: single .tar.gz with pigz (multi-threaded gzip)
  OUT_PATH="${OUT_DIR}/${ARCHIVE_BASENAME}.tar.gz"
  time tar -I "pigz -${LEVEL}" -cf "$OUT_PATH" "$BASE"
else
  echo "ERROR: Neither zstd nor pigz found on PATH." >&2
  exit 1
fi

echo "Archive created:"
ls -lh "$OUT_PATH"
