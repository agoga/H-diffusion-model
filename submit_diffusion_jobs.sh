#!/bin/bash -l

# Author: Adam (2025)
# Usage: ./submit_diffusion_jobs.sh <subfolder_name>
# Submits jobs using parameters in parameter_sweep_inputs.csv

# === Require one positional argument ===
if [ -z "$1" ]; then
    echo "Usage: $0 <subfolder_name>"
    echo "You must provide a subfolder name to store the output (e.g., results_623K)"
    exit 1
fi

# Optional second argument: number of jobs to run
max_jobs=-1  # default: no limit

if [ -n "$2" ]; then
    max_jobs=$2
    echo "Limiting to first $max_jobs jobs"
fi

subfolder="$1"
base_outdir=$(pwd)/output/"$subfolder"
mkdir -p "$base_outdir"

log_dir=$(pwd)"/slurm_output/"$subfolder
mkdir -p "$log_dir"

script_path=$(pwd)/diffusion_calculator.py
param_file="parameter_sweep_inputs.csv"

# === Read each row in CSV and submit job ===
job_count=0

tail -n +2 "$param_file" | while IFS=',' read -r job_id A_detrap B_detrap B_trap C_detrap C_trap partition
do
    if [ $max_jobs -ge 0 ] && [ $job_count -ge $max_jobs ]; then
        break
    fi

    jobname="diff_${job_id}"
    outdir="$base_outdir"

    sbatch <<-EOT
#!/bin/bash -l
#SBATCH -D ./
#SBATCH --job-name=$jobname
#SBATCH --partition=$partition
#SBATCH --output=$log_dir/j-%j_$job_id.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,END
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000  # exact fit
#SBATCH -t 2-00:00:00

module purge
module load slurm
module load conda


conda activate diff

echo "Running job $jobname in partition $partition"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python3 "$script_path" \\
    -id $job_id \\
    -L 150 \\
    -fol $outdir \\
    -AH $A_detrap \\
    -BH $B_detrap \\
    -HB $B_trap \\
    -CH $C_detrap \\
    -HC $C_trap
EOT

    job_count=$((job_count + 1))

done