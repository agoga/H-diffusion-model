# create_sweep.py
# ---------------
# Author: Adam (2025)
#
# Script to generate a parameter sweep for diffusion simulations.
# Avoids repeating jobs that have already been run (within a tolerance) by checking existing results.
# Outputs a CSV file with jobs to run, partitioned by estimated runtime.

import itertools
import pandas as pd
import numpy as np

# For simulation result filtering
import sys
from simulation_data import SimulationData, map_sweep_to_sim_keys, compare_param_sets
import argparse



def get_existing_param_dicts(sim_folder, min_time=1400):
    """
    Load all SimulationResults from sim_folder with max_t > min_time.

    Parameters:
        sim_folder (str): Path to folder with simulation results.
        min_time (float): Minimum max_t for a run to be considered completed.

    Returns:
        list[dict]: List of parameter dictionaries for comparison.
    """
    sim_data = SimulationData(sim_folder, min_time)
    param_dicts = []
    for r in sim_data:
        # Only keep numeric values for comparison
        pdict = {k: float(v) for k, v in r.param_dict.items() if isinstance(v, (int, float, np.integer, np.floating, np.float64, np.float32, np.int64, np.int32, float)) or str(v).replace('.','',1).isdigit()}
        param_dicts.append(pdict)
    return param_dicts

def make_center_percent(center, percent, count):
    """
    Generate a symmetric range of values around a center, spanning +/- percent.

    Parameters:
        center (float): Center value.
        percent (float): Fractional percent (e.g., 0.2 for ±20%).
        count (int): Number of points.

    Returns:
        np.ndarray: Array of values.
    """
    delta = center * percent
    return np.linspace(center - delta, center + delta, count)

def make_range(min_val, max_val, count):
    """
    Generate a linear range of values from min_val to max_val (inclusive).

    Parameters:
        min_val (float): Minimum value.
        max_val (float): Maximum value.
        count (int): Number of points.

    Returns:
        np.ndarray: Array of values.
    """
    return np.linspace(min_val, max_val, count)

# === PARAMETER SWEEP DEFINITIONS ===
num_params = 4

param_ranges = {
    "A_detrap":             make_range         (1.5, 2.4, 8),
    # "A_trap":               make_range         (0.5, 1.5, 8),

    "B_detrap":             make_center_percent(0.7, 0.2, num_params),
    "B_trap":               make_center_percent(0.5, 0.2, num_params),

    "C_detrap":             make_center_percent(1.3, 0.2, num_params),
    "C_trap":               make_center_percent(0.5, 0.2, num_params),
}


# === GENERATE PARAMETER COMBINATIONS ===
param_names = list(param_ranges.keys())
combinations = list(itertools.product(*(param_ranges[k] for k in param_names)))
df = pd.DataFrame(combinations, columns=param_names)
df.insert(0, "job_id", range(1, len(df) + 1))


# === FILTER OUT EXISTING RUNS ===
parser = argparse.ArgumentParser(description="Generate parameter sweep, avoiding repeats.")
parser.add_argument('--existing_folder', type=str, default=None, help='Folder with existing simulation results to avoid repeats.')
parser.add_argument('--min_time', type=float, default=1000, help='Minimum max_t for a run to be considered completed.')
parser.add_argument('--tol', type=float, default=0.01, help='Fractional tolerance for parameter matching (default 1%).')
parser.add_argument('--output', type=str, default='parameter_sweep_inputs.csv', help='Output CSV file name.')
args, _ = parser.parse_known_args()

existing_param_dicts = []
if args.existing_folder:
    print(f"Loading existing runs from {args.existing_folder} (min_time={args.min_time})...")
    existing_param_dicts = get_existing_param_dicts(args.existing_folder, min_time=args.min_time)
    print(f"Loaded {len(existing_param_dicts)} completed runs.")

keep_rows = []
skipped = 0
for idx, row in df.iterrows():
    sweep_dict = {k: float(row[k]) for k in param_names}
    found = False
    if existing_param_dicts:
        for ep in existing_param_dicts:
            if compare_param_sets(sweep_dict, ep, tol=args.tol):
                # print(f'Skipping (already run): {sweep_dict}')
                skipped += 1
                found = True
                break
    if found:
        continue
    keep_rows.append(row)
df = pd.DataFrame(keep_rows, columns=df.columns)
print(f"Skipped {skipped} jobs due to existing completed runs.")

# === PARTITION ASSIGNMENT ===
# === ESTIMATE PRIORITY ===
def runtime_score(row):
    """
    Estimate the runtime score for a parameter set (lower is faster).

    Parameters:
        row (pd.Series): Row of parameter values.

    Returns:
        float: Estimated runtime score.
    """
    return 1.0 / row["B_trap"] + 1.0 / row["B_detrap"]

df["priority"] = df.apply(runtime_score, axis=1)

# Sort longest runs first
df = df.sort_values("priority", ascending=False).reset_index(drop=True)

# Assign top N to high2, the rest to med2
num_high2 = 256  # or 256 * actual cores per job (e.g., if you're stuck at 2 CPUs/job)
df["partition"] = ["high2" if i < num_high2 else "med2" for i in range(len(df))]

# For med2 jobs, re-shuffle to improve throughput (don't want all long ones in the back)
med2_df = df[df["partition"] == "med2"].copy()
np.random.seed(42)
med2_df["priority"] += np.random.uniform(0, 0.5, size=len(med2_df))
med2_df = med2_df.sort_values("priority", ascending=False).reset_index(drop=True)

# Combine back and drop priority from both
high2_df = df[df["partition"] == "high2"].drop(columns="priority")
df = pd.concat([high2_df, med2_df.drop(columns="priority")], ignore_index=True)

df["job_id"] = df["job_id"].astype(int)

# === SAVE CSV ===
df.to_csv(args.output, index=False)
print(f"Generated {len(df)} jobs:")
print(f" - Assigned {min(num_high2, len(df))} to high2")
print(f" - Assigned {max(0, len(df) - num_high2)} to med2")
