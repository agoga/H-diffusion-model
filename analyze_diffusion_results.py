# analyze_diffusion_results.py
# ----------------------------
# Author: Adam (2025)
#
# Analysis and plotting utilities for diffusion simulation results.
# Includes functions for aligning simulations to experimental data, plotting, and parameter/result analysis.

# --- Imports ---
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# --- Import simulation data utilities ---
from simulation_data import SimulationData, SimulationResult, get_label

# --- Constants and Parameters ---
REGION_INDEX = 3  # Which region of simulation to compare
SIM_FOLDER = '/home/adam/code/diffusion_model/farm_out/big_run_3/'
EXP_CSV = '/home/adam/code/diffusion_model/digitized_plots/McDonald_digitized.csv'
ALIGNMENT_RESULTS_CSV = "alignment_results_ranked.csv"
MIN_SIM_TIME = 1000
GROUP_KEYS = ['fig', 'Dopant', 'Width', 'Temperature']




# --- Helper: Load all valid simulation runs as SimulationData objects ---
def load_simulation_data(sim_folder=SIM_FOLDER, min_time=MIN_SIM_TIME):
    """
    Returns a SimulationData object containing all valid simulation runs in the folder.

    Parameters:
        sim_folder (str): Path to the folder containing simulation results.
        min_time (float): Minimum simulation time for a run to be considered valid.

    Returns:
        SimulationData: Object containing all valid simulation runs.
    """
    return SimulationData(sim_folder, min_time)
            
            
def align_and_score(sim_t, sim_c, exp_t, exp_c):
    """
    Align a simulation curve to experimental data using scaling and shifting.

    Parameters:
        sim_t (np.ndarray): Simulation time array.
        sim_c (np.ndarray): Simulation concentration array.
        exp_t (np.ndarray): Experimental time array.
        exp_c (np.ndarray): Experimental concentration array.

    Returns:
        tuple: (alignment score, [a, b, c, d] optimal parameters)
    """
    # Normalize both time and concentration
    exp_t_norm = exp_t / np.max(exp_t)
    exp_c_norm = exp_c / np.max(exp_c)
    sim_t_norm = sim_t / np.max(sim_t)
    sim_c_norm = sim_c / np.max(sim_c)
    interp_sim_norm = interp1d(sim_t_norm, sim_c_norm, bounds_error=False, fill_value="extrapolate")

    def cost(params):
        a, b, d = params
        t_warped = b * exp_t_norm
        sim_aligned = a * interp_sim_norm(t_warped) + d
        weights = np.linspace(0.2, 1.0, len(exp_c_norm))
        fit_error = np.average((sim_aligned - exp_c_norm) ** 2, weights=weights)
        penalty = 1e-2 * (abs(a - 1) + abs(b - 1))
        return fit_error + penalty

    result = minimize(
        cost,
        [1.0, 1.0, 0.0],
        bounds=[(0.1, 10.0), (0.1, 10.0), (-1.0, 1.0)],
        method='L-BFGS-B'
    )
    # Rescale parameters
    a_out = result.x[0] * (np.max(exp_c) / np.max(sim_c))
    d_out = result.x[2] * np.max(exp_c)
    b_out = result.x[1] * (np.max(exp_t) / np.max(sim_t))
    c_out = 0.0
    return result.fun, [a_out, b_out, c_out, d_out]

def plot_top_matches(results_csv=ALIGNMENT_RESULTS_CSV,sim_folder=SIM_FOLDER,exp_csv=EXP_CSV):
    """
    Plot the best-matching simulation results for each experimental group.

    Parameters:
        results_csv (str): Path to CSV file with alignment results.
        sim_folder (str): Folder containing simulation .npz files.
        exp_csv (str): Path to experimental data CSV.
    """
    results = pd.read_csv(results_csv)
    exp_df = pd.read_csv(exp_csv)
    best_results = results.sort_values('score').groupby(GROUP_KEYS).first().reset_index()

    for _, row in best_results.iterrows():
        fig_id = row['fig']
        dopant = row['Dopant']
        width = row['Width']
        temp = row['Temperature']
        sim_file = row['sim_file']
        a, b, c, d = row['a'], row['b'], row['c'], row['d']

        exp_subset = exp_df[
            (exp_df['fig'] == fig_id) &
            (exp_df['Dopant'] == dopant) &
            (exp_df['Width'] == width) &
            (exp_df['Temperature'] == temp)
        ].sort_values(by='Time')
        exp_t = exp_subset['Time'].values
        exp_c = exp_subset['[H]'].values

        sim_data = np.load(os.path.join(sim_folder, sim_file))
        sim_t = sim_data['times']
        sim_c = sim_data['ut'][:, REGION_INDEX]

        warped_t = b * sim_t + c
        aligned_sim_c = a * sim_c + d

        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        formatter.set_scientific(True)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.offsetText.set_visible(True)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
        ax.yaxis.offsetText.set_position((0, 1.02))
        plt.plot(exp_t, exp_c, 'ko-', label='Experimental')
        plt.plot(warped_t, aligned_sim_c, 'r--o', markersize=3, label='Aligned Simulation')
        plt.xlabel("Exp. Time (s)")
        plt.ylabel("[H] Concentration")
        run_label = os.path.splitext(sim_file)[0]
        plt.title(f"{fig_id} | {dopant} | W={width}μm | T={temp}°C\nSim: {run_label}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
    print("Plots shown for best matches.")

# --- Main Analysis Function ---
def run_analysis():
    """
    Main analysis routine: aligns all valid simulations to all experimental groups and saves results.
    """
    print("Loading experimental data...")
    exp_df = pd.read_csv(EXP_CSV)
    exp_groups = exp_df.groupby(GROUP_KEYS)


    print("Scanning valid simulation files...")
    sim_data_obj = load_simulation_data()
    print(f"Found {len(sim_data_obj)} valid simulation files.")

    all_results = []
    print("Aligning simulations to experimental groups...")
    for group_key, exp_group in exp_groups:
        exp_t = exp_group.sort_values(by='Time')['Time'].values
        exp_c = exp_group.sort_values(by='Time')['[H]'].values
        exp_width = group_key[2]

        for sim_result in tqdm(sim_data_obj, desc=f"Matching to {group_key}"):
            try:
                sim_filename = os.path.basename(sim_result.sim_path)
                if sim_result.sample_length_um is None:
                    print(f"Skipping {sim_filename}: 'sample_length_um' missing from parameter file.")
                    continue
                sim_width = float(sim_result.sample_length_um)
                if sim_width != exp_width:
                    continue  # only align if widths match
                sim_t = sim_result.sim_data['times']
                sim_c = sim_result.sim_data['ut'][:, REGION_INDEX]
                score, params = align_and_score(sim_t, sim_c, exp_t, exp_c)
                all_results.append({
                    'fig': group_key[0],
                    'Dopant': group_key[1],
                    'Width': group_key[2],
                    'Temperature': group_key[3],
                    'sim_file': sim_filename,
                    'score': score,
                    'a': params[0],
                    'b': params[1],
                    'c': params[2],
                    'd': params[3]
                })
            except Exception as e:
                print(f"Failed {sim_result.sim_path}: {e}")
                continue

    results_df = pd.DataFrame(all_results)
    print("Sorting and saving results...")
    sorted_results = (
        results_df
        .sort_values(['fig', 'score'])
        .groupby('fig')
        .head(50)
    )
    sorted_results.to_csv(ALIGNMENT_RESULTS_CSV, index=False)
    print(f"Saved to {ALIGNMENT_RESULTS_CSV}")


# --- Plot histogram(s) of simulation parameters ---
def plot_parameter_histograms(
    param=None,
    sim_folder=SIM_FOLDER,
    min_time=0,
    bins=20,
    plot_all=False
):
    """
    Plot a histogram of a specific parameter across all runs, or all parameters if plot_all=True.

    Parameters:
        param (str): Parameter to plot. Ignored if plot_all=True.
        sim_folder (str): Folder containing simulation .npz and parameter .csv files.
        min_time (float): Minimum simulation time for a run to be considered valid.
        bins (int): Number of bins for the histogram.
        plot_all (bool): If True, plot histograms for all parameters with >1 unique value.
    """
    sim_data_obj = load_simulation_data(sim_folder, min_time)
    if len(sim_data_obj) == 0:
        print("No valid simulation runs found.")
        return
    import pandas as pd
    df = pd.DataFrame([
        {**r.param_dict, 'sim_id': r.sim_id, 'max_t': r.get_max_time()} for r in sim_data_obj
    ])
    param_cols = [
        c for c in df.columns
        if c not in ('sim_id', 'max_t') and not c.startswith('_') and df[c].nunique() > 1
    ]
    if plot_all:
        for p in param_cols:
            plt.figure(figsize=(5,3))
            plt.hist(df[p].astype(float), bins=bins, alpha=0.8, color='C0', edgecolor='k')
            plt.xlabel(get_label(p))
            plt.ylabel(ylab)
            plt.ylabel('Count')
            plt.title(f"Histogram of {get_label(x)}  (n={len(df)})")
            plt.tight_layout()
            plt.show()
        return
    if param is None:
        print("Please specify a parameter name or set plot_all=True.")
        return
    if param not in df.columns:
        print(f"Parameter '{param}' not found in data.")
        return
    if df[param].nunique() <= 1:
        print(f"Parameter '{param}' has only one unique value; nothing to plot.")
        return
    plt.figure(figsize=(5,3))
    plt.hist(df[param].astype(float), bins=bins, alpha=0.8, color='C0', edgecolor='k')
    plt.xlabel(get_label(param))
    plt.ylabel('Count')
    plt.title(f"Histogram of {get_label(param)}  (n={len(df)})")
    plt.tight_layout()
    plt.show()

# --- Simulation Parameter/Result Analysis ---
def analyze_simulation_parameters(
    sim_folder=SIM_FOLDER,
    region_index=REGION_INDEX,
    min_time=0,
    x_param='diffusion_rate_input',
    y_param='total_runtime_s',
    plot_vs_max_time=False,
    color_by=None,
    filter_func=None
):
    """
    Analyze and plot simulation parameters vs. results.

    Parameters:
        sim_folder (str): Folder containing simulation .npz and parameter .csv files.
        region_index (int): Index of region to analyze.
        min_time (float): Minimum simulation time for a run to be considered valid.
        x_param (str): Parameter name for x-axis (from CSV), or 'all' to plot all parameters vs y_param.
        y_param (str): Parameter name for y-axis (from CSV), or 'all' to plot x_param vs all other parameters.
        plot_vs_max_time (bool): If True, y-axis is max simulation time from .npz file.
        color_by (str): Optional parameter name to color scatter plot by.
        filter_func (callable): Optional function(param_dict) -> bool to filter runs.
    """
    # Collect all parameter sets
    sim_data_obj = load_simulation_data(sim_folder, min_time)
    if len(sim_data_obj) == 0:
        print("No valid simulation runs found.")
        return
    if filter_func:
        sim_data_obj = [r for r in sim_data_obj if filter_func(r.param_dict)]
    df = pd.DataFrame([
        {**r.param_dict, 'sim_id': r.sim_id, 'max_t': r.get_max_time()} for r in sim_data_obj
    ])
    n_runs = len(df)

    # Determine which parameters to plot (only those with >1 unique value)
    param_cols = [
        c for c in df.columns
        if c not in ('sim_id', 'max_t') and not c.startswith('_') and df[c].nunique() > 1
    ]

    def plot_pair(xkey, ykey, xlab=None, ylab=None):
        xs = df[xkey].astype(float)
        ys = df[ykey].astype(float)
        xlab = xlab or get_label(xkey)
        ylab = ylab or get_label(ykey)
        plt.figure(figsize=(6, 4))
        if color_by and color_by in df.columns:
            cvals = df[color_by].astype(float)
            sc = plt.scatter(xs, ys, c=cvals, cmap='viridis', alpha=0.7)
            cbar = plt.colorbar(sc)
            cbar.set_label(get_label(color_by))
        else:
            plt.scatter(xs, ys, alpha=0.7)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(f"{ylab} vs {xlab}  (n={n_runs})")
        plt.tight_layout()
        plt.show()

    # Handle combinations of 'all'
    if x_param == 'all' and y_param == 'all':
        print("Cannot plot all vs all. Please specify at least one axis.")
        return

    if x_param == 'all':
        for x in param_cols:
            if x == y_param:
                continue
            ykey = 'max_t' if plot_vs_max_time or y_param == 'max_t' else y_param
            plot_pair(x, ykey)
        return

    if y_param == 'all':
        for y in param_cols:
            if y == x_param:
                continue
            xkey = 'max_t' if plot_vs_max_time or x_param == 'max_t' else x_param
            plot_pair(xkey, y)
        return

    # Standard single plot
    ykey = 'max_t' if plot_vs_max_time or y_param == 'max_t' else y_param
    plot_pair(x_param, ykey)

# --- Main Entrypoint ---
if __name__ == "__main__":
    run_analysis()
    plot_top_matches(
        results_csv=ALIGNMENT_RESULTS_CSV,
        sim_folder=SIM_FOLDER,
        exp_csv=EXP_CSV
    )
    # Example usage for parameter/result analysis:
    # analyze_simulation_parameters(x_param='diffusion_rate_input', y_param='total_runtime_s')
    # analyze_simulation_parameters(x_param='hopping_barrier', y_param='max_t', plot_vs_max_time=True)
    # Example usage for parameter analysis:
    # analyze_simulation_parameters(plot_param_vs=['A_detrap_Ea', 'B_trap_Ea'], plot_vs_runtime=True, plot_vs_maxtime=True)

