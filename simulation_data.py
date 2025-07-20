
# simulation_data.py
# ------------------
# Author: Adam (2025)
#
# Utilities and classes for loading, mapping, and comparing simulation parameter/result data.
# Provides SimulationResult and SimulationData classes for object-oriented access.

import os
import numpy as np
import pandas as pd

# --- Parameter name mapping for sweep <-> simulation CSV ---
SWEEP_TO_SIM_KEYS = {
    'A_detrap': 'A_detrap_Ea_eV',
    'B_detrap': 'B_detrap_Ea_eV',
    'B_trap': 'B_trap_Ea_eV',
    'C_detrap': 'C_detrap_Ea_eV',
    'C_trap': 'C_trap_Ea_eV',
    # Add more as needed
}

def map_sweep_to_sim_keys(sweep_dict):
    """
    Convert a dict with sweep parameter names to simulation parameter names.

    Parameters:
        sweep_dict (dict): Dictionary with sweep parameter names as keys.

    Returns:
        dict: Dictionary with simulation parameter names as keys.
    """
    return {SWEEP_TO_SIM_KEYS.get(k, k): v for k, v in sweep_dict.items()}

def compare_param_sets(sweep_dict, sim_param_dict, tol=0.01):
    """
    Compare a sweep param dict (with sweep names) to a sim param dict (with sim names).

    Parameters:
        sweep_dict (dict): Dictionary with sweep parameter names as keys.
        sim_param_dict (dict): Dictionary with simulation parameter names as keys.
        tol (float): Fractional tolerance for parameter matching.

    Returns:
        bool: True if all mapped values are within tol of the sim values, False otherwise.
    """
    mapped = map_sweep_to_sim_keys(sweep_dict)
    for k, v in mapped.items():
        if k not in sim_param_dict:
            return False
        try:
            v1 = float(v)
            v2 = float(sim_param_dict[k])
            if v1 == 0 and v2 == 0:
                continue
            if v1 == 0 or v2 == 0:
                if abs(v1 - v2) > tol:
                    return False
            elif abs(v1 - v2) / max(abs(v1), abs(v2)) > tol:
                return False
        except Exception:
            if v != sim_param_dict[k]:
                return False
    return True

PARAM_LABELS = {
    'max_t': 'Max simulation time reached (s)',
    'total_runtime_s': 'Simulation wall time (s)',
    'sample_length_um': 'Sample length (μm)',
    'temperature_K': 'Temperature (K)',
    'diffusion_rate_input': 'Input diffusion rate',
    'hopping_barrier_eV': 'Hopping barrier (eV)',
    'timestep_s': 'Simulation timestep (s)',
    'max_time_s': 'Max simulation time (s)',
    'A_trap_Ea_eV': 'A trap activation energy (eV)',
    'A_detrap_Ea_eV': 'A detrap activation energy (eV)',
    'A_trap_freq': 'A trap frequency (Hz)',
    'A_detrap_freq': 'A detrap frequency (Hz)',
    'B_trap_Ea_eV': 'B trap activation energy (eV)',
    'B_detrap_Ea_eV': 'B detrap activation energy (eV)',
    'B_trap_freq': 'B trap frequency (Hz)',
    'B_detrap_freq': 'B detrap frequency (Hz)',
    'C_trap_Ea_eV': 'C trap activation energy (eV)',
    'C_detrap_Ea_eV': 'C detrap activation energy (eV)',
    'C_trap_freq': 'C trap frequency (Hz)',
    'C_detrap_freq': 'C detrap frequency (Hz)',
    # Add more as needed
}

def get_label(param):
    """
    Return a human-readable label for a parameter, if available.

    Parameters:
        param (str): Parameter name.

    Returns:
        str: Human-readable label or the parameter name itself.
    """
    return PARAM_LABELS.get(param, param)

def is_valid_simulation(npz_path, min_time=1000):
    """
    Check if a simulation .npz file is valid (i.e., max time > min_time).

    Parameters:
        npz_path (str): Path to the .npz file.
        min_time (float): Minimum simulation time for a run to be considered valid.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        data = np.load(npz_path)
        times = data['times']
        return times[-1] > min_time
    except Exception:
        return False

class SimulationResult:
    def __init__(self, sim_path, param_path, sim_id, param_dict, sim_data):
        """
        Initialize a SimulationResult object.

        Parameters:
            sim_path (str): Path to the simulation .npz file.
            param_path (str): Path to the parameter .csv file.
            sim_id (str): Simulation ID.
            param_dict (dict): Dictionary of simulation parameters.
            sim_data (np.lib.npyio.NpzFile): Loaded simulation data.
        """
        self.sim_path = sim_path
        self.param_path = param_path
        self.sim_id = sim_id
        self.param_dict = param_dict
        self.sim_data = sim_data

    def get_max_time(self):
        """
        Get the maximum simulation time for this result.

        Returns:
            float: Maximum simulation time.
        """
        return float(self.sim_data['times'][-1])

    def get_param(self, key):
        """
        Get a parameter value by key.

        Parameters:
            key (str): Parameter name.

        Returns:
            Any: Parameter value or None if not found.
        """
        return self.param_dict.get(key, None)

    def set_param(self, key, value):
        """
        Set a parameter value by key.

        Parameters:
            key (str): Parameter name.
            value (Any): Value to set.
        """
        self.param_dict[key] = value

    def get_label(self, key):
        """
        Get a human-readable label for a parameter key.

        Parameters:
            key (str): Parameter name.

        Returns:
            str: Human-readable label.
        """
        return get_label(key)

    # Property for A_detrap_freq
    @property
    def a_detrap_freq(self):
        return self.param_dict.get('A_detrap_freq', None)

    @a_detrap_freq.setter
    def a_detrap_freq(self, value):
        self.param_dict['A_detrap_freq'] = value

    # Property for A_trap_Ea
    @property
    def a_trap_ea(self):
        return self.param_dict.get('A_trap_Ea_eV', None)

    @a_trap_ea.setter
    def a_trap_ea(self, value):
        self.param_dict['A_trap_Ea_eV'] = value

    # Property for A_detrap_Ea
    @property
    def a_detrap_ea(self):
        return self.param_dict.get('A_detrap_Ea_eV', None)

    @a_detrap_ea.setter
    def a_detrap_ea(self, value):
        self.param_dict['A_detrap_Ea_eV'] = value

    # Property for A_trap_freq
    @property
    def a_trap_freq(self):
        return self.param_dict.get('A_trap_freq', None)

    @a_trap_freq.setter
    def a_trap_freq(self, value):
        self.param_dict['A_trap_freq'] = value

    # Property for B_trap_Ea
    @property
    def b_trap_ea(self):
        return self.param_dict.get('B_trap_Ea_eV', None)

    @b_trap_ea.setter
    def b_trap_ea(self, value):
        self.param_dict['B_trap_Ea_eV'] = value

    # Property for B_detrap_Ea
    @property
    def b_detrap_ea(self):
        return self.param_dict.get('B_detrap_Ea_eV', None)

    @b_detrap_ea.setter
    def b_detrap_ea(self, value):
        self.param_dict['B_detrap_Ea_eV'] = value

    # Property for B_trap_freq
    @property
    def b_trap_freq(self):
        return self.param_dict.get('B_trap_freq', None)

    @b_trap_freq.setter
    def b_trap_freq(self, value):
        self.param_dict['B_trap_freq'] = value

    # Property for B_detrap_freq
    @property
    def b_detrap_freq(self):
        return self.param_dict.get('B_detrap_freq', None)

    @b_detrap_freq.setter
    def b_detrap_freq(self, value):
        self.param_dict['B_detrap_freq'] = value

    # Property for C_trap_Ea
    @property
    def c_trap_ea(self):
        return self.param_dict.get('C_trap_Ea_eV', None)

    @c_trap_ea.setter
    def c_trap_ea(self, value):
        self.param_dict['C_trap_Ea_eV'] = value

    # Property for C_detrap_Ea
    @property
    def c_detrap_ea(self):
        return self.param_dict.get('C_detrap_Ea_eV', None)

    @c_detrap_ea.setter
    def c_detrap_ea(self, value):
        self.param_dict['C_detrap_Ea_eV'] = value

    # Property for C_trap_freq
    @property
    def c_trap_freq(self):
        return self.param_dict.get('C_trap_freq', None)

    @c_trap_freq.setter
    def c_trap_freq(self, value):
        self.param_dict['C_trap_freq'] = value

    # Property for C_detrap_freq
    @property
    def c_detrap_freq(self):
        return self.param_dict.get('C_detrap_freq', None)

    @c_detrap_freq.setter
    def c_detrap_freq(self, value):
        self.param_dict['C_detrap_freq'] = value

    # Property for sample_length_um
    @property
    def sample_length_um(self):
        return self.param_dict.get('sample_length_um', None)

    @sample_length_um.setter
    def sample_length_um(self, value):
        self.param_dict['sample_length_um'] = value

    # Property for temperature_K
    @property
    def temperature_K(self):
        return self.param_dict.get('temperature_K', None)

    @temperature_K.setter
    def temperature_K(self, value):
        self.param_dict['temperature_K'] = value

    # Property for diffusion_rate_input
    @property
    def diffusion_rate_input(self):
        return self.param_dict.get('diffusion_rate_input', None)

    @diffusion_rate_input.setter
    def diffusion_rate_input(self, value):
        self.param_dict['diffusion_rate_input'] = value

    # Property for hopping_barrier
    @property
    def hopping_barrier(self):
        return self.param_dict.get('hopping_barrier_eV', None)

    @hopping_barrier.setter
    def hopping_barrier(self, value):
        self.param_dict['hopping_barrier_eV'] = value

    # Property for timestep
    @property
    def timestep(self):
        return self.param_dict.get('timestep_s', None)

    @timestep.setter
    def timestep(self, value):
        self.param_dict['timestep_s'] = value

    # Property for max_time
    @property
    def max_time(self):
        return self.param_dict.get('max_time_s', None)

    @max_time.setter
    def max_time(self, value):
        self.param_dict['max_time_s'] = value

    # Property for total_runtime
    @property
    def total_runtime(self):
        return self.param_dict.get('total_runtime_s', None)

    @total_runtime.setter
    def total_runtime(self, value):
        self.param_dict['total_runtime_s'] = value

    # Property for max_t
    @property
    def max_t(self):
        return self.param_dict.get('max_t', None)

    @max_t.setter
    def max_t(self, value):
        self.param_dict['max_t'] = value

class SimulationData:
    def __init__(self, sim_folder, min_time=1000):
        """
        Initialize a SimulationData object and load all valid simulation results.

        Parameters:
            sim_folder (str): Path to the folder containing simulation results.
            min_time (float): Minimum simulation time for a run to be considered valid.
        """
        self.sim_folder = sim_folder
        self.min_time = min_time
        self.results = self._load_all()

    def _load_all(self):
        """
        Load all valid simulation results from the folder.

        Returns:
            list[SimulationResult]: List of SimulationResult objects.
        """
        results = []
        for fname in sorted(os.listdir(self.sim_folder)):
            if fname.endswith('.npz') and is_valid_simulation(os.path.join(self.sim_folder, fname), self.min_time):
                sim_path = os.path.join(self.sim_folder, fname)
                sim_id = fname.split('_')[-1].replace('.npz', '')
                param_path = os.path.join(self.sim_folder, f'parameters_{sim_id}.csv')
                if not os.path.exists(param_path):
                    continue
                param_df = pd.read_csv(param_path)
                param_dict = dict(zip(param_df['Parameter'], param_df['Value']))
                sim_data = np.load(sim_path)
                results.append(SimulationResult(sim_path, param_path, sim_id, param_dict, sim_data))
        return results

    def __getitem__(self, idx):
        """
        Get a SimulationResult by index.

        Parameters:
            idx (int): Index of the result.

        Returns:
            SimulationResult: The simulation result at the given index.
        """
        return self.results[idx]

    def __len__(self):
        """
        Get the number of simulation results loaded.

        Returns:
            int: Number of results.
        """
        return len(self.results)

    def get_all_params(self, key):
        """
        Get a list of a specific parameter from all results.

        Parameters:
            key (str): Parameter name.

        Returns:
            list: List of parameter values.
        """
        return [r.get_param(key) for r in self.results]

    def get_all_max_times(self):
        """
        Get a list of the maximum simulation times from all results.

        Returns:
            list: List of max times.
        """
        return [r.get_max_time() for r in self.results]

    # Add more batch getters as needed
