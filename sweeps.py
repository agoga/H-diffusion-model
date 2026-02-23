# sweeps.py
# Authors: Adam Goga 
from __future__ import annotations

from collections.abc import Iterable
import numpy as np

import helpers as h

# ------------------------------ one-at-a-time param sweep ------------------------------
# Builds param dicts around the TRUE defaults from helpers.build_params(), varying
# exactly one key at a time by symmetric additive offsets, and (optionally) including
# the exact defaults once at the front.

_ATTEMPT_KEYS: set[str] = {
    "A_detrap_attemptfreq", "B_detrap_attemptfreq",
    "C_detrap_attemptfreq", "D_detrap_attemptfreq",
    "A_trap_attemptfreq",   "B_trap_attemptfreq",
    "C_trap_attemptfreq",   "D_trap_attemptfreq",
}

def _default_params() -> dict[str, float]:
    """Return canonical defaults as a plain dict (single source of truth)."""
    return dict(vars(h.build_params()))

def _symmetric_offsets(max_delta: float, n: int) -> list[float]:
    """
    Make n symmetric offsets in [-max_delta, +max_delta], EXCLUDING 0.
    Example: max_delta=0.1, n=4 → [-0.1, -0.05, +0.05, +0.1]
    """
    if n <= 0 or max_delta <= 0:
        return []
    vals = np.linspace(-max_delta, max_delta, n + 1)
    return [float(v) for v in vals if abs(v) > 1e-15]

def make_default_param_sweep(
    max_diff: dict[str, float],
    num_diff: int,
    *,
    include_default: bool = True,
) -> list[dict[str, float]]:
    """
    Create a one-at-a-time additive sweep around the true defaults.

    Arguments
    ---------
    max_diff : dict[str, float]
        For each param to vary, give the maximum absolute *additive* delta
        around the default. (These are deltas, not the defaults themselves.)
    num_diff : int
        Number of symmetric samples per param (default excluded automatically).
        E.g., num_diff=4 and delta=0.1 → [-0.1, -0.05, +0.05, +0.1].
    include_default : bool
        If True, include the exact defaults once at the start.

    Returns
    -------
    list[dict[str, float]]
        First item is defaults (if requested), then one-at-a-time variants.
    """
    base = _default_params()
    out: list[dict[str, float]] = []

    if include_default:
        out.append(base.copy())

    for key, delta in max_diff.items():
        if delta is None or delta <= 0:
            continue
        for off in _symmetric_offsets(float(delta), num_diff):
            trial = base.copy()
            new_val = trial[key] + off
            # attempt frequencies must remain positive
            if key in _ATTEMPT_KEYS and new_val <= 0:
                continue
            trial[key] = new_val
            out.append(trial)
    return out


# ------------------------------ schedule sweep helpers --------------------------------
# Simple, explicit schedule constructors that return a list of schedule strings
# produced by helpers.build_schedule(...). These keep ALL logic for parsing/formatting
# centralized in helpers and fourstates.

def make_unfired_sweep(
    anneal_temp: int | float,
    *,
    fire_temp: int | float = 25,
    fire_s: int,
    anneal_s: int,
) -> list[str]:
    """
    For a fixed firing temp, build one schedule per anneal temperature.
    Returns schedule strings like '100:650C, 5000000:250C' via helpers.build_schedule.
    """
    schedule = [
        h.build_schedule(
            fire_C=float(fire_temp),
            anneal_C=float(anneal_temp),
            fire_s=int(fire_s),
            anneal_s=int(anneal_s),
            include_room=False)
    ]
    stage_names=["firing","annealing"]
    return schedule, stage_names

def make_annealing_sweep(
    anneal_temps: Iterable[int | float],
    *,
    fire_temp: int | float,
    fire_s: int,
    anneal_s: int,
    include_room: bool,
    room_temp: int | float,
    room_s: int,
    n_cycles: int = 1,   
) -> list[str]:
    """
    For a fixed firing temp, build one schedule per anneal temperature.
    Returns schedule strings like '100:650C, 5000000:250C' via helpers.build_schedule.
    """
    schedule = [
        h.build_schedule(
            fire_C=float(fire_temp),
            anneal_C=float(Ta),
            fire_s=int(fire_s),
            anneal_s=int(anneal_s),
            include_room=include_room,
            room_C=float(room_temp),
            room_s=int(room_s),
            n_cycles=n_cycles,
        )
        for Ta in anneal_temps
    ]
    if include_room:
        stage_names=["firing","room","annealing"]
    else:
        stage_names=["firing","annealing"]
    return schedule, stage_names

def make_firing_sweep(
    firing_temps: Iterable[int | float],
    *,
    anneal_temp: int | float,
    fire_s: int,
    anneal_s: int,
    include_room: bool,
    room_temp: int | float,
    room_s: int,
    n_cycles: int = 1,   # NEW
) -> list[str]:
    """
    For a fixed anneal temp, build one schedule per firing temperature.
    Returns schedule strings via helpers.build_schedule.
    """
    schedule=[
        h.build_schedule(
            fire_C=float(Tf),
            anneal_C=float(anneal_temp),
            fire_s=int(fire_s),
            anneal_s=int(anneal_s),
            include_room=include_room,
            room_C=float(room_temp),
            room_s=int(room_s),
            n_cycles=n_cycles,   # NEW
        )
        for Tf in firing_temps
    ]
    if include_room:
        stage_names=["firing","room","annealing"]
    else:
        stage_names=["firing","annealing"]
    return schedule, stage_names

def make_fancy_firing(
    
) -> list[str]:
    """
    For a fixed anneal temp, build one schedule per firing temperature.
    Returns schedule strings via helpers.build_schedule.
    """
    return ["2:650C, 180:450C"]#,"30:350C, 1:650C, 180:350C"]




# --- add to sweeps.py ---------------------------------------------------------
from itertools import product
from typing import Mapping, Sequence, Iterable, Dict, Any, Optional
import helpers as h

def make_product_sweep(
    axes: Mapping[str, Sequence[Any]],
    *,
    base: Optional[Dict[str, Any]] = None,
    strict_keys: bool = False,
    include_base: bool = False,
) -> Iterable[Dict[str, Any]]:
    """
    Build a full-factorial (Cartesian product) sweep of parameters.

    Parameters
    ----------
    axes
        Dict mapping parameter name -> list/tuple of values to try.
        Example:
            {
              "A_detrap": [2.2, 2.3, 2.4],
              "B_detrap": [1.1, 1.2],
              "C_detrap_attemptfreq": [1e11, 1e12],
            }
    base
        Optional base record to start from. If None, uses your canonical
        defaults from helpers.build_params().
    strict_keys
        If True, raise a KeyError when an axis key is not present in `base`.
        If False, allow introducing new keys (they’ll just be added).
    include_base
        If True, include the unmodified base record once as the first element.

    Returns
    -------
    Iterable[Dict[str, Any]]
        Each dict is a full simulation record (copy of `base` + one value for
        every axis key), suitable for fourstates.run_or_load_sim(rec=...).

    Notes
    -----
    - This function does *not* do any deduping; if an axis uses the base value,
      you may get duplicates unless you filter externally.
    - For very large grids, consider sharding via SLURM array (as in your driver).
    """
    # 1) Start from canonical defaults unless a custom base was provided
    base_rec: Dict[str, Any] = dict(vars(h.build_params())) if base is None else dict(base)

    # 2) Optional validation
    if strict_keys:
        missing = [k for k in axes.keys() if k not in base_rec]
        if missing:
            raise KeyError(f"Axis contains keys not in base: {missing}")

    # 3) Optionally yield the unmodified base first
    if include_base:
        yield dict(base_rec)

    # 4) Cartesian product over all axis values
    if not axes:
        return  # nothing to vary

    keys = list(axes.keys())
    value_lists = [list(axes[k]) for k in keys]
    for combo in product(*value_lists):
        rec = dict(base_rec)
        for k, v in zip(keys, combo):
            rec[k] = v
        yield rec
# --- end addition --------------------------------------------------------------
