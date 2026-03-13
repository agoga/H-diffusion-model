#!/usr/bin/env python3
# gen_tasks_nofast.py — build tasks + config in ONE file (single_sweep_config.json)
from __future__ import annotations
import json
from itertools import product

# ---------- Sweep (everything "slow") ----------
sweep_lists = {
    # ---- AlOx (A) ----
    "A_trap":               [0.9, 1.0,1.1],
    "A_detrap":             [2.05, 2.15, 2.25, 2.35, 2.45],
    "A_trap_attemptfreq":   [1e12, 5e12, 1e13],

    # ---- poly-Si (B) ----
    "B_detrap":             [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    "B_trap_attemptfreq":   [1e12, 5e12, 1e13],

    # ---- SiOx (C) ----
    "C_detrap":             [1.2, 1.3, 1.45, 1.50, 1.55, 1.6],
    "C_trap_attemptfreq":   [1e12, 5e12, 1e13],

    # ---- c-Si (D) ----
    "D_detrap":             [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    "D_trap_attemptfreq":   [1e12, 5e12, 1e13],
}

# ---------- Config used by runner ----------
config = {
    # anneal sweep
    "anneal_sweep": {
        "fire_temp_C": 650,
        "fire_s": 10,
        "anneal_temps_C": [200, 225, 250,275],
        "anneal_s": 1_500_000,
        "include_room": False,
        "room_C": 27,
        "room_s": 0,
    },

    # single firing sweep (fixed anneal @ 250 °C)
    "firing_sweep_1": {
        "firing_temps_C": [650, 700, 750, 800],
        "anneal_temp_C": 250,
        "fire_s": 10,
        "anneal_s": 1_500_000,
        "include_room": False,
        "room_C": 27,
        "room_s": 0,
    },
}

# ---------- Build tasks ----------
keys = list(sweep_lists.keys())
vals = [sweep_lists[k] for k in keys]
tasks = [dict(zip(keys, combo)) for combo in product(*vals)]
ntasks = len(tasks)
print(f"Generated {ntasks} parameter combinations.")

# ---------- Estimate total data size ----------
npz_file_size = 1  # MB per npz
n_anneal = len(config["anneal_sweep"]["anneal_temps_C"])
n_firing = len(config["firing_sweep_1"]["firing_temps_C"])
n_sweeps = n_anneal + n_firing

est_files = ntasks * n_sweeps
est_size_gb = est_files * npz_file_size / 1024
print(f"Estimated output: {est_files:,} npz files × {npz_file_size} MB ≈ {est_size_gb:.1f} GB total")

# ---------- Write bundle ----------
bundle = {"tasks": tasks, "config": config}
with open("single_sweep_config.json", "w") as f:
    json.dump(bundle, f, indent=2)
print("Wrote single_sweep_config.json")
