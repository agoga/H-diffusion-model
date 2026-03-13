#!/usr/bin/env python3
# run_sweep_json.py — data-only sweeps with pretest gate + per-shard CSV
from __future__ import annotations

import os, sys, csv, json, argparse
from pathlib import Path
import numpy as np

import sweeps
import helpers as h
import fourstates as fs
import simulation_manager as sm
import nrel_exp as nrel


# ---------------------------
# Small utilities (no dupes)
# ---------------------------
def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None or v == "":
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)

def _slice_for_shard(N: int, shard_id: int, num_shards: int) -> range:
    if num_shards <= 1:
        return range(0, N)
    per = N // num_shards
    rem = N % num_shards
    start = shard_id * per + min(shard_id, rem)
    stop  = start + per + (1 if shard_id < rem else 0)
    return range(start, min(stop, N))

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _append_row_csv(out_path: Path, row: dict) -> None:
    new_file = not out_path.exists()
    _ensure_parent(out_path)
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)

def _tpeak_hr_from_series(t_h, y_norm):
    if len(t_h) == 0 or len(y_norm) == 0 or not np.isfinite(y_norm).any():
        return None
    i = int(np.nanargmax(y_norm))
    return float(t_h[i])


# --------------------------------
# Pretest: firing temp → t_peak ↑
# --------------------------------
# def pretest_firing_peak_monotone(
#     base_params_ns,
#     *,
#     firing_temps_C: list[int],
#     anneal_temp_C: int,
#     schedules: list[str],
#     layer: str = "C",
#     kind: str = "trapped",
#     results_dir: str | None = None,
#     petsc_options: dict | None = None,
#     verbose: bool = False,
# ) -> tuple[bool, list[tuple[int, float]]]:
#     """
#     Build a manager over 'schedules' for these firing temps, extract normalized
#     sim peak times at 'anneal_temp_C', and require strict monotonic increase.
#     """
#     mgr = sm.SimulationManager(
#         base_params=base_params_ns,
#         temp_schedules=schedules,
#         results_dir=results_dir,
#         petsc_options=petsc_options,
#         verbose=verbose,
#     )

#     peaks: list[tuple[int, float]] = []
#     for fC in firing_temps_C:
#         t_sim_h, y_sim = nrel._pick_sim_for_anneal_T(
#             mgr,
#             firing_temp_C=fC,
#             anneal_temp_C=anneal_temp_C,
#             layer=layer,
#             kind=kind,
#         )
#         if t_sim_h is None:
#             peaks.append((fC, float("nan")))
#             continue
#         tN, yN, *_ = h.normalize_baseline_peak(t_sim_h, y_sim, t0_window=(0.1, 0.3))
#         tpk = _tpeak_hr_from_series(tN, yN)
#         peaks.append((fC, float("nan") if tpk is None else float(tpk)))

#     vals = [t for _, t in peaks]
#     ok = all(np.isfinite(vals)) and all(vals[i+1] > vals[i] for i in range(len(vals)-1))
    
    
def pretest_firing_peak_monotone(
    base_params_ns,
    *,
    firing_temps_C: list[int],
    anneal_temp_C: int,
    schedules: list[str],
    layer: str = "C",
    kind: str = "trapped",
    results_dir: str | None = None,
    petsc_options: dict | None = None,
    verbose: bool = False,
) -> tuple[bool, list[tuple[int, float]]]:

    mgr = sm.SimulationManager(
        base_params=base_params_ns,
        temp_schedules=schedules,
        results_dir=results_dir,
        petsc_options=petsc_options,
        verbose=verbose,
    )

    # Collect peaks as (firing_temp_C, t_peak_hours or NaN)
    peaks: list[tuple[int, float]] = []
    for fC in firing_temps_C:
        t_sim_h, y_sim = nrel._pick_sim_for_anneal_T(
            mgr,
            firing_temp_C=fC,
            anneal_temp_C=anneal_temp_C,
            layer=layer,
            kind=kind,
        )
        if t_sim_h is None:
            peaks.append((fC, float("nan")))
            continue

        tN, yN, *_ = h.normalize_baseline_peak(t_sim_h, y_sim, exp=False)
        tpk = _tpeak_hr_from_series(tN, yN)
        peaks.append((fC, float("nan") if tpk is None else float(tpk)))

    # Sort by firing temperature so "increasing" is unambiguous
    peaks_sorted = sorted(peaks, key=lambda x: x[0])
    temps = [T for T, _ in peaks_sorted]
    vals  = [p for _, p in peaks_sorted]

    # Need all finite and at least 2 points
    if len(vals) < 2 or not all(np.isfinite(vals)):
        return False, peaks

    # 1) Strictly increasing with firing T
    monotone_ok = all(vals[i+1] > vals[i] for i in range(len(vals)-1))

    # 2) Exists T_hi > T_lo with peak_hi ≥ 2 × peak_lo
    factor2_ok = any(
        vals[j] >= 2.0 * vals[i]
        for i in range(len(vals))
        for j in range(i+1, len(vals))
    )

    ok = monotone_ok and factor2_ok
    return ok, peaks

# -------------
# Main driver
# -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="single_sweep_config.json")
    ap.add_argument("--results-dir", default="sim_data")
    ap.add_argument("--csv-dir", default="shard_csv")  # centralized here
    ap.add_argument("--verbose", action="store_true")
    # optional CLI sharding if not on SLURM
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    args = ap.parse_args()

    # ---------- Centralized paths & shard info ----------
    SHARD_ID   = _env_int("SLURM_ARRAY_TASK_ID",  args.shard_id or 0)
    NUM_SHARDS = _env_int("SLURM_ARRAY_TASK_COUNT", args.num_shards or 1)

    RESULTS_DIR = Path(args.results_dir).resolve()
    CSV_DIR     = Path(args.csv_dir).resolve()
    OUT_PATH    = CSV_DIR / f"scores_shard_{SHARD_ID}.csv"
    
    if OUT_PATH.exists():
        OUT_PATH.unlink()  

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load config / tasks ----------
    cfg   = json.load(open(args.config, "r"))
    tasks = list(cfg["tasks"])
    conf  = cfg["config"]

    # ---------- Build schedules (match your notebook) ----------
    # Anneal sweep schedules (fixed firing)
    a = conf["anneal_sweep"]
    schedules_anneal = sweeps.make_annealing_sweep(
        anneal_temps=a["anneal_temps_C"],
        fire_temp=a["fire_temp_C"],
        fire_s=a["fire_s"],
        anneal_s=a["anneal_s"],
        include_room=a["include_room"],
        room_temp=a["room_C"],
        room_s=a["room_s"],
    )

    # Firing groups (for pretest)
    firing_groups = []
    if "firing_sweep_1" in conf:
        f1 = conf["firing_sweep_1"]
        s_f1 = sweeps.make_firing_sweep(
            firing_temps=f1["firing_temps_C"],
            anneal_temp=f1["anneal_temp_C"],
            fire_s=f1["fire_s"],
            anneal_s=f1["anneal_s"],
            include_room=f1["include_room"],
            room_temp=f1["room_C"],
            room_s=f1["room_s"],
        )
        firing_groups.append({
            "label": "firing_sweep_1",
            "firing_temps_C": list(f1["firing_temps_C"]),
            "anneal_temp_C": int(f1["anneal_temp_C"]),
            "schedules": s_f1,
        })
    if "firing_sweep_2" in conf:
        f2 = conf["firing_sweep_2"]
        s_f2 = sweeps.make_firing_sweep(
            firing_temps=f2["firing_temps_C"],
            anneal_temp=f2["anneal_temp_C"],
            fire_s=f2["fire_s"],
            anneal_s=f2["anneal_s"],
            include_room=f2["include_room"],
            room_temp=f2["room_C"],
            room_s=f2["room_s"],
        )
        firing_groups.append({
            "label": "firing_sweep_2",
            "firing_temps_C": list(f2["firing_temps_C"]),
            "anneal_temp_C": int(f2["anneal_temp_C"]),
            "schedules": s_f2,
        })

    # ---------- Sharding ----------
    idx_range = _slice_for_shard(len(tasks), SHARD_ID, NUM_SHARDS)

    # ---------- PETSc options (override if needed) ----------
    petsc_opts = None

    print(f"[run] shard {SHARD_ID}/{NUM_SHARDS} → tasks {idx_range.start}..{idx_range.stop-1} "
          f"({idx_range.stop-idx_range.start} tasks) × anneal_schedules={len(schedules_anneal)}")

    # ---------- Load experiment once ----------
    expt = nrel.load_nrel_anneal_csv(
        "/home/agoga/sandbox/diffusion/annealing_sweep_NREL.csv"
    )

    any_failed = False

    # Notebook scorer inputs
    anneal_fire_C  = conf["anneal_sweep"]["fire_temp_C"]
    anneal_temps_C = conf["anneal_sweep"]["anneal_temps_C"]

    for i in idx_range:
        try:
            params = dict(tasks[i])  # one param-set
            bp = h.build_params(**params)  # no schedule baked in

            # -------- PRETEST: pass if ANY firing group is monotone --------
            pretest_pass = False
            pretest_tables = {}

            for grp in firing_groups:
                ok, table = pretest_firing_peak_monotone(
                    bp,
                    firing_temps_C=grp["firing_temps_C"],
                    anneal_temp_C=grp["anneal_temp_C"],
                    schedules=grp["schedules"],
                    layer=params.get("layer", "C"),
                    kind=params.get("kind", "trapped"),
                    results_dir=str(RESULTS_DIR),
                    petsc_options=petsc_opts,
                    verbose=args.verbose,
                )
                pretest_tables[grp["label"]] = table
                pretest_pass = pretest_pass or ok

            sig = fs.sim_signature(fs.build_sim_record(params=bp))

            if not pretest_pass:
                # Minimal FAIL row (same out path as success)
                row = {
                    "sig": sig,
                    "overall": float("nan"),
                    "pretest": "FAIL",
                }
                for k, v in params.items():
                    row[f"param_{k}"] = v

                # (Optional: store peak tables)
                # for label, table in pretest_tables.items():
                #     row[f"{label}_peaks"] = ",".join(
                #         f"{fC}:{tpk:.4g}" if np.isfinite(tpk) else f"{fC}:nan"
                #         for fC, tpk in table
                #     )

                _append_row_csv(OUT_PATH, row)
                continue

            # -------- SCORING: anneal sweep only (like notebook), no plot --------
            mgr = sm.SimulationManager(
                base_params=bp,
                temp_schedules=schedules_anneal,  # anneal-only schedules
                results_dir=str(RESULTS_DIR),
                petsc_options=petsc_opts,
                verbose=args.verbose,
            )

            ax, sim_nums, overall = nrel.plot_exp_vs_sim_normalized_overlay_hours(
                mgr,
                expt_path='/home/agoga/sandbox/diffusion/annealing_sweep_NREL.csv',
                firing_temp_C=anneal_fire_C,
                anneal_temps_C=anneal_temps_C,
                layer=params.get("layer", "C"),
                kind=params.get("kind", "trapped"),
                plot=False,
                print_data=False,
                return_data=True,
            )


            # Success row
            row = {
                "sig": sig,
                "overall": overall,
                "pretest": "OK",
            }
            for k, v in params.items():
                row[f"param_{k}"] = v

            for T, d in sim_nums.items():
                s = d.get("shape_score")
                if not s:
                    continue
                row[f"T{T}_score"]   = s.get("score", np.nan)
                row[f"T{T}_shift"]   = s.get("shift_decades", np.nan)
                row[f"T{T}_overlap"] = s.get("overlap_decades", np.nan)
                # optional extras your scorer returns:
                row[f"T{T}_mae_plateau"] = s.get("mae_plateau", np.nan)
                row[f"T{T}_mae_rise"]    = s.get("mae_rise", np.nan)
                row[f"T{T}_mae_return"]  = s.get("mae_return", np.nan)

            _append_row_csv(OUT_PATH, row)

        except Exception as e:
            any_failed = True
            print(f"[ERROR] task idx={i}: {e}", file=sys.stderr)

    print("[done]")
    if any_failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
