#!/usr/bin/env python3
# run_tasks_sharded.py — single-arg driver (everything from JSON), one figure per task with all fast columns
from __future__ import annotations
import os, json, gc, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourstates import run_or_load_sim
from helpers import (
    make_stem_compact, short_param_tag, format_value_for_title,
    extract_stage_curves, _thin_xy
)
from vizkit import map_colors_low_to_high, draw_curves, legend_handles

def build_schedule(fire_C: int, anneal_C: int, *, fire_s: int, anneal_s: int,
                   include_room: bool=False, room_C: int=27, room_s: int=0) -> str:
    parts = [f"{int(fire_s)}:{int(fire_C)}C"]
    if include_room and room_s > 0:
        parts.append(f"{int(room_s)}:{int(room_C)}C")
    parts.append(f"{int(anneal_s)}:{int(anneal_C)}C")
    return ", ".join(parts)

def pick_shard_range(total: int) -> range:
    """Shard by SLURM env only. If not present, run all."""
    env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    env_cnt = os.environ.get("SLURM_ARRAY_TASK_COUNT") or os.environ.get("SLURM_ARRAY_TASK_MAX")
    if env_idx is None or env_cnt is None:
        return range(0, total)
    i = int(env_idx); m = int(env_cnt)
    if i >= m:  # accept 1-based IDs
        i -= 1
    size = total // m
    rem  = total % m
    start = i * size + min(i, rem)
    end   = start + size + (1 if i < rem else 0)
    return range(start, end)

def make_multicol_figure(ncols: int, col_labels: list[str]):
    # one grid: 8 rows (two-row blocks ×3 with separators) × ncols columns
    fig_w = max(5 * ncols, 5)  # scale width with columns
    fig = plt.figure(figsize=(fig_w, 18))
    gs = fig.add_gridspec(nrows=8, ncols=ncols, height_ratios=[1,1,0.01,1,1,0.01,1,1])
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    axes = np.empty((8, ncols), dtype=object)
    for rr in range(8):
        for cc in range(ncols):
            axes[rr, cc] = None if rr in (2, 5) else fig.add_subplot(gs[rr, cc])

    # titles per column
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=14)

    return fig, axes

def render_fast_grid(base_rec: dict,
                     fast_param: str,
                     fast_values: list[float],
                     cfg: dict,
                     out_dir: Path,
                     out_name: str):
    FIRE_S                        = int(cfg["fire_s"])
    UT_INDEX                      = int(cfg["ut_index"])
    SCALE_Y                       = float(cfg["scale_y"])
    INCLUDE_ROOM                  = bool(cfg["include_room"])
    STAGE_ANN                     = 2 if INCLUDE_ROOM else 1

    firing_list                   = [int(x) for x in cfg["firing_list"]]
    anneal_list                   = [int(x) for x in cfg["anneal_list"]]
    default_anneal                = int(cfg["default_anneal"])
    default_anneal_time           = int(cfg["default_anneal_time"])
    alt_anneal_after_firing       = int(cfg["alt_anneal_after_firing"])
    alt_anneal_after_firing_time  = int(cfg["alt_anneal_after_firing_time"])
    default_firing                = int(cfg["default_firing"])
    anneal_sweep_time             = int(cfg["anneal_sweep_time"])
    dpi                           = int(cfg["dpi"])

    ncols = max(1, len(fast_values))
    col_labels = [f"{short_param_tag(fast_param)}={format_value_for_title(fast_param, v)}"
                  for v in fast_values]

    fig, axes = make_multicol_figure(ncols, col_labels)
    anneal_colors = map_colors_low_to_high(list(anneal_list))
    firing_colors = map_colors_low_to_high(list(firing_list))
    x_lim_hours=3e-3

    # build + draw each column
    for j, fv in enumerate(fast_values):
        rec0 = dict(base_rec); rec0[fast_param] = float(fv)

        # anneal sweep (rows 0–1) → x in hours
        anneal_curves = []
        for aC in sorted(anneal_list):
            rec = dict(rec0)
            rec["temp_schedule"] = build_schedule(default_firing, aC, fire_s=FIRE_S, anneal_s=anneal_sweep_time)
            rec['output_dt']=10
            res, _ = run_or_load_sim(rec=rec)
            t, y, _ = extract_stage_curves(res, stage_index=STAGE_ANN, ut_index=UT_INDEX, scale=SCALE_Y)
            if t.size and y.size:
                t, y = _thin_xy(t, y)
                print(f"[anneal] {aC}°C → {t.size:6d} points (first={t[0]:.2e}, last={t[-1]:.2e})")
                anneal_curves.append((aC, t, y))

            del res, t, y
        for row in (0, 1):
            ax = axes[row, j]
            draw_curves(ax, anneal_curves, xscale=3600.0, ylog=(row == 0), color_map=anneal_colors)
            if j == 0:
                ax.set_ylabel("[H] in SiOx (LOG)" if row == 0 else "[H] in SiOx (LINEAR)")
            ax.set_xlabel("Time (hours)")
            
            # clamp left x-limit to ≥ 1e-3 hours
            xmin, xmax = ax.get_xlim()
            if xmin < x_lim_hours:
                ax.set_xlim(x_lim_hours, xmax)


        # firing @ default anneal (rows 3–4) → x in minutes
        firing_curves = []
        for fC in sorted(firing_list):
            rec = dict(rec0)
            rec["temp_schedule"] = build_schedule(fC, default_anneal, fire_s=FIRE_S, anneal_s=default_anneal_time)
            res, _ = run_or_load_sim(rec=rec)
            t, y, _ = extract_stage_curves(res, stage_index=STAGE_ANN, ut_index=UT_INDEX, scale=SCALE_Y)
            if t.size and y.size:
                t, y = _thin_xy(t, y)
                print(f"[firing] {fC}°C → {t.size:6d} points (first={t[0]:.2e}, last={t[-1]:.2e})")
                firing_curves.append((fC, t, y))

            del res, t, y
        for row in (3, 4):
            ax = axes[row, j]
            draw_curves(ax, firing_curves, xscale=60.0, ylog=(row == 3), color_map=firing_colors)
            if j == 0:
                ax.set_ylabel("[H] in SiOx (LOG)" if row == 3 else "[H] in SiOx (LINEAR)")
            ax.set_xlabel("Time (minutes)")

        # firing @ alt anneal (rows 6–7) → x in hours
        firing_curves_alt = []
        for fC in sorted(firing_list):
            rec = dict(rec0)
            rec["temp_schedule"] = build_schedule(fC, alt_anneal_after_firing, fire_s=FIRE_S, anneal_s=alt_anneal_after_firing_time)
            rec['output_dt']=10
            res, _ = run_or_load_sim(rec=rec)
            t, y, _ = extract_stage_curves(res, stage_index=STAGE_ANN, ut_index=UT_INDEX, scale=SCALE_Y)
            if t.size and y.size:
                t, y = _thin_xy(t, y)
                print(f"[firing_alt] {fC}°C → {t.size:6d} points (first={t[0]:.2e}, last={t[-1]:.2e})")
                firing_curves_alt.append((fC, t, y))

            del res, t, y
        for row in (6, 7):
            ax = axes[row, j]
            draw_curves(ax, firing_curves_alt, xscale=3600.0, ylog=(row == 6), color_map=firing_colors)
            if j == 0:
                ax.set_ylabel("[H] in SiOx (LOG)" if row == 6 else "[H] in SiOx (LINEAR)")
            ax.set_xlabel("Time (hours)")
            
            # clamp left x-limit to ≥ 1e-3 hours
            xmin, xmax = ax.get_xlim()
            if xmin < x_lim_hours:
                ax.set_xlim(x_lim_hours, xmax)


    # put legends over the middle column
    mid = max(0, ncols // 2)
    axes[0, mid].legend(handles=legend_handles(anneal_colors, "{}°C anneal"),
                        loc="best", frameon=False, fontsize=9)
    axes[3, mid].legend(handles=legend_handles(firing_colors, "{}°C fire"),
                        loc="best", frameon=False, fontsize=9)
    axes[6, mid].legend(handles=legend_handles(firing_colors, "{}°C fire"),
                        loc="best", frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=int(cfg["dpi"]))
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Run sweep using a single JSON config.")
    ap.add_argument("--config", default="sweep_config.json")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        bundle = json.load(f)

    # NOTE: if you accidentally pass a plain tasks.json, bundle will be a list and the next line would fail.
    # Ensure you're pointing to the combined config (with "tasks", "fast_param", etc.).
    tasks       = bundle["tasks"]
    fast_param  = bundle["fast_param"]
    fast_values = bundle["fast_values"]
    fast_index  = bundle.get("fast_index", None)
    cfg         = bundle["config"]

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # pin to one fast index (via JSON) if desired
    if fast_index is not None:
        idx = int(fast_index)
        if not (0 <= idx < len(fast_values)):
            raise ValueError(f"fast_index out of range (0..{len(fast_values)-1})")
        fast_values = [fast_values[idx]]

    rng = pick_shard_range(len(tasks))
    print(f"[driver] total tasks: {len(tasks)} | running indices: {rng.start}..{rng.stop-1}")

    # render one FIGURE per task, with all columns (fast_values)
    for ti in rng:
        base = dict(tasks[ti])

        # short, unique filename:
        # - slow params compact stem (helpers.make_stem_compact is already filesystem-safe)
        # - plus fast-param tag indicating number of columns (e.g., bHn6)
        slow_keys_sorted = sorted(base.keys())
        slow_stem = make_stem_compact(base, slow_keys_sorted)  # no double "__"
        fast_tag  = f"{short_param_tag(fast_param)}n{len(fast_values)}"
        out_name  = f"{slow_stem}_{fast_tag}.png"

        render_fast_grid(base, fast_param, fast_values, cfg, out_dir, out_name)
        gc.collect()

    print("[driver] done.")

if __name__ == "__main__":
    main()
