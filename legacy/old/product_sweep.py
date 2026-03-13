#!/usr/bin/env python3
# product_sweep.py  (modernized thin wrapper)
from __future__ import annotations
import argparse
from pathlib import Path
from itertools import product
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourstates import build_params, run_or_load_sim
from helpers import (
    make_stem_compact, short_param_tag, format_value_for_title,
    extract_stage_curves, schedule_tag
)
from vizkit import map_colors_low_to_high, draw_curves, legend_handles

# basic schedule builder copied from your convention (seconds explicit)
def build_schedule(fire_C: int, anneal_C: int, *, fire_s: int, anneal_s: int,
                   include_room: bool=False, room_C: int=27, room_s: int=0) -> str:
    parts = [f"{int(fire_s)}:{int(fire_C)}C"]
    if include_room and room_s > 0:
        parts.append(f"{int(room_s)}:{int(room_C)}C")
    parts.append(f"{int(anneal_s)}:{int(anneal_C)}C")
    return ", ".join(parts)

UT_INDEX = 2
SCALE_Y  = 1e22
FIRE_S   = 10
DPI      = 200
STAGE_ANN = 1  # (0=firing, [1=anneal] without room)

def product_sweep_dualtemp(sweep_spec: dict, fast_param: str,
                           *, firing_list=(650,700,750), anneal_list=(250,300,350),
                           default_firing=677, default_anneal=350,
                           out_dir='./product_sweep_dualtemp/', dpi=DPI):
    out_dir = Path(out_dir); (out_dir / "npz").mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "plots"; img_dir.mkdir(parents=True, exist_ok=True)

    fast_values = list(sweep_spec[fast_param])
    slow_keys = [k for k in sweep_spec.keys() if k != fast_param]
    slow_combos = [dict(zip(slow_keys, vals)) for vals in product(*[sweep_spec[k] for k in slow_keys])] if slow_keys else [{}]

    anneal_colors = map_colors_low_to_high(list(anneal_list))
    firing_colors = map_colors_low_to_high(list(firing_list))

    for slow_combo in slow_combos:
        ncols = len(fast_values)
        fig = plt.figure(figsize=(5 * ncols, 14))
        gs = fig.add_gridspec(nrows=5, ncols=ncols, height_ratios=[1,1,0.05,1,1])
        fig.subplots_adjust(hspace=0.2)
        axes = np.empty((5, ncols), dtype=object)
        for r in range(5):
            for c in range(ncols):
                axes[r,c] = None if r == 2 else fig.add_subplot(gs[r,c])

        per_col = []
        labels  = []

        for fv in fast_values:
            base_rec = dict(slow_combo); base_rec[fast_param] = fv
            labels.append(f"{short_param_tag(fast_param)}={format_value_for_title(fast_param, fv)}")

            # anneal sweep (firing fixed)
            anneal_curves = []
            for aC in sorted(anneal_list):
                rec = dict(base_rec)
                rec["temp_schedule"] = build_schedule(default_firing, aC, fire_s=FIRE_S, anneal_s=1_000_000)
                stem = make_stem_compact(rec, list(sweep_spec.keys()))
                simres, _ = run_or_load_sim(rec=rec)
                t, y, _ = extract_stage_curves(simres, stage_index=STAGE_ANN, ut_index=UT_INDEX, scale=SCALE_Y)
                if t.size and y.size: anneal_curves.append((aC, t, y))

            # firing sweep (anneal fixed)
            firing_curves = []
            for fC in sorted(firing_list):
                rec = dict(base_rec)
                rec["temp_schedule"] = build_schedule(fC, default_anneal, fire_s=FIRE_S, anneal_s=100_000)
                stem = make_stem_compact(rec, list(sweep_spec.keys()))
                simres, _ = run_or_load_sim(rec=rec)
                t, y, _ = extract_stage_curves(simres, stage_index=STAGE_ANN, ut_index=UT_INDEX, scale=SCALE_Y)
                if t.size and y.size: firing_curves.append((fC, t, y))

            per_col.append((anneal_curves, firing_curves))

        # plotting
        for j, (label, (anneal_curves, firing_curves)) in enumerate(zip(labels, per_col)):
            # anneal rows → hours
            xunit, xscale = "hours", 3600.0
            for row in (0,1):
                ax = axes[row, j]
                draw_curves(ax, anneal_curves, xscale=xscale, ylog=(row==0), color_map=anneal_colors)
                if j == 0: ax.set_ylabel("[H] in SiOx (LOG)" if row==0 else "[H] in SiOx (LINEAR)")
                ax.set_title(label, fontsize=14)
                ax.set_xlabel(f"Time ({xunit})")
            # firing rows → minutes
            xunit, xscale = "minutes", 60.0
            for row in (3,4):
                ax = axes[row, j]
                draw_curves(ax, firing_curves, xscale=xscale, ylog=(row==3), color_map=firing_colors)
                if j == 0: ax.set_ylabel("[H] in SiOx (LOG)" if row==3 else "[H] in SiOx (LINEAR)")
                ax.set_xlabel(f"Time ({xunit})")

        # legends
        mid = max(0, (len(fast_values)//2))
        axes[0, mid].legend(handles=legend_handles(anneal_colors, "{}°C anneal"), loc="best", frameon=False, fontsize=9)
        axes[3, mid].legend(handles=legend_handles(firing_colors, "{}°C fire"),   loc="best", frameon=False, fontsize=9)

        slow_label = ", ".join([f"{short_param_tag(k)}={format_value_for_title(k, v)}" for k, v in slow_combo.items()]) or "default"
        fig.suptitle(f"{schedule_tag(build_schedule(default_firing, default_anneal, fire_s=FIRE_S, anneal_s=100_000))} — fixed: {slow_label}", fontsize=15)
        fig.tight_layout()

        img_path = (out_dir/"plots"/f"product_fast__{short_param_tag(fast_param)}__{slow_label or 'default'}.png")
        fig.savefig(img_path, dpi=dpi); plt.close(fig)
        print(f"✔ saved: {img_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-param", required=True)
    ap.add_argument("--out-dir", default="./product_sweep_dualtemp/")
    ap.add_argument("--dpi", type=int, default=DPI)
    # quick demo defaults — edit as needed or pass a JSON file like before
    args = ap.parse_args()

    sweep_spec = {
        "A_detrap": [2.3],
        "B_detrap": [0.7,0.95,1.1],
        "C_detrap": [1.3],
        "D_detrap": [0.9],
        "A_trap_attemptfreq": [1e12],
        "B_trap_attemptfreq": [1e12],
        "C_trap_attemptfreq": [1e12],
    }
    product_sweep_dualtemp(sweep_spec, args.fast_param, out_dir=args.out_dir, dpi=args.dpi)

if __name__ == "__main__":
    main()
