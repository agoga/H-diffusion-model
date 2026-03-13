#!/usr/bin/env python3
# resweep_from_csv.py  (updated to shared helpers/core)
from __future__ import annotations

import argparse, json, csv
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Only use Agg backend when running as script (not when imported)
if __name__ == '__main__':
    matplotlib.use("Agg")
from itertools import product
from fourstates import run_or_load_sim
from helpers import (
    make_stem_compact, short_param_tag, format_value_for_title,
    extract_stage_curves, x_unit_and_scale
)
from vizkit import map_colors_low_to_high, draw_curves, legend_handles

FIRE_S = 10
UT_INDEX = 2
SCALE_Y  = 1e22
DPI = 200
INCLUDE_ROOM = False
STAGE_ANN = 2 if INCLUDE_ROOM else 1

def build_schedule(fire_C: int, anneal_C: int, *, fire_s: int, anneal_s: int,
                   include_room: bool=False, room_C: int=27, room_s: int=0) -> str:
    parts = [f"{int(fire_s)}:{int(fire_C)}C"]
    if include_room and room_s > 0:
        parts.append(f"{int(room_s)}:{int(room_C)}C")
    parts.append(f"{int(anneal_s)}:{int(anneal_C)}C")
    return ", ".join(parts)

def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _default_tag_map():
    return {
        "AH": "A_detrap", "BH": "B_detrap", "CH": "C_detrap", "DH": "D_detrap",
        "kHA": "A_trap_attemptfreq", "kHB": "B_trap_attemptfreq",
        "kHC": "C_trap_attemptfreq", "kHD": "D_trap_attemptfreq",
        "kAH": "A_detrap_attemptfreq", "kBH": "B_detrap_attemptfreq",
        "kCH": "C_detrap_attemptfreq", "kDH": "D_detrap_attemptfreq",
    }

def _coerce_num(s):
    if isinstance(s, (int, float, np.floating)): return float(s)
    if not isinstance(s, str): return s
    t = s.replace('_','.')
    try: return float(t)
    except: return s

def rows_from_csv(path: str | Path, tag_map: dict) -> list[dict]:
    rows = []
    with open(path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            canon = {}
            for k, v in r.items():
                if not k or v in (None, ""): continue
                ck = tag_map.get(k, k)
                canon[ck] = _coerce_num(v)
            rows.append(canon)
    return rows

def sweep_from_rows(
    rows: list[dict],
    fast_param: str,
    fast_values: list[float],
    *,
    firing_list=(650,700,750),
    anneal_list=(250,300,350),
    default_anneal=350,
    default_anneal_time=30_000,
    alt_anneal_after_firing=250,
    alt_anneal_after_firing_time=1_000_000,
    default_firing=677,
    anneal_sweep_time=1_000_000,
    out_dir: Path = Path('./resweep_from_csv/'),
    dpi: int = DPI,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"npz").mkdir(exist_ok=True)
    (out_dir/"plots").mkdir(exist_ok=True)

    all_keys = set().union(*(r.keys() for r in rows))
    tag_map = _default_tag_map()
    fast_param = tag_map.get(fast_param, fast_param)
    slow_keys = sorted([k for k in all_keys if k != fast_param])

    # group rows by slow params
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = tuple((k, r.get(k, None)) for k in slow_keys)
        groups[key].append(r)

    anneal_colors = map_colors_low_to_high(list(anneal_list))
    firing_colors = map_colors_low_to_high(list(firing_list))

    keys_in_play = [k for k in slow_keys + [fast_param]]

    for slow_key, rlist in groups.items():
        ncols = max(1, len(fast_values))
        fig = plt.figure(figsize=(5 * ncols, 18))
        gs = fig.add_gridspec(nrows=8, ncols=ncols, height_ratios=[1,1,0.01,1,1,0.01,1,1])
        fig.subplots_adjust(hspace=0.25)

        axes = np.empty((8, ncols), dtype=object)
        for rr in range(8):
            for cc in range(ncols):
                axes[rr, cc] = None if rr in (2,5) else fig.add_subplot(gs[rr, cc])

        cols = []
        col_labels = []

        for cval in fast_values:
            base_rec = dict(rlist[0]); base_rec[fast_param] = cval
            col_labels.append(f"{short_param_tag(fast_param)}={format_value_for_title(fast_param, cval)}")

            # anneal sweep
            anneal_curves = []
            for aC in sorted(anneal_list):
                rec = dict(base_rec)
                rec["temp_schedule"] = build_schedule(default_firing, aC, fire_s=FIRE_S, anneal_s=anneal_sweep_time)
                stem = make_stem_compact(rec, keys_in_play)
                res, _ = run_or_load_sim(rec=rec)
                t, y, _ = extract_stage_curves(res, stage_index=STAGE_ANN, ut_index=UT_INDEX, scale=SCALE_Y)
                if t.size and y.size: anneal_curves.append((aC, t, y))

            # firing @ default anneal
            firing_curves = []
            for fC in sorted(firing_list):
                rec = dict(base_rec)
                rec["temp_schedule"] = build_schedule(fC, default_anneal, fire_s=FIRE_S, anneal_s=default_anneal_time)
                stem = make_stem_compact(rec, keys_in_play)
                res, _ = run_or_load_sim(rec=rec)
                t, y, _ = extract_stage_curves(res, stage_index=STAGE_ANN, ut_index=UT_INDEX, scale=SCALE_Y)
                if t.size and y.size: firing_curves.append((fC, t, y))

            # firing @ alt anneal
            firing_curves_alt = []
            for fC in sorted(firing_list):
                rec = dict(base_rec)
                rec["temp_schedule"] = build_schedule(fC, alt_anneal_after_firing, fire_s=FIRE_S, anneal_s=alt_anneal_after_firing_time)
                stem = make_stem_compact(rec, keys_in_play)
                res, _ = run_or_load_sim(rec=rec)
                t, y, _ = extract_stage_curves(res, stage_index=STAGE_ANN, ut_index=UT_INDEX, scale=SCALE_Y)
                if t.size and y.size: firing_curves_alt.append((fC, t, y))

            cols.append((anneal_curves, firing_curves, firing_curves_alt))

        # draw (hours for anneal & alt; minutes for default firing)
        for j, (label, (anneal_curves, firing_curves, firing_curves_alt)) in enumerate(zip(col_labels, cols)):
            # anneal rows 0–1
            xunit, xscale = "hours", 3600.0
            for row in (0,1):
                ax = axes[row, j]
                draw_curves(ax, anneal_curves, xscale=xscale, ylog=(row==0), color_map=anneal_colors)
                if j == 0: ax.set_ylabel("[H] in SiOx (LOG)" if row==0 else "[H] in SiOx (LINEAR)")
                ax.set_title(label, fontsize=14)
                ax.set_xlabel(f"Time ({xunit})")
            # firing (default) rows 3–4 (minutes)
            xunit, xscale = "minutes", 60.0
            for row in (3,4):
                ax = axes[row, j]
                draw_curves(ax, firing_curves, xscale=xscale, ylog=(row==3), color_map=firing_colors)
                if j == 0: ax.set_ylabel("[H] in SiOx (LOG)" if row==3 else "[H] in SiOx (LINEAR)")
                ax.set_xlabel(f"Time ({xunit})")
            # firing (alt) rows 6–7 (hours)
            xunit, xscale = "hours", 3600.0
            for row in (6,7):
                ax = axes[row, j]
                draw_curves(ax, firing_curves_alt, xscale=xscale, ylog=(row==6), color_map=firing_colors)
                if j == 0: ax.set_ylabel("[H] in SiOx (LOG)" if row==6 else "[H] in SiOx (LINEAR)")
                ax.set_xlabel(f"Time ({xunit})")

        if ncols > 0:
            mid = max(0, ncols//2)
            axes[0, mid].legend(handles=legend_handles(anneal_colors, "{}°C anneal"), loc="best", frameon=False, fontsize=9)
            axes[3, mid].legend(handles=legend_handles(firing_colors, "{}°C fire"),   loc="best", frameon=False, fontsize=9)
            axes[6, mid].legend(handles=legend_handles(firing_colors, "{}°C fire"),   loc="best", frameon=False, fontsize=9)

        slow_label = ", ".join(f"{short_param_tag(k)}={format_value_for_title(k, v)}" for k, v in slow_key if v is not None) or "default"
        fig.text(0.5, 0.29, "Firing sweep followed by 250°C annealing", fontsize=20, fontweight="bold", ha="center")
        fig.text(0.5, 0.63, "Firing sweep followed by 350°C annealing", fontsize=20, fontweight="bold", ha="center")
        fig.suptitle(f"Fixed: {slow_label}", fontsize=15)
        fig.tight_layout()

        out_path = out_dir/"plots"/f"resweep_fast__{short_param_tag(fast_param)}__{slow_label}.png"
        fig.savefig(out_path, dpi=dpi); plt.close(fig)
        print(f"✔ saved: {out_path}")


def sweep_from_spec(
    sweep_spec: dict,
    fast_param: str,
    fast_values: list | None = None,
    *,
    firing_list=(650,700,750),
    anneal_list=(250,300,350),
    default_anneal=350,
    default_anneal_time=30_000,
    alt_anneal_after_firing=250,
    alt_anneal_after_firing_time=1_000_000,
    default_firing=677,
    anneal_sweep_time=1_000_000,
    out_dir: Path | None = None,
    dpi: int = DPI,
    show: bool = True,
    save: bool = False,
):
    """
    Create the same resweep figure as `resweep_from_rows`, but take a parameter
    specification dictionary instead of reading CSV rows.

    Parameters
    - sweep_spec: mapping param_name -> list(values). Must include the fast_param key
      or you can pass fast_values explicitly.
    - fast_param: which parameter varies across columns
    - fast_values: optional explicit list; otherwise derived from sweep_spec[fast_param]
    - firing_list / anneal_list: lists of temps to sweep
    - *_time: durations (seconds) for the various stages
    - default_firing: firing temp used for the anneal sweep
    - show: if True, call plt.show()
    - save: if True, save to out_dir (if provided)

    Returns (fig, axes) or list of (fig, axes) if multiple slow-combos
    """
    # Normalize inputs
    fast_values = list(fast_values) if fast_values is not None else list(sweep_spec.get(fast_param, []))
    if not fast_values:
        raise ValueError("fast_values must be provided either as argument or present in sweep_spec[fast_param]")

    # Build a list of slow-key combinations: treat any other listed keys in sweep_spec
    slow_keys = [k for k in sweep_spec.keys() if k != fast_param]
    # If there are no slow keys, create a single empty combo
    slow_combos = [dict(zip(slow_keys, vals)) for vals in product(*[sweep_spec[k] for k in slow_keys])] if slow_keys else [{}]

    # Reuse the same plotting layout as resweep_from_rows (8 rows: two-row blocks ×3)
    anneal_colors = map_colors_low_to_high(list(anneal_list))
    firing_colors = map_colors_low_to_high(list(firing_list))

    figs = []
    for slow_combo in slow_combos:
        ncols = max(1, len(fast_values))
        fig = plt.figure(figsize=(5 * ncols, 18))
        gs = fig.add_gridspec(nrows=8, ncols=ncols, height_ratios=[1,1,0.01,1,1,0.01,1,1])
        fig.subplots_adjust(hspace=0.25)

        axes = np.empty((8, ncols), dtype=object)
        for rr in range(8):
            for cc in range(ncols):
                axes[rr, cc] = None if rr in (2,5) else fig.add_subplot(gs[rr, cc])

        cols = []
        col_labels = []

        keys_in_play = [k for k in slow_keys + [fast_param]]

        for cval in fast_values:
            base_rec = dict(slow_combo); base_rec[fast_param] = cval
            col_labels.append(f"{short_param_tag(fast_param)}={format_value_for_title(fast_param, cval)}")

            # anneal sweep
            anneal_curves = []
            for aC in sorted(anneal_list):
                rec = dict(base_rec)
                rec["temp_schedule"] = build_schedule(default_firing, aC, fire_s=FIRE_S, anneal_s=anneal_sweep_time)
                stem = make_stem_compact(rec, keys_in_play)
                res, _ = run_or_load_sim(rec=rec)
                t, y, _ = extract_stage_curves(res, stage_index=STAGE_ANN, ut_index=UT_INDEX, scale=SCALE_Y)
                if t.size and y.size: anneal_curves.append((aC, t, y))

            # firing @ default anneal
            firing_curves = []
            for fC in sorted(firing_list):
                rec = dict(base_rec)
                rec["temp_schedule"] = build_schedule(fC, default_anneal, fire_s=FIRE_S, anneal_s=default_anneal_time)
                stem = make_stem_compact(rec, keys_in_play)
                res, _ = run_or_load_sim(rec=rec)
                t, y, _ = extract_stage_curves(res, stage_index=STAGE_ANN, ut_index=UT_INDEX, scale=SCALE_Y)
                if t.size and y.size: firing_curves.append((fC, t, y))

            # firing @ alt anneal
            firing_curves_alt = []
            for fC in sorted(firing_list):
                rec = dict(base_rec)
                rec["temp_schedule"] = build_schedule(fC, alt_anneal_after_firing, fire_s=FIRE_S, anneal_s=alt_anneal_after_firing_time)
                stem = make_stem_compact(rec, keys_in_play)
                res, _ = run_or_load_sim(rec=rec)
                t, y, _ = extract_stage_curves(res, stage_index=STAGE_ANN, ut_index=UT_INDEX, scale=SCALE_Y)
                if t.size and y.size: firing_curves_alt.append((fC, t, y))

            cols.append((anneal_curves, firing_curves, firing_curves_alt))

        # draw (hours for anneal & alt; minutes for default firing)
        for j, (label, (anneal_curves, firing_curves, firing_curves_alt)) in enumerate(zip(col_labels, cols)):
            # anneal rows 0–1
            xunit, xscale = "hours", 3600.0
            for row in (0,1):
                ax = axes[row, j]
                draw_curves(ax, anneal_curves, xscale=xscale, ylog=(row==0), color_map=anneal_colors)
                if j == 0: ax.set_ylabel("[H] in SiOx (LOG)" if row==0 else "[H] in SiOx (LINEAR)")
                ax.set_title(label, fontsize=14)
                ax.set_xlabel(f"Time ({xunit})")
            # firing (default) rows 3–4 (minutes)
            xunit, xscale = "minutes", 60.0
            for row in (3,4):
                ax = axes[row, j]
                draw_curves(ax, firing_curves, xscale=xscale, ylog=(row==3), color_map=firing_colors)
                if j == 0: ax.set_ylabel("[H] in SiOx (LOG)" if row==3 else "[H] in SiOx (LINEAR)")
                ax.set_xlabel(f"Time ({xunit})")
            # firing (alt) rows 6–7 (hours)
            xunit, xscale = "hours", 3600.0
            for row in (6,7):
                ax = axes[row, j]
                draw_curves(ax, firing_curves_alt, xscale=xscale, ylog=(row==6), color_map=firing_colors)
                if j == 0: ax.set_ylabel("[H] in SiOx (LOG)" if row==6 else "[H] in SiOx (LINEAR)")
                ax.set_xlabel(f"Time ({xunit})")

        if ncols > 0:
            mid = max(0, ncols//2)
            axes[0, mid].legend(handles=legend_handles(anneal_colors, "{}°C anneal"), loc="best", frameon=False, fontsize=9)
            axes[3, mid].legend(handles=legend_handles(firing_colors, "{}°C fire"),   loc="best", frameon=False, fontsize=9)
            axes[6, mid].legend(handles=legend_handles(firing_colors, "{}°C fire"),   loc="best", frameon=False, fontsize=9)

        slow_label = ", ".join(f"{short_param_tag(k)}={format_value_for_title(k, v)}" for k, v in slow_combo.items() if v is not None) or "default"
        fig.text(0.5, 0.29, "Firing sweep followed by 250°C annealing", fontsize=20, fontweight="bold", ha="center")
        fig.text(0.5, 0.63, "Firing sweep followed by 350°C annealing", fontsize=20, fontweight="bold", ha="center")
        fig.suptitle(f"Fixed: {slow_label}", fontsize=15)
        fig.tight_layout()

        if save and out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir/f"resweep_fast__{short_param_tag(fast_param)}__{slow_label}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi)
            print(f"✔ saved: {out_path}")

        if show:
            try: plt.show()
            except Exception: pass
        else:
            plt.close(fig)

        figs.append((fig, axes))

    # If only one figure generated, return that pair, else return list
    return figs[0] if len(figs) == 1 else figs

def main():
    # ap = argparse.ArgumentParser(description="Resweep from CSV (thin driver)")
    # ap.add_argument("--from-csv", required=True)
    # ap.add_argument("--fast-param", default="B_detrap")
    # ap.add_argument("--fast-values", default="0.7,0.95,1.1,1.4,1.7")
    # ap.add_argument("--firing-list", default="650,700,750")
    # ap.add_argument("--anneal-list", default="250,300,350")
    # ap.add_argument("--default-anneal", type=int, default=350)
    # ap.add_argument("--default-anneal-time", type=int, default=30_000)
    # ap.add_argument("--alt-anneal-after-firing", type=int, default=250)
    # ap.add_argument("--alt-anneal-after-firing-time", type=int, default=1_000_000)
    # ap.add_argument("--default-firing", type=int, default=677)
    # ap.add_argument("--anneal-sweep-time", type=int, default=1_000_000)
    # ap.add_argument("--out-dir", default="./resweep_from_csv/")
    # ap.add_argument("--dpi", type=int, default=DPI)
    # args = ap.parse_args()





    # === FAST PARAM: B_detrap in 0.1 eV steps (0.8 → 1.3) ===
    FAST_VALUES = [round(x, 1) for x in np.arange(0.8, 1.3 + 0.0001, 0.1)]

    # === Sweep spec ===
    # Singletons = fixed; lists = swept.
    # NOTE: per your instruction, *detrap attempt freqs* are FIXED (not swept).
    sweep_spec = {
        # ----------------- AlOx (A) -----------------
        "A_trap":                [1.1],          # small band around 1.1 eV
        "A_detrap":              [2.3,2.5],                      # fixed center value
        "A_trap_attemptfreq":    [1e13],                # small sweep
        "A_detrap_attemptfreq":  [1e12],                      # FIXED (do not sweep)

        # ----------------- poly-Si (B) -----------------
        "B_trap":                [0.50],          # small band around 0.6 eV
        "B_detrap":              FAST_VALUES,                 # FAST PARAM: 0.8–1.3 eV in 0.1 steps
        "B_trap_attemptfreq":    [3e12,1e13],                # small sweep
        "B_detrap_attemptfreq":  [1e12],                      # FIXED (do not sweep)

        # ----------------- SiOx (C) -----------------
        "C_trap":                [0.50],          # small band around 0.5 eV
        "C_detrap":              [1.50, 1.55, 1.60],          # small band around 1.55 eV
        "C_trap_attemptfreq":    [1e12,1e13],                # small sweep
        "C_detrap_attemptfreq":  [1e12],                      # FIXED (do not sweep)

        # ----------------- c-Si (D) -----------------
        "D_trap":                [0.50],          # small band around 0.48 eV
        "D_detrap":              [0.90],          # small band (you can fix to [0.90] if desired)
        "D_trap_attemptfreq":    [2e12,1e13],                # small sweep
        "D_detrap_attemptfreq":  [1e12],                      # FIXED (do not sweep)
    }

    # Use your existing plotting/schedule defaults from sweep_from_spec
    sweep_from_spec(
        sweep_spec=sweep_spec,
        fast_param="B_detrap",
        fast_values=FAST_VALUES,
        # Optional: tweak these (left as your function defaults):
        firing_list=(650, 700, 750),
        anneal_list=(250, 300, 350),
        default_anneal=350,
        default_anneal_time=30_000,
        alt_anneal_after_firing=250,
        alt_anneal_after_firing_time=1_000_000,
        default_firing=650,
        anneal_sweep_time=1_000_000,
        # Output
        out_dir=Path("./sweep_Bdetrap_small/"),
        dpi=200,
        show=False,
        save=True,
    )
    
if __name__ == "__main__":
    main()
