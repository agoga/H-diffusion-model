# vizkit.py
# Authors: Adam Goga 
from __future__ import annotations
import numpy as np
from typing import Any, Iterable, Optional  # non-container typing only
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import csv
from pathlib import Path
import helpers as h
from simulation_manager import SimulationResult
from simulation_manager import SimulationManager
import nrel_exp as exp 
from scipy.interpolate import make_interp_spline


from matplotlib.ticker import ScalarFormatter
from matplotlib.axes import Axes
_LAYER_LABELS = {"A": "Al2O3", "B": "polySi", "C": "SiOx", "D": "cSi"}
from fourstates import SAMPLE_LENGTH, SCALE_A, SCALE_B, SCALE_C, SCALE_D, SCALE_E


# ------------------------------ Global Matplotlib Style ------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# ------------------------------ Color Utilities --------------------------------------
def distinct_colors(n: int) -> list[str]:
    """
    Return n bright, distinct colors (repeats base palette deterministically if needed).
    """
    base = ["#1f77b4","#2ca02c","#17becf","#9467bd","#ff7f0e",
            "#8c564b","#e377c2","#7f7f7f","#bcbd22","#d62728"]
    if n <= len(base):
        return base[:n]
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]

def bright_distinct_colors(n: int) -> list[str]:
    """
    Return n bright, high-contrast, maximally distinct colors.

    Order chosen for maximum perceptual separation, unlike the muted matplotlib defaults.
    """
    base= [  "#1f77b4",  # blue
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    ]
    # base = [
    #     "#FF0000",  # bright red
    #     "#0000FF",  # bright blue
    #     "#00CC00",  # bright green
    #     "#FF00FF",  # magenta
    #     "#FF8800",  # orange
    #     "#00FFFF",  # cyan
    #     "#000000",  # black
    #     "#FFFF00",  # yellow (only if repeating)
    # ]

    if n <= len(base):
        return base[:n]

    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]

def _normalize_temp_key(x: float | int) -> int:
    """
    Normalize a temperature value (°C) to an integer °C key used everywhere
    for color-map indexing. This eliminates float/int mismatches.
    """
    return int(round(float(x)))

def map_colors_low_to_high(vals: Iterable[float | int]) -> dict[int, str]:
    """
    Map a set of temperature-like values (°C) to colors, sorted low→high.
    - Keys in the returned dict are **integer °C** (rounded).
    - Hottest value is forced to red for immediate visual emphasis.
    """
    # Normalize to int °C and de-duplicate while preserving order
    ints = sorted({_normalize_temp_key(v) for v in vals})
    palette = distinct_colors(len(ints)) if ints else []
    if ints:
        palette[-1] = "#FF0000"  # hottest = red
    return {t: palette[i] for i, t in enumerate(ints)}

def _color_for_key(color_map: dict[int, str], key: int) -> str:
    """
    Safe lookup: if key not in color_map (e.g., due to rounding/novel temps),
    fall back to the nearest existing key's color, or a neutral gray if empty.
    This prevents KeyError crashes while keeping colors semantically close.
    """
    if key in color_map:
        return color_map[key]
    if color_map:
        nearest = min(color_map.keys(), key=lambda k: abs(k - key))
        return color_map[nearest]
    return "#333333"

# ------------------------------ Fast, Simple Thinner ----------------------------------
def _thin_xy(t: np.ndarray, y: np.ndarray, *, max_pts: int = 2000, keep_peak: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (t_thin, y_thin) with at most max_pts samples.
    - O(N) downsample by stride; always keeps the global peak (argmax y) if requested.
    - Keeps order; stable and allocation-light.
    """
    n = t.size
    if n <= max_pts or max_pts <= 0:
        return t, y

    step = int(np.ceil(n / max_pts))
    idx = np.arange(0, n, step, dtype=int)

    if keep_peak and n:
        ip = int(np.argmax(y))
        if ip not in idx:
            idx = np.sort(np.concatenate([idx, np.array([ip], dtype=int)]))

    return t[idx], y[idx]

# ------------------------------ Drawing Primitives ------------------------------------
def draw_curves(
    ax,
    curves: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    xscale: float,
    ylog: bool,
    color_map: dict[int, str],
    label: Optional[str] = None,
    scatter: bool = False,
    max_pts: int = 50_000,
    draw_max_x = True,
    offset=0,
    ls='-'
):
    """
    Draw multiple curves on `ax`. Each curve tuple is:
        (key:int, t:np.ndarray[s], y:np.ndarray[scaled concentration])

    Conventions & Tricky Bits:
    - `key` is an **integer °C** tag (e.g., the "other stage" temperature).
      This must match the keys used in `color_map` (also integer °C).
    - We thin BEFORE unit scaling to avoid biasing the sampler.
    - Safe color lookup: if a key is missing, we fall back to nearest color.
    """
    for key, t, y in curves:
        # Thin BEFORE scaling
        t_plot, y_plot = _thin_xy(np.asarray(t, float), np.asarray(y, float),
                                  max_pts=max_pts, keep_peak=True)
        y_plot+=offset
        tx = t_plot / xscale
        c = _color_for_key(color_map, key)

        if scatter:
            ax.scatter(tx, y_plot, color=c)
        else:
            ax.plot(tx, y_plot, lw=1.6, color=c, linestyle=ls, label=label)

        # Mark global peak of the plotted series (after thinning)
        if y_plot.size and draw_max_x:
            pk = int(np.argmax(y_plot))
            ax.scatter([tx[pk]], [y_plot[pk]], s=50, marker="x", linewidths=2, color=c)

    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    if ylog:
        ax.set_yscale("log")


def legend_handles(color_map: dict[int, str], label_fmt: str) -> list[Line2D]:
    """
    Build legend proxy lines using the provided color map.
    """
    return [Line2D([0],[0], lw=2, color=color_map[v], label=label_fmt.format(v))
            for v in sorted(color_map)]

def _tpeak_series(curves: list[tuple[int, np.ndarray, np.ndarray, float]], scale: float = 1.0) -> list[tuple[int, float]]:
    """
    From [(fC, t, y, seg_len)] compute sorted list of (fC:int°C, t_peak/scale).
    Returns [] if any curve has no samples.
    """
    out: list[tuple[int, float]] = []
    for fC, t, y, _seg in curves:
        if t.size == 0 or y.size == 0:
            return []
        imax = int(np.argmax(y))
        out.append((_normalize_temp_key(fC), float(t[imax]) / scale))
    out.sort(key=lambda kv: kv[0])
    return out

def _title_from_params(params: dict[str, Any], keys: list[str]) -> str:
    """
    Build a human-readable title from params using helpers:
      - short tags (AH, BH, kHA, …) via h.short_param_tag
      - nice numeric formatting via h.format_value_for_title
      - comma-separated, no underscores
    Only includes keys present in params.
    """
    parts: list[str] = []
    for k in keys:
        if k in params:
            tag = h.short_param_tag(k)
            val = h.format_value_for_title(k, params[k])
            parts.append(f"{tag}={val}")
    return ", ".join(parts) if parts else "Parameters"

# ------------------------------ Axis Helpers ------------------------------------------
def _to_hours(t: np.ndarray) -> np.ndarray:
    return np.asarray(t, float) / 3600.0

def _other_stage(names: list[str], name: str) -> Optional[str]:
    """
    Given stage names like ["firing","annealing"], return the other one.
    Returns None if not found or ambiguous.
    """
    if name not in names or len(names) < 2:
        return None
    return names[1 - names.index(name)]

# ------------------------------ Plot Routines -----------------------------------------
def plot_siox_C_all_stages(
    res_list,
    *,
    labels=None,                  # NEW: list[str] or None
    kind: str = "total",           # "total" | "trapped" | "mobile"
    ylog: bool = False,
    xlog: bool = True,
    ax=None,
    legend: bool = True,
    max_pts: int = 50_000,
):
    """
    Plot [H] in layer C (SiOx) for every stage for each result in `res_list`,
    all on the same axes.

    Parameters
    ----------
    res_list:
        Iterable of SimulationResult or raw dicts returned by fs.simulate/run_or_load_sim.
    labels:
        Optional list of legend prefixes for each result. If None, uses "result 1", "result 2", ...

    Notes
    -----
    - Uses SimulationResult stage slicing (res.series/res.total). No duplication of stage math.
    - Each stage uses its own re-zeroed time base (t=0 at stage start).
    - Colors are keyed by stage temperature (°C, rounded to int) when available.
    - Line style distinguishes results; color distinguishes stages.
    """
    # Normalize to a list
    if res_list is None:
        raise ValueError("res_list must be a non-empty list/iterable of results")
    if not isinstance(res_list, (list, tuple)):
        res_list = [res_list]

    nres = len(res_list)
    if nres == 0:
        raise ValueError("res_list must be non-empty")

    # Default labels
    if labels is None:
        labels = [f"result {i+1}" for i in range(nres)]
    else:
        labels = list(labels)
        if len(labels) != nres:
            raise ValueError(f"labels must have same length as res_list ({nres}), got {len(labels)}")

    # Convert any raw dict -> SimulationResult
    norm_results = []
    for r in res_list:
        if isinstance(r, dict):
            r = SimulationResult.from_solver_dict(r)
        norm_results.append(r)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # Distinguish different results by line style (keeps stage colors meaningful)
    linestyles = ["-", "--", ":", "-."]

    # Plot each result
    for j, (res, prefix) in enumerate(zip(norm_results, labels)):
        nstage = int(res.num_stages())
        if nstage <= 0:
            continue

        # Stage display names
        names = res.stage_names()
        if not names:
            names = [f"stage{i}" for i in range(nstage)]

        # Color map keyed by stage temperature (preferred) else by stage index
        keys = []
        for i in range(nstage):
            try:
                keys.append(_normalize_temp_key(res.temperature_for_stage(i)))
            except Exception:
                keys.append(int(i))
        color_map = map_colors_low_to_high(keys)

        ls = linestyles[j % len(linestyles)]

        for i in range(nstage):
            if kind == "total":
                t, y = res.total("C", stage=i)
            elif kind in ("trapped", "mobile"):
                t, y = res.series(layer="C", kind=kind, stage=i)
            else:
                raise ValueError("kind must be 'total', 'trapped', or 'mobile'")

            t = np.asarray(t, float)
            y = np.asarray(y, float)
            if t.size == 0:
                continue

            t_plot, y_plot = _thin_xy(t, y, max_pts=max_pts, keep_peak=True)

            if ylog:
                y_plot = np.where(y_plot > 0, y_plot, np.nan)

            try:
                key = _normalize_temp_key(res.temperature_for_stage(i))
                stage_lbl = f"{names[i]} ({key}°C)"
            except Exception:
                key = int(i)
                stage_lbl = f"{names[i]}"

            c = _color_for_key(color_map, key)
            lbl = f"{prefix} | {stage_lbl}"

            ax.plot(t_plot, y_plot, lw=1.6, ls=ls, color=c, label=lbl)

    ax.set_xlabel("Time in stage (s)")
    if kind == "total":
        ax.set_ylabel("[H]$_{C}$ total (cm$^{-3}$)")
    else:
        ax.set_ylabel(f"[H]$_{{C}}$ {kind} (cm$^{{-3}}$)")

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.3)

    if legend:
        ax.legend()

    fig.tight_layout()
    return ax


def plot_layer_other_stagetemp(
    mgr: SimulationManager,
    *,
    stage_filter: str,          # "firing" | "annealing" (temperature to match)
    target_T: float,            # °C to match in stage_filter
    layer: str = "C",           # "A".."E"
    kind: str = "trapped",      # "trapped" | "mobile"
    stage_to_plot: int|str = "annealing",  # slice/plot this stage (independent of filter)
    ylog: bool = False,
    x_lim_left: Optional[float] = None,
    ax=None,                    # optional matplotlib Axes
    legend=False,
    cycle_len = 2
):
    """
    Filter by stage_filter≈target_T(°C), then plot (layer, kind) during `stage_to_plot`
    for each matched simulation. Colors are keyed by the *other* stage's temperature
    (rounded integer °C), so the legend communicates the sweep axis.

    Tricky Bits fixed here:
    - All color keys are normalized to **int °C** via _normalize_temp_key to avoid
      float/int mismatches that caused KeyError in draw_curves.
    - If the "other stage" doesn't exist (e.g. 1-stage schedule), we degrade gracefully
      to a single color and no legend.
    """
    matches = mgr.results_with_stage_temperature(stage=stage_filter, target_T=target_T)
    if not matches:
        print(f"No simulations found with {stage_filter} ≈ {target_T:.1f} °C.")
        return None

    # Determine which stage labels the colors/legend
    names0 = matches[0].stage_names()
    label_stage = _other_stage(names0, stage_filter)

    # Build the color map keyed by int °C of the label stage
    temps: list[int] = []
    if label_stage:
        for r in matches:
            try:
                temps.append(_normalize_temp_key(r.temperature_for_stage(label_stage)))
            except KeyError:
                continue
    temp_color_map = map_colors_low_to_high(temps)

    # Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # X axis scaling & titles
    if isinstance(stage_to_plot,str):
        if stage_to_plot == "annealing":
            stage_to_plot=1
        else:
            stage_to_plot=0
        
        
        
    if stage_to_plot % cycle_len == 1:
        scale = 3600.0
        xlab = "Time (hours)"
        title = f"[H]_{layer} — {stage_filter}@{target_T:.0f}°C"
    else:
        scale = 1.0
        xlab = "Time (seconds)"
        title = f"[H]_{layer} during firing stage"

    # Plot each matched run
    for res in matches:
        try:
            t_s, y = res.series(layer=layer, kind=kind, stage=stage_to_plot)
        except KeyError:
            continue

        t_s = np.asarray(t_s, float)
        y = np.asarray(y, float)
        y = np.where(y > 0, y, np.nan)  # avoid log(<=0) issues

        # Choose label and color key consistently (int °C)
        if label_stage:
            try:
                k_int = _normalize_temp_key(res.temperature_for_stage(label_stage))
                label = f"{k_int:.0f}°C"
            except KeyError:
                k_int = _normalize_temp_key(target_T)
                label = res.schedule_spec
        else:
            k_int = _normalize_temp_key(target_T)
            label = res.schedule_spec

        # Draw
        draw_curves(
            ax,
            [(k_int, t_s, y)],
            xscale=scale,
            ylog=ylog,
            color_map=temp_color_map,
            label=None,
            offset=1e18
        )


    ax.set_xlabel(xlab)
    ax.set_ylabel(f"{kind.capitalize()} [H]_{layer} (cm⁻³)")
    ax.set_xscale("log")
    if x_lim_left is not None:
        ax.set_xlim(left=x_lim_left)
    ax.set_title(title)
    if label_stage and temp_color_map and legend:
        # Sorted by numeric °C ascending via legend_handles()
        handles = legend_handles(temp_color_map, "{}°C")
        ax.legend(handles=handles, title=f"{label_stage} temp (°C)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return (t_s,y)


# paper printing version
def plot_layer_other_stagetemp_paper(
    mgr: SimulationManager,
    *,
    stage_filter: str,          # "firing" | "annealing" (temperature to match)
    target_T: float,            # °C to match in stage_filter
    unfired_target_T: float = None,
    layer: str = "C",           # "A".."E"
    kind: str = "trapped",      # "trapped" | "mobile"
    stage_to_plot: str = "annealing",  # slice/plot this stage (independent of filter)
    xlabel=None,
    xscale=1,
    ylog: bool = False,
    x_lims = None,
    legend=False,
    draw_max_x=True,
    detection_offset=0,
    ax=None,
    csv_name=None,
    test=False
):
    """
    Filter by stage_filter≈target_T(°C), then plot (layer, kind) during `stage_to_plot`
    for each matched simulation. Colors are keyed by the *other* stage's temperature.

    If unfired_target_T is provided:
      → also overlays the unfired simulation(s) as black dotted lines with legend 'unfired'.
    """

    curves_for_export = {} 
    matches = mgr.results_with_stage_temperature(stage=stage_filter, target_T=target_T)
    if not matches:
        print(f"No simulations found with {stage_filter} ≈ {target_T:.1f} °C.")
        return None
    
    # --- unfired comparison simulations ---
    unfired_matches = []
    if unfired_target_T is not None:
        unfired_matches = mgr.results_with_stage_temperature(
            stage=stage_filter, target_T=unfired_target_T
        )

    # --- pick math-safe layer names for subscripts ---
    if layer == "A":
        layer_plain = "Al2O3"
        layer_math  = r"\mathrm{Al_{2}O_{3}}"
    elif layer == "B":
        layer_plain = "polySi"
        layer_math  = r"\mathrm{poly\text{-}Si}"
    elif layer == "C":
        layer_plain = "SiOx"
        layer_math  = r"\mathrm{SiO_{x}}"
    elif layer == "D":
        layer_plain = "cSi"
        layer_math  = r"\mathrm{c\text{-}Si}"
    else:
        layer_plain = layer
        layer_math  = rf"\mathrm{{{layer}}}"

    names0 = matches[0].stage_names()
    label_stage = _other_stage(names0, stage_filter)

    temps = []
    if label_stage:
        for r in matches:
            try:
                temps.append(_normalize_temp_key(r.temperature_for_stage(label_stage)))
            except KeyError:
                continue

    temp_color_map = map_colors_low_to_high(temps)

    # Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if xlabel is None:
        xlabel = "Time (seconds)"

    # ------------- PLOT MAIN MATCHED RUNS (colored) ----------------
    for res in matches:
        try:
            t_s, y = res.series(layer=layer, kind=kind, stage=stage_to_plot)
        except KeyError:
            continue

        t_s = np.asarray(t_s, float)
        y = np.where(np.asarray(y, float) > 0, y, np.nan)

        if label_stage:
            try:
                k_int = _normalize_temp_key(res.temperature_for_stage(label_stage))
                label = f"{k_int:.0f}°C"
            except KeyError:
                k_int = _normalize_temp_key(target_T)
                label = res.schedule_spec
        else:
            k_int = _normalize_temp_key(target_T)
            label = res.schedule_spec

        label_str = f"{k_int}C" if label_stage else f"{target_T}C"
        curves_for_export[label_str] = (t_s.copy(), y.copy())
        if test:
            print(y[0:500])
        draw_curves(
            ax,
            [(k_int, t_s, y)],
            xscale=xscale,
            ylog=ylog,
            color_map=temp_color_map,
            label=None,
            offset=detection_offset,
            draw_max_x=draw_max_x
        )

    # ------------- PLOT UNFIRED OVERLAY (black, dotted) ----------------
    if unfired_matches:
        for res in unfired_matches:
            try:
                t_s, y = res.series(layer=layer, kind=kind, stage=stage_to_plot)
            except KeyError:
                continue

            t_s = np.asarray(t_s, float)
            y = np.where(np.asarray(y, float) > 0, y, np.nan)

            curves_for_export["unfired"] = (t_s.copy(), y.copy())
            
            draw_curves(
            ax,
            [(0, t_s, y)],
            xscale=xscale,
            ylog=ylog,
            color_map={0:'k'},
            label='unfired',
            offset=detection_offset,
            draw_max_x=False,
            ls='--'
        )

    # -----------------------------------------------------------------

    ax.set_xlabel(xlabel)
    ax.set_ylabel(rf"$\Delta [H]_{{{layer_math}}}$ (cm$^{{-3}}$)")
    ax.set_xscale("log")

    if x_lims is not None:
        if x_lims[0] is not None:
            if x_lims[1] is not None:
                ax.set_xlim(left=x_lims[0], right=x_lims[1])
            else:
                ax.set_xlim(left=x_lims[0])

    # --- legend ---
    if legend:
        handles = []

        # Colored temperature legend
        if label_stage and temp_color_map:
            handles.extend(legend_handles(temp_color_map, "{}°C"))

        # Unfired legend entry
        if unfired_matches:
            import matplotlib.lines as mlines
            handles.append(
                mlines.Line2D(
                    [], [], color="black", linestyle=":", linewidth=2, label="unfired"
                )
            )

        if handles:
            ax.legend(handles=handles, title=f"{label_stage.capitalize()} temperature" if label_stage else None)
    
    default_path='./origin_exports/'
    if csv_name is None:
        csv_name=default_path+f"{layer}_{kind}_{stage_filter}_{target_T:.0f}C.csv"
    else:
        csv_name=default_path+csv_name+".csv"
    save_curves_for_origin(
            curves_for_export,
            filepath=csv_name,
            max_pts=2000000
            )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return ax


def plot_peaktime_vs_stage(
    mgr: SimulationManager,
    *,
    target_T: float,                 # °C in stage_for_peak to include
    layer: str = "C",
    kind: str = "trapped",
    stage_for_peak: str = "annealing",
    stage_sort: str = "firing",      # x-axis = firing temperature
    ylog: bool = True,
    ylims: list[float] = [1, 100],
    exp_data=None,
    ax=None,
):
    """
    Scatter of peak time (hours) vs `stage_sort` temperature (°C) for runs with
    `stage_for_peak == target_T` (± tolerance). Pure viz helper around
    SimulationManager.peak_times_for_stageT.
    """
    data = mgr.peak_times_for_stageT(
        target_T=target_T,
        stage_for_peak=stage_for_peak,
        stage_sort=stage_sort,
        layer=layer,
        kind=kind,
    )
    if not data:
        print("No peak data found.")
        return None
    curves_for_export = {}
    
    temps, t_peaks_h = zip(*[(int(round(T)), t/3600.0) for (T, t) in data])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
        
    curves_for_export["T_vs_peak"] = (
            np.asarray(temps, float),
            np.asarray(t_peaks_h, float),
        )

    ax.scatter(temps, t_peaks_h, s=50,c='r',marker='x',label="simulation")
    print(exp_data)
    
    if exp_data is not None:
        Texp=exp_data[0]
        pexp=exp_data[1]
        print(f"{Texp} - {pexp}")
        ax.scatter(Texp,pexp,s=50, c='b',label='measured')
    # ax.plot(temps,t_peaks_h)
    ax.set_xlabel(f"{stage_sort.capitalize()} temperature (°C)")
    ax.set_ylabel("Peak time (hours)")
    if ylog:
        ax.set_yscale("log")
        
    if exp_data is not None:
        plt.legend()
        
        
    save_curves_for_origin(
            curves_for_export,
            filepath=f"./origin_exports/FireT_vs_peak.csv",
            max_pts=10000
        )
    ax.grid(True, alpha=0.3)
    # ax.set_title(f"Peak time vs {stage_sort} temp, for {stage_for_peak}={_normalize_temp_key(target_T):.0f}°C")
    ax.set_ylim(bottom=ylims[0], top=ylims[1])
    fig.tight_layout()
    return ax


def plot_paper(
    mgr: SimulationManager,
    types: list[str] | str,
    temps: list[float | int] | float | int,
    *,
    layer: str = "C",
    kind: str = "trapped",
    stage_to_plot: str = "annealing",
    exp_together=True,
    show_params_in_title: bool = False,
    unfired_T=0
    ):
    # listify inputs
    if isinstance(types, str):
        types = [types]
    if np.isscalar(temps):
        temps = [temps]

    if len(types) != len(temps):
        raise ValueError("'types' and 'temps' must match in length")

    figs = []
    axes = []

    # build suptitle string once (only if used)
    if show_params_in_title:
        p = mgr.params
        title_parts = []
        if all(hasattr(p, x) for x in ("A_detrap","B_detrap","C_detrap","D_detrap")):
            title_parts.append(
                f"detrap[eV] - Al2O3={p.A_detrap:g}, Poly-Si={p.B_detrap:g}, "
                f"SiOx={p.C_detrap:g}, cSi={p.D_detrap:g}"
            )
        if hasattr(p, "A_trap"):
            title_parts.append(f"trap[eV] - Al2O3={p.A_trap:g}")
        if all(hasattr(p, x) for x in ("A_trap_attemptfreq","B_trap_attemptfreq",
                                       "C_trap_attemptfreq","D_trap_attemptfreq")):
            title_parts.append(
                f"f_attempt[s⁻¹] - A={p.A_trap_attemptfreq:g}, "
                f"B={p.B_trap_attemptfreq:g}, C={p.C_trap_attemptfreq:g}, "
                f"D={p.D_trap_attemptfreq:g}"
            )
        suptitle_text = " | ".join(title_parts) if title_parts else None
    else:
        suptitle_text = None


        
        
    for i, (tname, T) in enumerate(zip(types, temps)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        tname = str(tname)
        ylog = tname.endswith("log")

        if tname.startswith("anneal_sweep"):
            plot_layer_other_stagetemp_paper(
                mgr,
                stage_filter="firing",
                target_T=float(T),
                layer=layer,
                kind=kind,
                stage_to_plot=stage_to_plot,
                xscale=3600,
                ylog=False,
                x_lims=[1e-2,None],
                ax=ax,
                xlabel=f"Time (hours)",
                legend=True,
                unfired_target_T=unfired_T,
                detection_offset=0,
                csv_name="anneal_sweep"
            )

        elif tname.startswith("fire_sweep"):
            plot_layer_other_stagetemp_paper(
                mgr,
                stage_filter="annealing",
                target_T=float(T),
                layer=layer,
                kind=kind,
                stage_to_plot=stage_to_plot,
                ylog=ylog,
                x_lims=[1e-2,None],
                xscale=3600,
                ax=ax,
                xlabel=f"Time (hours) during a {T:0.0f}°C anneal",
                legend=False,
                csv_name="fire_sweep"

            )
        
        elif tname.startswith("fire2_sweep"):
            plot_layer_other_stagetemp_paper(
                mgr,
                stage_filter="annealing",
                target_T=float(T),
                layer=layer,
                kind=kind,
                stage_to_plot=stage_to_plot,
                ylog=ylog,
                x_lims=[1e-5,None],
                xscale=3600,
                xlabel=f"Time (hours) during a {T:0.0f}°C anneal",
                ax=ax,
                legend=True,
                csv_name="fire2_sweep"

            )
        

        elif tname.startswith("peak_firingT"):
            min_to_hr=60
            exp_data = [[600, 650, 700, 750, 800],
                            [x/min_to_hr for x in [364, 272, 695, 1031, 5342]],
                        ]
            plot_peaktime_vs_stage(
                mgr,
                target_T=float(T),
                layer=layer,
                kind=kind,
                exp_data=exp_data,
                stage_for_peak="annealing",
                stage_sort="firing",
                ax=ax,
            )

        elif tname.startswith("firing_siox"):
            plot_layer_other_stagetemp_paper(
                mgr,
                stage_filter="annealing",
                target_T=float(T),
                layer=layer,
                kind=kind,
                stage_to_plot="firing",
                xlabel=f"Time (seconds) during firing stage",
                ylog=ylog,
                draw_max_x=False,
                ax=ax,
                csv_name="firing_siox",
            )

        
        fig.tight_layout(rect=[0, 0, 1, 0.93] if suptitle_text else None)
        
        figs.append(fig)
        axes.append(ax)
        plt.show()

    return figs, axes


def plot_full_single_analysis(
    mgr: SimulationManager,
    types: list[str] | str,
    temps: list[float | int] | float | int,
    *,
    layer: str = "C",
    kind: str = "trapped",
    stage_to_plot: str = "annealing",
    x_lim_left: float = 1e-3,
    exp_together=True,
    show_params_in_title: bool = False,   # <--- NEW
    ):
    # listify inputs
    if isinstance(types, str):
        types = [types]
    if np.isscalar(temps):
        temps = [temps]  # type: ignore[assignment]

    if len(types) != len(temps):
        raise ValueError(f"'types' and 'temps' must match in length (got {len(types)} vs {len(temps)})")

    n = len(types)
    if 'exp_overlay' in types and not exp_together:
        n -= 1

    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = np.atleast_1d(axes)

    # -------------- NEW: optional parameter suptitle --------------
    if show_params_in_title:
        p = mgr.params  # SimpleNamespace from fourstates.build_params
        # formatting helper: try to use any helper you already have, else sane fallback
        try:
            import helpers as h
            def fmt(x):  # prefer any helper formatting you’ve defined
                # If you have something like h.format_value_for / h.short_value, use that:
                if hasattr(h, "format_value_for"):
                    return h.format_value_for("param", x)  # type: ignore[attr-defined]
                return f"{x:g}"
        except Exception:
            def fmt(x): return f"{x:g}"

        # pull required fields (detrap in eV; trap in eV; attempt freq in s^-1)
        A_det = getattr(p, "A_detrap", None); B_det = getattr(p, "B_detrap", None)
        C_det = getattr(p, "C_detrap", None); D_det = getattr(p, "D_detrap", None)
        A_trp = getattr(p, "A_trap",   None)

        Af = getattr(p, "A_trap_attemptfreq", None)
        Bf = getattr(p, "B_trap_attemptfreq", None)
        Cf = getattr(p, "C_trap_attemptfreq", None)
        Df = getattr(p, "D_trap_attemptfreq", None)

        # Build a compact, human-readable line. (Units: eV for energies, s⁻¹ for freq)
        # Example: detrap[eV]=A:2.5,B:0.8,C:1.45,D:1.2 | A_trap[eV]=0.90 | f_attempt[s⁻¹]=A:1e13,B:1e12,C:5e12,D:1e12
        parts = []
        if None not in (A_det, B_det, C_det, D_det):
            parts.append(
                "detrap[eV] - Al203={}, Poly-Si={}, SiOx={}, cSi={}".format(
                    fmt(A_det), fmt(B_det), fmt(C_det), fmt(D_det)
                )
            )
        if A_trp is not None:
            parts.append(" trap[eV] - Al203={}".format(fmt(A_trp)))
        if None not in (Af, Bf, Cf, Df):
            parts.append(
                " f_attempt[s⁻¹]-A={}, B={}, C={}, D={}".format(
                    fmt(Af), fmt(Bf), fmt(Cf), fmt(Df)
                )
            )
        if parts:
            fig.suptitle(" | ".join(parts), fontsize=24)

    # ---------------- existing panel-building loop -----------------
    for i, (tname, T) in enumerate(zip(types, temps)):
        if i < n:
            ax = axes[i]
        tname = str(tname)
        ylog = tname.endswith("log")

        if tname.startswith("anneal_sweep"):
            plot_layer_other_stagetemp(
                mgr,
                stage_filter="firing",
                target_T=float(T),
                layer=layer,
                kind=kind,
                stage_to_plot=stage_to_plot,
                ylog=ylog,
                x_lim_left=x_lim_left,
                ax=ax,
                legend=True
            )

        elif tname.startswith("fire_sweep"):
            plot_layer_other_stagetemp(
                mgr,
                stage_filter="annealing",
                target_T=float(T),
                layer=layer,
                kind=kind,
                stage_to_plot=stage_to_plot,
                ylog=ylog,
                x_lim_left=x_lim_left,
                ax=ax,
                legend=True

            )

        elif tname.startswith("peak_firingT"):
            plot_peaktime_vs_stage(
                mgr,
                target_T=float(T),
                layer=layer,
                kind=kind,
                stage_for_peak="annealing",
                stage_sort="firing",
                ax=ax,
            )

        elif tname.startswith("firing_siox"):
            plot_layer_other_stagetemp(
                mgr,
                stage_filter="annealing",
                target_T=float(T),
                layer=layer,
                kind=kind,
                stage_to_plot="firing",
                ylog=ylog,
                ax=ax,
            )

        elif tname.startswith("exp_overlay"):
            if not exp_together:
                ax1 = None
            else:
                ax1 = ax
            exp.plot_exp_vs_sim_normalized_overlay_hours(
                mgr,
                expt_path='/home/adam/code/diffusion_model/annealing_sweep_NREL.csv',
                firing_temp_C=float(T),
                x_left=1e-5,
                legend_where="right",
                y_clip=(-2,1.2),
                plot=True,
                ax=ax1,
            )
        else:
            ax.set_visible(False)

    # Leave a little room for the suptitle if present
    if show_params_in_title:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    plt.show()
    return fig, axes




def plot_multistage_firing_sweep_panels(
    mgr: SimulationManager,
    *,
    annealing_temp_C: float,
    layer: str = "C",
    kind: str = "trapped",
    ylog: bool = False,
    x_lim_left: float = 1e-3,
    cycle_len: int = 2,
    ):
    """
    Make a 1×N figure where N = number of stages in the schedule
    (e.g. 2, 4, 6,… for firing/annealing cycles).

    Each column is a call to `plot_layer_other_stagetemp` with:
      - stage_filter = "annealing"
      - target_T     = annealing_temp_C (the anneal temp of the firing sweep)
      - stage_to_plot = 0,1,2,... (all stages in order)

    This is effectively like repeatedly using the "firing_siox" and "fire_sweep"
    branches of plot_full_single, but extended to arbitrarily many cycles:

        firing, annealing, firing, annealing, ...

    Color coding and legend follow the same logic as plot_layer_other_stagetemp
    (colors keyed by the *other* stage’s temperature).
    """
    # Find all runs whose *annealing* stage matches the requested temperature
    matches = mgr.results_with_stage_temperature(
        stage="firing1_2",
        target_T=annealing_temp_C,
    )
    print(matches)
    if not matches:
        raise RuntimeError(
            f"No simulations found with annealing ≈ {annealing_temp_C:.1f} °C."
        )

    # Choose the run with the most stages (in case there are shorter variants)
    res0 = max(matches, key=lambda r: len(r.stage_names()))
    stage_names = res0.stage_names()
    n_stages = len(stage_names)

    if n_stages == 0:
        raise RuntimeError("Matched simulation has no stages defined.")

    # Set up 1×N panel layout
    fig, axes = plt.subplots(
        1,
        n_stages,
        figsize=(6 * n_stages, 4),
        sharey=False,
    )
    if n_stages == 1:
        axes = np.atleast_1d(axes)
    curves_for_export = {}
    # Build each panel by stepping through all stage indices
    for idx in range(n_stages):
        ax = axes[idx]

        # Use the same “firing_siox / fire_sweep” logic but generalized:
        # - stage_filter="annealing"
        # - target_T = annealing_temp_C
        # - stage_to_plot = idx
        t_s,y = plot_layer_other_stagetemp(
            mgr,
            stage_filter="firing1_2",
            target_T=float(annealing_temp_C),
            layer=layer,
            kind=kind,
            stage_to_plot=idx,
            ylog=ylog,
            x_lim_left=x_lim_left,
            ax=ax,
            cycle_len=cycle_len,
        )
        
        
        curves_for_export[f"stage_{idx}"] = (
                np.asarray(t_s, float),
                np.asarray(y, float),
            )

        # Nice per-panel title: "firing @ 780°C", "annealing @ 400°C", ...
        stage_name = stage_names[idx]
        try:
            T_stage = res0.temperature_for_stage(stage_name)
            title = f"{stage_name} @ {T_stage:.0f}°C"
        except KeyError:
            title = stage_name
        ax.set_title(title)

        # Drop redundant y-labels on all but the first panel
        # if idx > 0:
        #     ax.set_ylabel("")
        #     ax.tick_params(labelleft=False)

    save_curves_for_origin(
            curves_for_export,
            filepath=f"./origin_exports/{annealing_temp_C}_cycle.csv",
            max_pts=10000
        )
    # Overall title for the whole strip
    # fig.suptitle(
    #     f"All stages for anneal ≈ {annealing_temp_C:.0f}°C",
    #     fontsize=16,
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plt.show()
    return fig, axes



def _unique_agg(x, y, reducer=np.mean):
    """Sort by x and collapse duplicate x by averaging y."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    uniq, idx, counts = np.unique(x, return_index=True, return_counts=True)
    if uniq.size == x.size:
        return x, y
    y_sum = np.add.reduceat(y, idx)
    y_mean = y_sum / counts
    return uniq, y_mean

def smooth_log_x(t, y, n_points=2):
    """
    Robust smoothing for log-x plots:
      - removes nonpositive/NaN
      - sorts, collapses duplicate x
      - cubic spline in log10(t); falls back to PCHIP/linear if needed
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    mask = (t > 0) & np.isfinite(y)
    if mask.sum() < 4:
        return t, y

    t, y = _unique_agg(t[mask], y[mask])
    if t.size < 4:
        return t, y

    logt = np.log10(t)
    # (safety) collapse duplicates introduced by float quirks
    logt, y = _unique_agg(logt, y)

    logt_new = np.linspace(logt.min(), logt.max(), int(max(50, n_points)))

    try:
        from scipy.interpolate import make_interp_spline
        k = min(3, logt.size - 1)  # adapt if few points
        spline = make_interp_spline(logt, y, k=k)
        y_new = spline(logt_new)
    except Exception:
        # fallback: shape-preserving if available, else linear
        try:
            from scipy.interpolate import PchipInterpolator
            y_new = PchipInterpolator(logt, y, extrapolate=False)(logt_new)
        except Exception:
            y_new = np.interp(logt_new, logt, y)

    return 10**logt_new, y_new


def _plot_layers_with_deltaA_for_stage_int_thickness(
    ax,
    res,  # SimulationResult
    *,
    stage: str,              # "firing" | "annealing"
    ylog: bool = False,
    use_seconds: bool = True,
    x_lim_left=None,
    neg_a: bool = False,
):
    """
    Plot per-layer H concentrations (cm^-3) during a stage:

      - A: Δ(concentration) relative to start of stage (cm^-3)
      - B, C: absolute concentration (cm^-3)
      - D+E: thickness-weighted average concentration (cm^-3)

    All curves plotted on ONE shared y-axis.
    """

    # --- thicknesses (for weighting D+E only) ---
    dxA = (SCALE_A * SAMPLE_LENGTH) * 1e-7
    dxB = (SCALE_B * SAMPLE_LENGTH) * 1e-7
    dxC = (SCALE_C * SAMPLE_LENGTH) * 1e-7
    dxD = (SCALE_D * SAMPLE_LENGTH) * 1e-7
    dxE = SCALE_E * 1e-4

    colA, colB, colC, colD = bright_distinct_colors(4)

    def xscale(t):
        return t if use_seconds else (t / 3600.0)

    # ------------------------------------------------------------------
    # A: Δ concentration = nA(t) - nA(t0)
    # ------------------------------------------------------------------
    tA, nA = res.total("A", stage=stage)  # cm^-3
    dA = nA - nA[0]
    if neg_a:
        dA = -dA

    ax.plot(xscale(tA), dA, lw=1.8, color=colA, label="ΔH in Al₂O₃")

    # ------------------------------------------------------------------
    # B, C: absolute concentration (cm^-3)
    # ------------------------------------------------------------------
    for L, col, lab in [
        ("B", colB, "H in poly-Si"),
        ("C", colC, "H in SiOₓ"),
    ]:
        tL, nL = res.total(L, stage=stage)
        ax.plot(xscale(tL), nL, lw=1.8, color=col, label=lab)

    # ------------------------------------------------------------------
    # D + E: weighted concentration (cm^-3)
    # ------------------------------------------------------------------
    tD, nD = res.total("D", stage=stage)
    tE, nE = res.total("E", stage=stage)

    wsum = dxD + dxE
    n_csi = (nD * dxD + nE * dxE) / wsum

    ax.plot(xscale(tD), n_csi, lw=1.8, color=colD, label="H in c-Si (D+E)")

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    if x_lim_left is not None:
        ax.set_xlim(left=x_lim_left)

    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel(
        ("Firing" if stage == "firing" else "Annealing")
        + (" time (s)" if use_seconds else " time (h)")
    )
    ax.set_ylabel("H concentration (cm$^{-3}$)")
    ax.legend(loc="best")




def _plot_layers_with_deltaA_for_stage(
    ax,
    res,  # SimulationResult
    *,
    stage: str,              # "firing" | "annealing"
    ylog: bool = False,
    use_seconds: bool = True,
    x_lim_left=None,
    neg_a: bool = False,
    ):
    """
    Plot per-layer H concentrations (cm^-3) during a stage:
      - A: Δ(concentration) relative to start of stage (cm^-3)
      - B, C: absolute concentration (cm^-3)
      - D+E: thickness-weighted average concentration across c-Si (cm^-3)
    """

    # physical thicknesses (nm→cm, µm→cm) used ONLY to weight D+E
    dxA = (SCALE_A * SAMPLE_LENGTH) * 1e-7
    dxB = (SCALE_B * SAMPLE_LENGTH) * 1e-7
    dxC = (SCALE_C * SAMPLE_LENGTH) * 1e-7
    dxD = (SCALE_D * SAMPLE_LENGTH) * 1e-7
    dxE = SCALE_E * 1e-4

    colA, colB, colC, colD = bright_distinct_colors(4)

    def xscale(t):  # seconds or hours on x
        return t if use_seconds else (t / 3600.0)

    # A: Δ concentration (no thickness multiply)
    tA, nA = res.total("A", stage=stage)  # cm^-3
    dA = (nA - nA[0])                     # cm^-3
    a_scale = -1 if neg_a else 1
    a_label = "|ΔH| in Al₂O₃" if neg_a else "ΔH in Al₂O₃"
    ax.plot(xscale(tA), a_scale * dA, lw=1.8, color=colA, label=a_label)

    # B, C: absolute concentration (no thickness multiply)
    for L, col, lab in [("B", colB, "H in poly-Si"),
                        ("C", colC, "H in SiOx")]:
        tL, nL = res.total(L, stage=stage)  # cm^-3
        ax.plot(xscale(tL), nL, lw=1.8, color=col, label=lab)

    # D+E: thickness-weighted average concentration (cm^-3)
    tD, nD = res.total("D", stage=stage)
    tE, nE = res.total("E", stage=stage)
    wsum = dxD + dxE
    n_csi = (nD * dxD + nE * dxE) / wsum
    ax.plot(xscale(tD), n_csi, lw=1.8, color=colD, label="H in c-Si (D+E, wt. avg)")

    # axes / formatting
    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if x_lim_left is not None:
        ax.set_xlim(left=x_lim_left)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel(("Firing" if stage == "firing" else "Annealing")
                  + (" time (s)" if use_seconds else " time (h)"))
    ax.set_ylabel("H concentration (cm$^{-3}$)")
    ax.legend(loc="best")




def apply_offset_to_axis(
    ax: Axes,
    y_values,
    *,
    base_label: str,
    unit: str = r"cm$^{-2}$",
    color: str | None = None,
    offset: float | None = None,
    offset_mode: str = "initial",   # "initial", "mean", or "min"
    exp: int | None = None,         # force exponent if you want
    ) -> np.ndarray:
    """
    Given an axis and an array of absolute values y_values, compute an
    offset + scale for nicer tick labels and move that information into
    the axis label.

    Returns: y_scaled = (y_values - offset) / 10**exp  (safe for plotting)

    The axis `ax` is updated with:
      - ScalarFormatter without offset text
      - y-label of the form:
          base_label − offset (×10^exp unit)
    """
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(y)
    if not mask.any():
        # Nothing usable; just leave axis alone
        return y

    # --- choose offset ---
    if offset is None:
        if offset_mode == "mean":
            offset = float(np.mean(y[mask]))
        elif offset_mode == "min":
            offset = float(np.min(y[mask]))
        else:  # "initial"
            offset = float(y[mask][0])

    # --- subtract offset for plotting ---
    y_shift = y - offset

    # --- choose exponent for the shifted data ---
    if exp is None:
        max_abs = float(np.max(np.abs(y_shift[mask])))
        if max_abs == 0.0:
            exp = 0
        else:
            exp = int(np.floor(np.log10(max_abs)))
    scale = 10.0**exp if exp != 0 else 1.0

    y_scaled = y_shift / scale

    # --- formatter: no extra offset text, scientific if needed ---
    fmt = ScalarFormatter(useOffset=False, useMathText=True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_visible(False)

    # --- build label text ---
    # Example: H in Al2O3 − 1.0×10^17 (×10^15 cm^-2)
    offset_str = f"{offset:.2e}"
    if exp == 0:
        label = rf"{base_label} − {offset_str} ({unit})"
    else:
        label = (
            rf"{base_label} − {offset_str} "
            rf"(×10$^{{{exp}}}$ {unit})"
        )

    ax.set_ylabel(label)
    if color is not None:
        ax.tick_params(axis="y", colors=color)
        ax.yaxis.label.set_color(color)

    return y_scaled




def _plot_layers(
    ax,
    res,
    *,
    stage: str,
    units: str = "cm3",
    combine_axes: bool = True,
    ylog: bool = False,
    use_seconds: bool = True,
    x_lim_left=None,
    neg_a: bool = False,
    csv_name: str = None,
    plot_trapped_mobile: bool = False,
    legend=False
):
    """
    Unified layer plotter (FIXED):
    - A, B, C plotted normally
    - D+E shown as a single c-Si layer
    - trapped/mobile ALSO combined for c-Si (NOT per-layer D, E)
    """

    curves_for_export = {}

    # Global linestyles
    ls_trapped = (0, (1, 7))  # dotted
    ls_mobile  = "--"         # dashed

    # thicknesses in cm
    dxA = (SCALE_A * SAMPLE_LENGTH) * 1e-7
    dxB = (SCALE_B * SAMPLE_LENGTH) * 1e-7
    dxC = (SCALE_C * SAMPLE_LENGTH) * 1e-7
    dxD = (SCALE_D * SAMPLE_LENGTH) * 1e-7
    dxE = SCALE_E * 1e-4

    colA, colB, colC, colD = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")#bright_distinct_colors(4)

    def xscale(t):
        return t if use_seconds else (t / 3600.0)

    # ======================================================
    # TOTAL CURVES (A, B, C, and combined D+E)
    # ======================================================

    # --- A layer ---
    tT_A, fT_A = res.total("A", stage='firing')
    tA, nA = res.total("A", stage=stage)
    dA = nA - fT_A[0]
    if units == "cm2": dA *= dxA
    if neg_a: dA = -dA
    axA = ax if combine_axes else ax.twinx()
    print(tA[-1])
    axA.plot(xscale(tA), dA, lw=1.8, color=colA,
             label=f"ΔH in Al₂O₃ ({'cm⁻²' if units=='cm2' else 'cm⁻³'})")
    curves_for_export["A_total"] = (xscale(tA).copy(), dA.copy())

    # --- B, C layers ---
    for L, dx, col, name in [
        ("B", dxB, colB, "H in poly-Si"),
        ("C", dxC, colC, "H in SiOₓ")
    ]:
        tL, nL = res.total(L, stage=stage)
        y = nL * dx if units == "cm2" else nL
        print(f"{max(y):.6e}")

        ax.plot(xscale(tL), y, lw=1.8, color=col, label=name)
        curves_for_export[f"{L}_total"] = (xscale(tL).copy(), y.copy())

    # --- D+E COMBINED TOTAL ---
    tD, nD = res.total("D", stage=stage)
    tE, nE = res.total("E", stage=stage)
    if units == "cm2":
        yDE = nD * dxD + nE * dxE
    else:
        w = dxD + dxE
        yDE = (nD * dxD + nE * dxE) / w

    ax.plot(xscale(tD), yDE, lw=1.8, color=colD, label="H in c-Si")
    curves_for_export["cSi_total"] = (xscale(tD).copy(), yDE.copy())

    # ======================================================
    # FIXED PART — COMBINE TRAPPED/MOBILE FOR D+E
    # ======================================================
    if plot_trapped_mobile:
        
        # A trapped/mobile
        tT_A, nT_A = res.trapped("A", stage=stage)
        tM_A, nM_A = res.mobile("A", stage=stage)
        if units == "cm2":
            nT_A *= dxA; nM_A *= dxA
        nT_A = nT_A - nT_A[0]
        nM_A = nM_A - nM_A[0]
        if neg_a:
            nT_A = -nT_A; nM_A = -nM_A

        ax.plot(xscale(tT_A), nT_A, ls=ls_trapped, color=colA, label="Al₂O₃ trapped")
        ax.plot(xscale(tM_A), nM_A, ls=ls_mobile,  color=colA, label="Al₂O₃ mobile")

        # B trapped/mobile
        for L, dx, col, name in [
            ("B", dxB, colB, "poly-Si"),
            ("C", dxC, colC, "SiOₓ")
        ]:
            tT, nT = res.trapped(L, stage=stage)
            tM, nM = res.mobile(L, stage=stage)
            if units == "cm2": nT *= dx; nM *= dx
            ax.plot(xscale(tT), nT, ls=ls_trapped, color=col, label=f"{name} trapped")
            ax.plot(xscale(tM), nM, ls=ls_mobile,  color=col, label=f"{name} mobile")

        # === THE FIX: c-Si trapped/mobile = D+E combined ===
        tT_D, nT_D = res.trapped("D", stage=stage)
        tT_E, nT_E = res.trapped("E", stage=stage)
        tM_D, nM_D = res.mobile("D", stage=stage)
        tM_E, nM_E = res.mobile("E", stage=stage)

        # combine
        nT_csi = nT_D * dxD + nT_E * dxE if units == "cm2" else (nT_D * dxD + nT_E * dxE) / (dxD + dxE)
        nM_csi = nM_D * dxD + nM_E * dxE if units == "cm2" else (nM_D * dxD + nM_E * dxE) / (dxD + dxE)

        ax.plot(xscale(tT_D), nT_csi, ls=ls_trapped, color=colD, label="c-Si trapped")
        ax.plot(xscale(tM_D), nM_csi, ls=ls_mobile,  color=colD, label="c-Si mobile")

    # ======================================================
    # formatting
    # ======================================================

    ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    if x_lim_left: ax.set_xlim(left=x_lim_left)
    ax.grid(True, which="both",alpha=0.3)

    ylabel = "H (" + ("cm⁻²" if units == "cm2" else "cm⁻³") + ")"
    ax.set_xlabel(("Firing" if stage=="firing" else "Annealing") +
                  (" time (s)" if use_seconds else " time (h)"))
    ax.set_ylabel(ylabel)

    if legend:
        ax.legend(loc="best")

    if csv_name:
        save_curves_for_origin(curves_for_export,
                               f"./origin_exports/{csv_name}.csv",
                               max_pts=20000)




def plot_two_stage_dHdt_panels(
    mgr: SimulationManager,
    *,
    firing_temp_C: float,
    annealing_temp_C: float,
    ylog: bool = False,
    use_seconds: bool = False,   # seconds vs hours on x-axis
    ):
    """
    1×2 figure:
      left  = firing stage normalized dH/dt
      right = annealing stage normalized dH/dt

    Panels now have their own y-axis scales and labels.
    """
    tol = getattr(mgr, "_tol", 0.1)

    fire_matches = mgr.results_with_stage_temperature(
        stage="firing", target_T=float(firing_temp_C)
    )

    matches = []
    for r in fire_matches:
        try:
            if abs(r.temperature_for_stage("annealing") - float(annealing_temp_C)) <= tol:
                matches.append(r)
        except KeyError:
            continue

    if not matches:
        raise RuntimeError(
            f"No run found with firing≈{firing_temp_C}°C and annealing≈{annealing_temp_C}°C."
        )

    print(len(matches))
    res = matches[0]

    # ---- separate y-axes (sharey=False) ----
    # fig, (ax_fire, ax_ann) = plt.subplots(
    #     1, 2,
    #     figsize=(11, 4.4),
    #     sharey=False,   # <-- important
    #     gridspec_kw={"wspace": 0.25},
    # )
    ax_fire=None
    fig, ax_ann = plt.subplots(
        1, 1,
        figsize=(6, 4.4),
    )

    # plot_dHdt_areal(ax_fire, res, stage="firing", use_hours=False, x_lim_left=1e-3)
    # ax_fire.set_title(f"Firing @ {firing_temp_C:.0f}°C")

    # Annealing stage
    plot_dHdt_areal(ax_ann,  res, stage="annealing", use_hours=True,x_lim_left=1e-2)
    ax_ann.set_title(f"Annealing @ {annealing_temp_C:.0f}°C")

    # The two plots now keep their y-labels
    # (no axis label suppression here)

    plt.show()
    return fig, (ax_fire, ax_ann)



def plot_two_stage_deltaA_panels(
    mgr: SimulationManager,
    *,
    firing_temp_C: float,
    annealing_temp_C: float,
    ylog: bool = False,
    use_seconds: bool = False,
    ):
    tol = getattr(mgr, "_tol", 0.1)

    fire_matches = mgr.results_with_stage_temperature(
        stage="firing", target_T=float(firing_temp_C)
    )

    matches = []
    for r in fire_matches:
        try:
            if abs(r.temperature_for_stage("annealing") - float(annealing_temp_C)) <= tol:
                matches.append(r)
        except KeyError:
            continue

    if not matches:
        raise RuntimeError(
            f"No run found with firing≈{firing_temp_C}°C and annealing≈{annealing_temp_C}°C."
        )

    res = matches[0]

    #code to make the A layer have it's own axis
    # # # # fig, (ax_fire, ax_ann) = plt.subplots(
    # # # #     1, 2,
    # # # #     figsize=(10, 4.2),
    # # # #     sharey=True,                 # shares primary axis only
    # # # #     gridspec_kw={"wspace": 0.05},
    # # # # )

    # # # # # Firing: A on its own axis, but axis is hidden (no ticks/label)
    # # # # _plot_layers_with_A_secondary(
    # # # #     ax_fire,
    # # # #     res,
    # # # #     stage="firing",
    # # # #     ylog=ylog,
    # # # #     use_seconds=True,
    # # # #     x_lim_left=1e-3,
    # # # #     show_A_axis=False,
    # # # #     csv_name="H_dt_firing"
    # # # # )

    # # # # # Annealing: A on its own visible right axis
    # # # # _plot_layers_with_A_secondary(
    # # # #     ax_ann,
    # # # #     res,
    # # # #     stage="annealing",
    # # # #     ylog=ylog,
    # # # #     use_seconds=use_seconds,
    # # # #     x_lim_left=1e-2,
    # # # #     show_A_axis=True,
    # # # #     csv_name="H_dt_annealing"
    # # # # )
    
    fig, (ax_fire, ax_ann) = plt.subplots(
        1, 2,
        figsize=(25, 10),
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    
    # === Firing: ΔH(A) version ===
    _plot_layers(
        ax_fire,
        res,
        stage="firing",
        units="cm2",
        combine_axes=True,
        use_seconds=True,
        x_lim_left=1e-3,
        csv_name="H_dt_firing",
        legend=True
    )

    _plot_layers(
        ax_ann,
        res,
        stage="annealing",
        units="cm2",
        combine_axes=True,
        use_seconds=False,
        x_lim_left=1e-2,
        csv_name="H_dt_annealing"
    )




    # Clean up left label on the right panel
    ax_ann.set_ylabel("")
    ax_ann.tick_params(labelleft=False)
    plt.suptitle(f"Firing @ {firing_temp_C:.0f}°C and Annealing @ {annealing_temp_C:.0f}°C", fontsize=16)
    plt.show()
    return fig, (ax_fire, ax_ann)

def plot_two_stage_flux_panels(
    mgr: SimulationManager,
    *,
    firing_temp_C: float,
    annealing_temp_C: float,
    ylog: bool = False,
    use_seconds: bool = False,
    linthresh=1e0,
):
    tol = getattr(mgr, "_tol", 0.1)

    fire_matches = mgr.results_with_stage_temperature(
        stage="firing", target_T=float(firing_temp_C)
    )

    matches = []
    for r in fire_matches:
        try:
            if abs(r.temperature_for_stage("annealing") - float(annealing_temp_C)) <= tol:
                matches.append(r)
        except KeyError:
            pass

    if not matches:
        raise RuntimeError("No matching run found.")

    res = matches[0]

    # thicknesses in cm
    dxA = (SCALE_A * SAMPLE_LENGTH) * 1e-7
    dxB = (SCALE_B * SAMPLE_LENGTH) * 1e-7
    dxC = (SCALE_C * SAMPLE_LENGTH) * 1e-7
    dxD = (SCALE_D * SAMPLE_LENGTH) * 1e-7

    fig, (axF, axA) = plt.subplots(1, 2, figsize=(11, 4.5))

    names  = [r"$Al_2O_3 → poly\text{-}Si$", r"$poly\text{-}Si → SiO_x$", r"$SiO_x → c\text{-}Si_{sur} $", r"$c\text{-}Si_{sur} → c\text{-}Si_{bulk}$"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    dx     = [dxA, dxB, dxC, dxD]

    def _plot(ax, stage_name, title,use_seconds,x_lim_left=None):
        t, F = res.fluxes(stage=stage_name)
        t = np.asarray(t, float)
        F = np.asarray(F, float)

        # convert to cm^-2 s^-1
        F_area = np.zeros_like(F)
        for i in range(4):
            F_area[:, i] = F[:, i] * dx[i]

        xs = t if use_seconds else (t / 3600.0)

        for i in range(4):
            ax.plot(xs, F_area[:, i], lw=1.4, color=colors[i], label=names[i])

        ax.set_xscale("log")
        if ylog:
            ax.set_yscale("symlog", linthresh=linthresh)
        if x_lim_left is not None:
            ax.set_xlim(left=x_lim_left)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("time (h)" if not use_seconds else "time (s)")
        ax.set_ylabel("H flux per area (cm⁻² s⁻¹)")
        ax.legend()

    _plot(axF, "firing",   f"Firing @ {firing_temp_C}°C",use_seconds=True,x_lim_left=1e-3)
    _plot(axA, "annealing", f"Annealing @ {annealing_temp_C}°C",use_seconds=False,x_lim_left=1e-2)

    plt.tight_layout()
    return fig, (axF, axA)




def save_curves_for_origin(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    filepath: str,
    max_pts: int = 3000
    ):
    """
    Save multiple curves into one Origin-friendly CSV.

    curves:
        { "label": (t_array, y_array), ... }

    Each curve becomes two columns:
        time_<label>, value_<label>
    """
    # return
    # downsample helper
    def down(t, y):
        return t, y
        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]
        n = t.size
        if n > max_pts:
            idx = np.linspace(0, n - 1, max_pts).astype(int)
            t = t[idx]
            y = y[idx]
        return t, y

    # ensure folder exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Prepare merged table
    # Some curves may have different lengths → pad with empty cells
    all_labels = list(curves.keys())

    # Downsample all first
    curves_ds = {lbl: down(*curves[lbl]) for lbl in all_labels}
    max_len = max(len(v[0]) for v in curves_ds.values())

    # Prepare column headers
    headers = []
    for lbl in all_labels:
        safe = lbl.replace(" ", "_")
        headers += [f"time_{safe}", f"value_{safe}"]

    # Write CSV
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        # row-wise fill
        for i in range(max_len):
            row = []
            for lbl in all_labels:
                t, y = curves_ds[lbl]
                if i < len(t):
                    row += [t[i], y[i]]
                else:
                    row += ["", ""]
            w.writerow(row)

    print(f"[save_curves_for_origin] Saved {len(all_labels)} curves → {filepath}")






#older versions of plotting code below

# def _plot_layers_with_A_secondary(
#     ax,
#     res,  # SimulationResult
#     *,
#     stage: str,              # "firing" | "annealing"
#     csv_name,
#     ylog: bool = False,
#     use_seconds: bool = True,
#     x_lim_left=None,
#     show_A_axis: bool = False,   # show label/ticks for A on this axes?
#     color_A_axis: bool = False,
#     ):
#     """
#     Plot per-layer areal H contents (cm^-2) during a stage:

#       - A: absolute H in Al2O3 on a secondary y-axis
#       - B, C: H in poly-Si and SiOx (primary y-axis)
#       - D+E: combined c-Si (primary y-axis)
#     """
#     from matplotlib.ticker import ScalarFormatter
    
#     curves_for_export = {}
#     # --- Physical thicknesses (nm → cm or µm → cm) ---
#     dxA = (SCALE_A * SAMPLE_LENGTH) * 1e-7
#     dxB = (SCALE_B * SAMPLE_LENGTH) * 1e-7
#     dxC = (SCALE_C * SAMPLE_LENGTH) * 1e-7
#     dxD = (SCALE_D * SAMPLE_LENGTH) * 1e-7
#     dxE = SCALE_E * 1e-4  # µm → cm

#     # --- colors ---
#     colA, colB, colC, colD = bright_distinct_colors(4)

#     def xscale(t):
#         return t if use_seconds else (t / 3600.0)

#     # --------- A: absolute H in Al2O3 (secondary axis) ---------
#     axA = ax.twinx()
#     axA.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#     # axA.yaxis.get_offset_text().set_visible(False)

#     tA, nA_cm3 = res.total("A", stage=stage)
#     yA_cm2 = nA_cm3 * dxA

#     t_s, y_s = smooth_log_x(xscale(tA), yA_cm2)
#     curves_for_export["H_layerA"]=(t_s.copy(),y_s.copy())
#     axA.plot(t_s, y_s, lw=1.8, color=colA, label=r"H in Al$_2$O$_3$")
#     ax.plot(0,0,color=colA, label=r"H in Al$_2$O$_3$")
#     if ylog:
#         axA.set_yscale("log")

#     if show_A_axis:
#         # Label and color (see next section for color toggle)
#         if color_A_axis:
#             axA.set_ylabel(r"H in Al$_2$O$_3$ (cm$^{-2}$)", color=colA)
#             axA.tick_params(axis="y", colors=colA)
#         else:
#             axA.set_ylabel(r"H in Al$_2$O$_3$ (cm$^{-2}$)")
#             axA.tick_params(axis="y")  # defaults (black
            
#         fmt = ScalarFormatter()#useOffset=False)
#         fmt.set_powerlimits((-2, 2))  # optional: controls sci-notation threshold
#         axA.yaxis.set_major_formatter(fmt)
#     else:
#         # Hide the interior right axis (left panel) but keep its data
#         axA.tick_params(right=False, labelright=False)
#         axA.spines["right"].set_visible(False)

    
#     # --------- B, C, D+E: on primary axis ---------
#     for L, dx, col, lab in [
#         ("B", dxB, colB, "H in poly-Si"),
#         ("C", dxC, colC, "$\\Delta$H in SiO$_x$"),
#     ]:
#         tL, nL_cm3 = res.total(L, stage=stage)
#         curves_for_export[f"H_layer{L}"]=(xscale(tL).copy(),(nL_cm3* dx).copy())
#         ax.plot(xscale(tL), nL_cm3 * dx, lw=1.8, color=col, label=lab)

#     tD, nD_cm3 = res.total("D", stage=stage)
#     tE, nE_cm3 = res.total("E", stage=stage)
#     y_csi_cm2 = nD_cm3 * dxD + nE_cm3 * dxE
#     curves_for_export[f"H_layerD"]=(xscale(tD).copy(),y_csi_cm2.copy())
#     ax.plot(xscale(tD), y_csi_cm2, lw=1.8, color=colD, label="H in c-Si")

    
#     # --- formatting for primary axis ---
#     ax.set_xscale("log")
#     if ylog:
#         ax.set_yscale("log")

#     if x_lim_left is not None:
#         ax.set_xlim(left=x_lim_left)

#     ax.grid(True, which="both", alpha=0.3)
#     ax.set_xlabel(
#         ("Firing" if stage == "firing" else "Annealing")
#         + (" time (s)" if use_seconds else " time (h)")
#     )
#     ax.set_ylabel("H areal density (cm$^{-2}$)")

#     # Legend only from primary axis; A is described on the right axis label
#     if show_A_axis:
        
#         ax.legend(loc="best")
        
#     save_curves_for_origin(
#             curves_for_export,
#             filepath=f"./origin_exports/{csv_name}.csv",
#             max_pts=10000
#         )

# def _plot_layers_dHdt_for_stage(
#     ax,
#     res,
#     *,
#     stage,
#     ylog=False,
#     use_seconds=True,
#     x_lim_left=None,
# ):
#     # choose AU scaling to match old code
#     H_UNIT = 1e22     # matches ut[:]*1e22 conversion
#     DDT_SCALE = 1e3   # matches old 1e8 derivative scaling

#     def xscale(t):
#         return t if use_seconds else (t / 3600.0)

#     def dHdt_AU(t, H_cm3):
#         H_norm = H_cm3 / H_UNIT
#         dHdt = np.gradient(H_norm, t)
#         return DDT_SCALE * dHdt   # exact reproduction of old AU figs

#     # pull concentrations in cm^-3
#     tA, nA = res.total("A", stage=stage)
#     tB, nB = res.total("B", stage=stage)
#     tC, nC = res.total("C", stage=stage)
#     tD, nD = res.total("D", stage=stage)
#     tE, nE = res.total("E", stage=stage)

#     # weighted c-Si
#     dxD = (SCALE_D * SAMPLE_LENGTH) * 1e-7
#     dxE = SCALE_E * 1e-4
#     wsum = dxD + dxE
#     n_csi = (nD * dxD + nE * dxE) / wsum

#     # derivative curves (AU)
#     ax.plot(xscale(tA), dHdt_AU(tA, nA), color="r", label=r"$\partial_t H$ in Al$_2$O$_3$")
#     ax.plot(xscale(tB), dHdt_AU(tB, nB), color="b", label=r"$\partial_t H$ in poly-Si$")
#     ax.plot(xscale(tC), dHdt_AU(tC, nC), color="g", label=r"$\partial_t H$ in SiO$_x$")
#     ax.plot(xscale(tD), dHdt_AU(tD, n_csi), color="m",
#            label=r"$\partial_t H$ in c-Si$\,\text{(D+E)}$")

#     ax.set_xscale("log")
#     if ylog:
#         ax.set_yscale("symlog", linthresh=1e-4)

#     if x_lim_left is not None:
#         ax.set_xlim(left=x_lim_left)

#     ax.grid(True, alpha=0.3)
#     ax.set_ylabel(r"$\partial H/\partial t$ (AU)")
#     ax.set_xlabel(("Firing" if stage=="firing" else "Annealing") +
#                   (" time (s)" if use_seconds else " time (h)"))
#     ax.legend()

# def plot_dHdt_areal(
#     ax,
#     res,
#     *,
#     stage: str,
#     use_hours=False,
#     x_lim_left=None
#     ):
#     """
#     Plot d/dt of TOTAL H PER AREA for each layer (cm^-2/s or AU),
#     matching the logic of the old PETSc code.
#     """
#     curves_for_export = {}
    
#     # --- layer thicknesses in cm ---
#     dxA = (SCALE_A)
#     dxB = (SCALE_B)
#     dxC = (SCALE_C)
#     dxD = (SCALE_D)
#     dxE = SCALE_E

#     yscale=1e13
#     # --- colors ---
#     colA, colB, colC, colD = bright_distinct_colors(4)

#     # --- helper for time axis ---
#     def xscale(t):
#         return t / 3600.0 if use_hours else t

#     # ---- get total concentrations for each layer ----
#     # res.total(layer) returns (t, n[layer](t)) with n in cm^-3
#     tA, nA = res.total("A", stage=stage)
#     tB, nB = res.total("B", stage=stage)
#     tC, nC = res.total("C", stage=stage)
#     tD, nD = res.total("D", stage=stage)
#     tE, nE = res.total("E", stage=stage)

#     # --- convert to TOTAL H PER AREA (cm^-2) ---
#     HA = nA * dxA / yscale
#     HB = nB * dxB / yscale
#     HC = nC * dxC / yscale
#     HDE = (nD/ yscale) * dxD + (nE/ yscale) * dxE  # combined c-Si

#     # --- compute time derivatives ---
#     dHA_dt = np.gradient(HA, tA)
#     dHB_dt = np.gradient(HB, tB)
#     dHC_dt = np.gradient(HC, tC)
#     dHDE_dt = np.gradient(HDE, tD)

#     # --- plot ---
#     curves_for_export['Layer_A']=(xscale(tA).copy(), dHA_dt.copy)
#     curves_for_export['Layer_B']=(xscale(tB).copy(), dHB_dt.copy)
#     curves_for_export['Layer_C']=(xscale(tC).copy(), dHC_dt.copy)
#     curves_for_export['Layer_D']=(xscale(tD).copy(), dHDE_dt.copy)
    
#     ax.plot(xscale(tA), dHA_dt, color=colA, label=r'$Al_2O_3$')
#     ax.plot(xscale(tB), dHB_dt, color=colB, label=r'$poly-Si$')
#     ax.plot(xscale(tC), dHC_dt, color=colC, label=r'$SiO_x$')
#     ax.plot(xscale(tD), dHDE_dt, color=colD, label=r'$c-Si (D+E)$')

#     ax.set_xscale("log")
#     if x_lim_left is not None:
#         ax.set_xlim(left=x_lim_left)
        
#     ax.set_ylim(bottom=-1,top=1)
    
#     # ax.set_yscale("symlog")
#     ax.set_xlabel(("Time (h)" if use_hours else "Time (s)"))

#     ax.set_ylabel(r'dH/dt (cm$^{-2}$ s$^{-1}$)')
#     ax.grid(True, alpha=0.3)
#     ax.legend()
    
#     save_curves_for_origin(
#             curves_for_export,
#             filepath=f",.origin_exports/H_dt.csv",
#             max_pts=10000
#         )
