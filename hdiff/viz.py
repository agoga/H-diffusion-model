"""Plot helpers for hdiff simulations and parity checks.

Architecture overview
---------------------
This module is built around a simple two-step pipeline:

1) Build traces (data extraction)
    - A *trace* is one plottable time-series line: ``(t_s, y, label, style)``.
    - ``TraceData`` is the normalized container for that line.
    - Builder functions create traces from model objects:
      - ``build_simulation_trace``: one line from one ``Simulation`` selection.
      - ``build_campaign_stage_sweep_traces``: many lines from a ``Campaign``
         filtered by stage temperature.

2) Render traces (drawing/styling)
    - ``plot_traces`` is the shared renderer for any list of ``TraceData``.
    - It applies axis scales, labels, grid, optional x-limits, and legend.

Higher-level plot functions (for example ``plot_layer_stage_sweep`` and
``plot_all_layers_for_stage``) should mostly compose these primitives instead
of duplicating extraction and plotting logic.

Extension pattern
-----------------
To add a new plot type:

1. Write a builder that returns ``list[TraceData]`` from your source data.
2. Call ``plot_traces`` with those traces.
3. Add any panel-specific title/labels and legend-title tweaks.

This keeps new visualizations short and consistent while making styling and
axis behavior reusable across all panels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .campaign import (
    filter_simulations_by_stage_temperature as _campaign_filter_simulations,
    stage_temperature_C as _campaign_stage_temperature_C,
)
from .sim import Simulation


@dataclass(frozen=True)
class TraceData:
    """One drawable time-series trace with minimal style metadata."""

    t_s: np.ndarray
    y: np.ndarray
    label: str
    color: Any | None = None
    linestyle: str = "-"
    linewidth: float = 1.8
    alpha: float = 1.0


def build_simulation_trace(
    sim: Simulation,
    *,
    layer: str,
    kind: str,
    stage: str,
    occurrence: int = 0,
    units: str = "cm^-3",
    x_units: str = "seconds",
    label: str | None = None,
    color: Any | None = None,
    linestyle: str = "-",
    linewidth: float = 1.8,
    alpha: float = 1.0,
) -> TraceData:
    """Build one trace from a simulation series selection.

    This is the basic extraction primitive for simulation-backed plots.
    """

    t_s, y = sim.series(
        layer=layer,
        kind=kind,
        stage=stage,
        occurrence=occurrence,
        units=units,
    )
    x_scale = 3600.0 if x_units == "hours" else 1.0
    t = _to_array(t_s) / x_scale
    values = _to_array(y)
    return TraceData(
        t_s=t,
        y=values,
        label=(label if label is not None else layer),
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
    )


def plot_traces(
    ax,
    *,
    traces: Sequence[TraceData],
    logx: bool = True,
    logy: bool = True,
    title: str | None = None,
    xlabel: str = "Time (s)",
    ylabel: str = "Concentration (cm^-3)",
    legend: bool = True,
    x_units: str = "seconds",
    xlim: tuple[float, float] | None = None,
    min_x_hours: float | None = 1e-3,
) -> None:
    """Render a list of traces on one axis with shared axis styling."""

    if not traces:
        raise ValueError("traces must be non-empty")

    for trace in traces:
        kwargs: dict[str, Any] = {
            "linewidth": float(trace.linewidth),
            "linestyle": trace.linestyle,
            "alpha": float(trace.alpha),
            "label": trace.label,
        }
        if trace.color is not None:
            kwargs["color"] = trace.color
        ax.plot(trace.t_s, trace.y, **kwargs)

    _style_axes(
        ax,
        logx=logx,
        logy=logy,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    _apply_time_xlim(
        ax,
        x_units=x_units,
        xlim=xlim,
        min_x_hours=min_x_hours,
    )
    if legend:
        ax.legend()


def _to_array(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected 1-D array")
    return arr


def _interp(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    if t_src.size == 0 or y_src.size == 0:
        raise ValueError("source trace must be non-empty")
    if t_src.shape != y_src.shape:
        raise ValueError("source time and values must have matching shape")
    if t_dst.size == 0:
        raise ValueError("destination time axis must be non-empty")
    return np.interp(t_dst, t_src, y_src)


def _log_probe_grid(ref_t_s: np.ndarray, n_points: int = 2048) -> np.ndarray:
    ref_t = _to_array(ref_t_s)
    if n_points < 3:
        raise ValueError("n_points must be >= 3")
    t_max = float(np.max(ref_t))
    if t_max <= 0.0:
        return np.linspace(0.0, 0.0, n_points)
    positive = ref_t[ref_t > 0.0]
    t_min = float(np.min(positive)) if positive.size else t_max / 1e6
    t_min = max(t_min, t_max / 1e12, 1e-12)
    geom = np.geomspace(t_min, t_max, n_points - 1)
    return np.unique(np.concatenate(([0.0], geom)))


def _stage_segment(sim: Simulation, stage: str, occurrence: int = 0):
    compiled = sim.schedule.compile()
    segs = [seg for seg in compiled if seg.stage == stage]
    if occurrence < 0 or occurrence >= len(segs):
        raise ValueError(f"stage occurrence out of range for stage={stage}: {occurrence}")
    return segs[occurrence]


def stage_temperature_C(sim: Simulation, stage: str, occurrence: int = 0) -> float:
    """Return the stage temperature in Celsius for one simulation and stage."""

    return _campaign_stage_temperature_C(sim, stage, occurrence)


def filter_simulations_by_stage_temperature(
    campaign: Any,
    *,
    stage: str,
    target_temp_C: float,
    tolerance_C: float = 1e-6,
    occurrence: int = 0,
) -> list[Simulation]:
    """Select simulations matching stage temperature from a campaign object."""

    return _campaign_filter_simulations(
        campaign,
        stage=stage,
        target_temp_C=target_temp_C,
        tolerance_C=tolerance_C,
        occurrence=occurrence,
    )


def _other_stage_name(sim: Simulation, stage: str) -> str | None:
    names: list[str] = []
    for seg in sim.schedule.compile():
        if seg.stage not in names:
            names.append(seg.stage)
    for name in names:
        if name != stage:
            return name
    return None


def _style_axes(ax, *, logx: bool, logy: bool, title: str | None, xlabel: str, ylabel: str) -> None:
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)


def _apply_time_xlim(
    ax,
    *,
    x_units: str,
    xlim: tuple[float, float] | None,
    min_x_hours: float | None,
) -> None:
    """Apply caller x-limits, with optional default hour-scale lower bound.

    Priority:
    1) explicit xlim from caller
    2) if viewing in hours, clamp left x-limit to min_x_hours (default 1e-3)
    """
    if xlim is not None:
        ax.set_xlim(xlim)
        return

    if x_units != "hours" or min_x_hours is None:
        return

    left, right = ax.get_xlim()
    min_left = float(min_x_hours)
    if right <= min_left:
        right = min_left * 10.0
    if left < min_left:
        ax.set_xlim(min_left, right)


def build_campaign_stage_sweep_traces(
    campaign: Any,
    *,
    match_stage: str,
    target_temp_C: float,
    layer: str = "C",
    kind: str = "total",
    plot_stage: str = "annealing",
    match_occurrence: int = 0,
    plot_occurrence: int = 0,
    units: str = "cm^-3",
    tolerance_C: float = 1e-6,
    x_units: str = "seconds",
) -> tuple[list[TraceData], str | None]:
    """Build overlay traces for campaign runs selected by one stage temperature.

    Returns ``(traces, other_stage_name)`` where ``other_stage_name`` is used as
    legend title when available.
    """

    import matplotlib.pyplot as plt

    matched = filter_simulations_by_stage_temperature(
        campaign,
        stage=match_stage,
        target_temp_C=target_temp_C,
        tolerance_C=tolerance_C,
        occurrence=match_occurrence,
    )
    if not matched:
        raise ValueError(
            f"no simulations found with {match_stage} approximately {target_temp_C}C"
        )

    other_stage = _other_stage_name(matched[0], match_stage)
    color_key_vals: list[float] = []
    if other_stage is not None:
        for sim in matched:
            try:
                color_key_vals.append(stage_temperature_C(sim, other_stage, 0))
            except Exception:
                pass

    cmap = plt.get_cmap("viridis")
    if color_key_vals:
        vmin = float(np.min(color_key_vals))
        vmax = float(np.max(color_key_vals))
        span = max(vmax - vmin, 1e-12)
    else:
        vmin = vmax = 0.0
        span = 1.0

    traces: list[TraceData] = []
    for i, sim in enumerate(matched):
        color = cmap(0.5)
        label = f"run_{i}"
        if other_stage is not None:
            try:
                other_temp = stage_temperature_C(sim, other_stage, 0)
                frac = (other_temp - vmin) / span
                color = cmap(float(np.clip(frac, 0.0, 1.0)))
                label = f"{other_stage}={other_temp:.0f}C"
            except Exception:
                pass
        traces.append(
            build_simulation_trace(
                sim,
                layer=layer,
                kind=kind,
                stage=plot_stage,
                occurrence=plot_occurrence,
                units=units,
                x_units=x_units,
                label=label,
                color=color,
            )
        )

    return traces, other_stage


def plot_trace_overlay(
    ax,
    *,
    t_ref_s: Sequence[float] | np.ndarray,
    y_ref: Sequence[float] | np.ndarray,
    t_sim_s: Sequence[float] | np.ndarray,
    y_sim: Sequence[float] | np.ndarray,
    probe_t_s: Sequence[float] | np.ndarray | None = None,
    ref_label: str = "reference",
    sim_label: str = "hdiff",
    title: str | None = None,
    logx: bool = True,
    logy: bool = True,
) -> None:
    """Panel: overlay reference and simulation traces on one axis."""

    t_ref = _to_array(t_ref_s)
    y_ref_arr = _to_array(y_ref)
    t_sim = _to_array(t_sim_s)
    y_sim_arr = _to_array(y_sim)

    if probe_t_s is None:
        x = t_ref
        y_ref_plot = y_ref_arr
        y_sim_plot = _interp(t_sim, y_sim_arr, x)
    else:
        x = _to_array(probe_t_s)
        y_ref_plot = _interp(t_ref, y_ref_arr, x)
        y_sim_plot = _interp(t_sim, y_sim_arr, x)

    traces = [
        TraceData(t_s=x, y=y_ref_plot, label=ref_label, linewidth=2.0),
        TraceData(t_s=x, y=y_sim_plot, label=sim_label, linestyle="--", linewidth=1.7),
    ]
    plot_traces(
        ax,
        traces=traces,
        logx=logx,
        logy=logy,
        title=title,
        xlabel="Time (s)",
        ylabel="Concentration (cm^-3)",
        legend=True,
        x_units="seconds",
        xlim=None,
        min_x_hours=None,
    )


def plot_abs_error(
    ax,
    *,
    t_ref_s: Sequence[float] | np.ndarray,
    y_ref: Sequence[float] | np.ndarray,
    t_sim_s: Sequence[float] | np.ndarray,
    y_sim: Sequence[float] | np.ndarray,
    probe_t_s: Sequence[float] | np.ndarray | None = None,
    title: str | None = None,
    logx: bool = True,
    logy: bool = True,
) -> None:
    """Panel: absolute error |sim-ref| versus time."""

    t_ref = _to_array(t_ref_s)
    y_ref_arr = _to_array(y_ref)
    t_sim = _to_array(t_sim_s)
    y_sim_arr = _to_array(y_sim)

    x = t_ref if probe_t_s is None else _to_array(probe_t_s)
    ref_on_x = _interp(t_ref, y_ref_arr, x)
    sim_on_x = _interp(t_sim, y_sim_arr, x)
    abs_err = np.abs(sim_on_x - ref_on_x)

    ax.plot(x, abs_err, color="crimson", linewidth=1.8)
    _style_axes(
        ax,
        logx=logx,
        logy=logy,
        title=title or "Absolute error",
        xlabel="Time (s)",
        ylabel="|sim - ref|",
    )


def plot_rel_error(
    ax,
    *,
    t_ref_s: Sequence[float] | np.ndarray,
    y_ref: Sequence[float] | np.ndarray,
    t_sim_s: Sequence[float] | np.ndarray,
    y_sim: Sequence[float] | np.ndarray,
    probe_t_s: Sequence[float] | np.ndarray | None = None,
    title: str | None = None,
    logx: bool = True,
    logy: bool = True,
    eps: float = 1e-30,
) -> None:
    """Panel: pointwise relative error |sim-ref| / max(|ref|, eps)."""

    t_ref = _to_array(t_ref_s)
    y_ref_arr = _to_array(y_ref)
    t_sim = _to_array(t_sim_s)
    y_sim_arr = _to_array(y_sim)

    x = t_ref if probe_t_s is None else _to_array(probe_t_s)
    ref_on_x = _interp(t_ref, y_ref_arr, x)
    sim_on_x = _interp(t_sim, y_sim_arr, x)
    rel = np.abs(sim_on_x - ref_on_x) / np.maximum(np.abs(ref_on_x), float(eps))

    ax.plot(x, rel, color="darkorange", linewidth=1.8)
    _style_axes(
        ax,
        logx=logx,
        logy=logy,
        title=title or "Pointwise relative error",
        xlabel="Time (s)",
        ylabel="|sim-ref| / |ref|",
    )


def make_parity_figure(
    results: Sequence[Mapping[str, Any]],
    *,
    probe_t_s: np.ndarray | None = None,
    n_probe_points: int = 2048,
):
    """Build a parity figure with rows=cases and cols=overlay/abs/rel error."""

    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("results must be non-empty")

    n_cases = len(results)
    fig, axes = plt.subplots(n_cases, 3, figsize=(16, 4.6 * n_cases), constrained_layout=True)
    if n_cases == 1:
        axes = np.asarray([axes])

    for i, result in enumerate(results):
        case = result.get("case")
        case_name = getattr(case, "name", f"case_{i}")

        t_ref = _to_array(result["t_ref"])
        y_ref = _to_array(result["y_ref"])
        t_sim = _to_array(result["t_new"])
        y_sim = _to_array(result["y_new"])

        probe = probe_t_s
        if probe is None:
            probe = _log_probe_grid(t_ref, n_points=n_probe_points)

        plot_trace_overlay(
            axes[i, 0],
            t_ref_s=t_ref,
            y_ref=y_ref,
            t_sim_s=t_sim,
            y_sim=y_sim,
            probe_t_s=probe,
            title=f"{case_name} overlay",
        )
        plot_abs_error(
            axes[i, 1],
            t_ref_s=t_ref,
            y_ref=y_ref,
            t_sim_s=t_sim,
            y_sim=y_sim,
            probe_t_s=probe,
            title=f"{case_name} abs error",
        )
        plot_rel_error(
            axes[i, 2],
            t_ref_s=t_ref,
            y_ref=y_ref,
            t_sim_s=t_sim,
            y_sim=y_sim,
            probe_t_s=probe,
            title=f"{case_name} rel error",
        )

    return fig, axes


def plot_layer_stage_sweep(
    ax,
    *,
    campaign: Any,
    match_stage: str,
    target_temp_C: float,
    layer: str = "C",
    kind: str = "total",
    plot_stage: str = "annealing",
    match_occurrence: int = 0,
    plot_occurrence: int = 0,
    units: str = "cm^-3",
    tolerance_C: float = 1e-6,
    x_units: str = "seconds",
    logx: bool = True,
    logy: bool = True,
    legend: bool = True,
    xlim: tuple[float, float] | None = None,
    min_x_hours: float | None = 1e-3,
) -> None:
    """Panel: one layer during one stage for all runs matching one stage temp.

    Example: hold firing at 650C, then overlay annealing-stage traces colored by
    annealing temperature.
    """

    traces, other_stage = build_campaign_stage_sweep_traces(
        campaign,
        match_stage=match_stage,
        target_temp_C=target_temp_C,
        layer=layer,
        kind=kind,
        plot_stage=plot_stage,
        match_occurrence=match_occurrence,
        plot_occurrence=plot_occurrence,
        units=units,
        tolerance_C=tolerance_C,
        x_units=x_units,
    )

    x_label = "Time in stage (hours)" if x_units == "hours" else "Time in stage (s)"
    plot_traces(
        ax,
        traces=traces,
        logx=logx,
        logy=logy,
        title=f"{layer} {kind} during {plot_stage} | {match_stage}={target_temp_C:.0f}C",
        xlabel=x_label,
        ylabel=f"{layer} {kind} concentration ({units})",
        legend=legend,
        x_units=x_units,
        xlim=xlim,
        min_x_hours=min_x_hours,
    )
    if legend and other_stage is not None:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title(f"{other_stage} temp")


def plot_all_layers_for_stage(
    ax,
    *,
    sim: Simulation,
    stage: str,
    kind: str = "total",
    occurrence: int = 0,
    units: str = "cm^-3",
    x_units: str = "seconds",
    logx: bool = True,
    logy: bool = True,
    legend: bool = True,
    xlim: tuple[float, float] | None = None,
    min_x_hours: float | None = 1e-3,
) -> None:
    """Panel: overlay all layers for one simulation and one stage."""

    x_label = "Time in stage (hours)" if x_units == "hours" else "Time in stage (s)"

    traces = [
        build_simulation_trace(
            sim,
            layer=layer.name,
            kind=kind,
            stage=stage,
            occurrence=occurrence,
            units=units,
            x_units=x_units,
            label=layer.name,
        )
        for layer in sim.structure.layers
    ]

    temp_C = stage_temperature_C(sim, stage, occurrence)
    plot_traces(
        ax,
        traces=traces,
        logx=logx,
        logy=logy,
        title=f"All layers during {stage} ({temp_C:.0f}C)",
        xlabel=x_label,
        ylabel=f"Concentration ({units})",
        legend=legend,
        x_units=x_units,
        xlim=xlim,
        min_x_hours=min_x_hours,
    )
    if legend:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("Layer")


def make_all_layers_over_phases_figure(
    sim: Simulation,
    *,
    stages: Sequence[str] | None = None,
    kind: str = "total",
    occurrence: int = 0,
    units: str = "cm^-3",
    x_units: str = "seconds",
    logx: bool = True,
    logy: bool = True,
    xlim: tuple[float, float] | None = None,
    min_x_hours: float | None = 1e-3,
):
    """Build one subplot per stage, each showing all layer concentrations."""

    import matplotlib.pyplot as plt

    if stages is None:
        ordered: list[str] = []
        for seg in sim.schedule.compile():
            if seg.stage not in ordered:
                ordered.append(seg.stage)
        stages = ordered

    if not stages:
        raise ValueError("no stages provided")

    fig, axes = plt.subplots(1, len(stages), figsize=(6.0 * len(stages), 4.8), constrained_layout=True)
    if len(stages) == 1:
        axes = np.asarray([axes])

    for i, stage in enumerate(stages):
        plot_all_layers_for_stage(
            axes[i],
            sim=sim,
            stage=stage,
            kind=kind,
            occurrence=occurrence,
            units=units,
            x_units=x_units,
            logx=logx,
            logy=logy,
            legend=True,
            xlim=xlim,
            min_x_hours=min_x_hours,
        )

    return fig, axes


def plot_peak_time_vs_stage_temperature(
    ax,
    *,
    campaign: Any,
    target_stage_temp_C: float,
    stage_for_peak: str = "annealing",
    stage_sort: str = "firing",
    layer: str = "C",
    kind: str = "trapped",
    tolerance_C: float = 1e-6,
    x_occurrence: int = 0,
    y_occurrence: int = 0,
    y_units: str = "hours",
    logy: bool = True,
) -> None:
    """Panel: peak-time scatter vs stage temperature for matched simulations."""

    xs: list[float] = []
    ys: list[float] = []

    for sim in campaign.simulations:
        try:
            y_stage_temp = stage_temperature_C(sim, stage_for_peak, y_occurrence)
        except Exception:
            continue
        if abs(y_stage_temp - float(target_stage_temp_C)) > float(tolerance_C):
            continue

        try:
            x_temp = stage_temperature_C(sim, stage_sort, x_occurrence)
            t_s, y = sim.series(
                layer=layer,
                kind=kind,
                stage=stage_for_peak,
                occurrence=y_occurrence,
                units="cm^-3",
            )
        except Exception:
            continue

        t = _to_array(t_s)
        values = _to_array(y)
        if t.size == 0:
            continue
        i_peak = int(np.argmax(values))
        t_peak_s = float(t[i_peak])
        xs.append(float(x_temp))
        ys.append(t_peak_s / 3600.0 if y_units == "hours" else t_peak_s)

    if not xs:
        raise ValueError("no peak-time points found")

    ax.scatter(xs, ys, marker="x", s=60, color="crimson")
    _style_axes(
        ax,
        logx=False,
        logy=logy,
        title=f"Peak time vs {stage_sort} temp | {stage_for_peak}={target_stage_temp_C:.0f}C",
        xlabel=f"{stage_sort} temperature (C)",
        ylabel=("Peak time (hours)" if y_units == "hours" else "Peak time (s)"),
    )


def plot_sweep_heatmap(
    ax,
    *,
    sweep_table: Any,
    x_key: str,
    y_key: str,
    value_key: str,
    agg: str = "mean",
    cmap: str = "viridis",
):
    """Panel: heatmap from a sweep table with x, y, and value columns.

    sweep_table can be a pandas DataFrame or any mapping-like object that
    supports column access with sweep_table[key].
    """

    x_vals = np.asarray(sweep_table[x_key], dtype=float)
    y_vals = np.asarray(sweep_table[y_key], dtype=float)
    z_vals = np.asarray(sweep_table[value_key], dtype=float)
    if not (x_vals.shape == y_vals.shape == z_vals.shape):
        raise ValueError("x, y, and value columns must have matching shape")

    x_unique = np.unique(x_vals)
    y_unique = np.unique(y_vals)
    z_grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)

    for i, yv in enumerate(y_unique):
        for j, xv in enumerate(x_unique):
            mask = (x_vals == xv) & (y_vals == yv)
            if not np.any(mask):
                continue
            subset = z_vals[mask]
            if agg == "mean":
                z_grid[i, j] = float(np.mean(subset))
            elif agg == "median":
                z_grid[i, j] = float(np.median(subset))
            elif agg == "min":
                z_grid[i, j] = float(np.min(subset))
            elif agg == "max":
                z_grid[i, j] = float(np.max(subset))
            else:
                raise ValueError(f"unsupported agg: {agg}")

    mesh = ax.imshow(
        z_grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
    )
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"{value_key} heatmap")
    ax.figure.colorbar(mesh, ax=ax, label=value_key)


__all__ = [
    "TraceData",
    "build_campaign_stage_sweep_traces",
    "build_simulation_trace",
    "filter_simulations_by_stage_temperature",
    "make_all_layers_over_phases_figure",
    "make_parity_figure",
    "plot_traces",
    "plot_abs_error",
    "plot_all_layers_for_stage",
    "plot_layer_stage_sweep",
    "plot_peak_time_vs_stage_temperature",
    "plot_rel_error",
    "plot_sweep_heatmap",
    "plot_trace_overlay",
    "stage_temperature_C",
]
