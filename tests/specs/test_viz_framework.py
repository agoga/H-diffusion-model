"""Unit checks for the composable hdiff plotting framework."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hdiff.campaign import (
    Campaign,
    filter_simulations_by_stage_temperature as campaign_filter_simulations_by_stage_temperature,
)
from hdiff.defaults import DEFAULT_SAMPLING, DEFAULT_SOLVER, DEFAULT_STRUCTURE
from hdiff.schedule import parse_temp_schedule_spec
from hdiff.sim import Simulation
from hdiff.viz import (
    build_campaign_stage_sweep_traces,
    build_simulation_trace,
    filter_simulations_by_stage_temperature,
    make_parity_figure,
    plot_abs_error,
    plot_layer_stage_sweep,
    plot_rel_error,
    plot_traces,
    plot_sweep_heatmap,
    plot_trace_overlay,
    stage_temperature_C,
)


def _demo_sim(spec: str) -> Simulation:
    return Simulation(
        structure=DEFAULT_STRUCTURE,
        schedule=parse_temp_schedule_spec(spec, stage_names=["firing", "annealing"]),
        sampling=DEFAULT_SAMPLING,
        solver=DEFAULT_SOLVER,
    )


def test_stage_temperature_and_filter_without_run() -> None:
    sim_650_250 = _demo_sim("10s:650C, 800s:250C")
    sim_750_250 = _demo_sim("10s:750C, 800s:250C")
    campaign = Campaign(
        structure=DEFAULT_STRUCTURE,
        temp_schedules=["10s:650C, 800s:250C", "10s:750C, 800s:250C"],
        stage_names=["firing", "annealing"],
        auto_run=False,
    )

    assert stage_temperature_C(sim_650_250, "firing") == 650.0
    matched = filter_simulations_by_stage_temperature(
        campaign,
        stage="firing",
        target_temp_C=650.0,
    )
    assert matched == [sim_650_250]

    matched_campaign = campaign_filter_simulations_by_stage_temperature(
        campaign,
        stage="firing",
        target_temp_C=650.0,
    )
    assert matched_campaign == [sim_650_250]


def test_viz_filter_accepts_campaign_object() -> None:
    mgr = Campaign(
        structure=DEFAULT_STRUCTURE,
        temp_schedules=["10s:650C, 800s:250C", "10s:750C, 800s:250C"],
        stage_names=["firing", "annealing"],
        auto_run=False,
    )

    matched = filter_simulations_by_stage_temperature(
        mgr,
        stage="firing",
        target_temp_C=650.0,
    )
    assert len(matched) == 1
    assert stage_temperature_C(matched[0], "firing") == 650.0


def test_composable_trace_primitives_build_and_draw() -> None:
    campaign = Campaign(
        structure=DEFAULT_STRUCTURE,
        temp_schedules=["10s:650C, 800s:250C"],
        stage_names=["firing", "annealing"],
        auto_run=False,
    )
    sim = campaign.simulations[0]

    traces, _ = build_campaign_stage_sweep_traces(
        campaign,
        match_stage="firing",
        target_temp_C=650.0,
        layer="C",
        kind="total",
        plot_stage="annealing",
        x_units="seconds",
    )
    assert len(traces) == 1

    one = build_simulation_trace(
        sim,
        layer="C",
        kind="total",
        stage="annealing",
        units="cm^-3",
        x_units="seconds",
        label="single",
    )
    assert one.label == "single"
    assert one.t_s.size > 0

    fig, ax = plt.subplots(figsize=(4, 3))
    plot_traces(
        ax,
        traces=[one],
        logx=True,
        logy=True,
        title="demo",
        xlabel="Time (s)",
        ylabel="C total (cm^-3)",
        legend=True,
    )
    assert len(ax.lines) == 1
    plt.close(fig)


def test_parity_panels_draw() -> None:
    t_ref = np.array([0.0, 1.0, 10.0, 100.0])
    y_ref = np.array([1.0, 2.0, 3.0, 2.5])
    t_sim = np.array([0.0, 0.8, 9.0, 100.0])
    y_sim = np.array([1.0, 1.9, 3.1, 2.4])

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    plot_trace_overlay(axs[0], t_ref_s=t_ref, y_ref=y_ref, t_sim_s=t_sim, y_sim=y_sim)
    plot_abs_error(axs[1], t_ref_s=t_ref, y_ref=y_ref, t_sim_s=t_sim, y_sim=y_sim)
    plot_rel_error(axs[2], t_ref_s=t_ref, y_ref=y_ref, t_sim_s=t_sim, y_sim=y_sim)
    assert len(axs[0].lines) == 2
    assert len(axs[1].lines) == 1
    assert len(axs[2].lines) == 1
    plt.close(fig)


def test_make_parity_figure_one_case() -> None:
    result = {
        "case": type("Case", (), {"name": "demo_case"})(),
        "t_ref": np.array([0.0, 1.0, 10.0]),
        "y_ref": np.array([1.0, 2.0, 3.0]),
        "t_new": np.array([0.0, 1.2, 10.0]),
        "y_new": np.array([1.0, 2.1, 2.9]),
    }
    fig, axes = make_parity_figure([result], n_probe_points=128)
    assert axes.shape == (1, 3)
    plt.close(fig)


def test_plot_sweep_heatmap_draws() -> None:
    table = {
        "fire_C": np.array([650, 650, 700, 700], dtype=float),
        "anneal_C": np.array([225, 250, 225, 250], dtype=float),
        "score": np.array([0.2, 0.15, 0.12, 0.1], dtype=float),
    }

    fig, ax = plt.subplots(figsize=(4, 3))
    plot_sweep_heatmap(
        ax,
        sweep_table=table,
        x_key="fire_C",
        y_key="anneal_C",
        value_key="score",
        agg="mean",
    )
    assert ax.images
    plt.close(fig)


def test_plot_layer_stage_sweep_requires_matches() -> None:
    campaign = Campaign(
        structure=DEFAULT_STRUCTURE,
        temp_schedules=["10s:650C, 800s:250C"],
        stage_names=["firing", "annealing"],
        auto_run=False,
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        plot_layer_stage_sweep(
            ax,
            campaign=campaign,
            match_stage="firing",
            target_temp_C=750.0,
        )
    except ValueError as exc:
        assert "no simulations found" in str(exc)
    else:
        raise AssertionError("expected ValueError when no simulations match")
    finally:
        plt.close(fig)
