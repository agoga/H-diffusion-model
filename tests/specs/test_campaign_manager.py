"""Tests for hdiff campaign manager orchestration.

These validate migration-facing behavior: schedule-string parsing, duplicate
schedule compression by temperature sequence, and stage-aware result queries.
"""

from __future__ import annotations

import pytest

from hdiff.campaign import (
    Campaign,
)
from hdiff.defaults import DEFAULT_STRUCTURE
from hdiff.schedule import (
    make_annealing_sweep,
    make_firing_sweep,
    make_unfired_sweep,
    parse_temp_schedule_spec,
)
from hdiff.sim import Sampling, SolverConfig


_LEGACY_PARAM_BASE = {
    "A_trap": 0.9,
    "B_trap": 0.5,
    "A_trap_attemptfreq": 1e12,
}
_ATTEMPT_KEYS = {"A_trap_attemptfreq"}


def _symmetric_offsets(max_delta: float, n: int) -> list[float]:
    if n <= 0 or max_delta <= 0.0:
        return []
    if n == 1:
        return [float(max_delta)]

    step = (2.0 * max_delta) / float(n)
    offsets: list[float] = []
    for index in range(n + 1):
        value = -max_delta + step * float(index)
        if abs(value) > 1e-15:
            offsets.append(float(value))
    return offsets


def make_default_param_sweep(
    max_diff: dict[str, float],
    num_diff: int,
    *,
    include_default: bool = True,
) -> list[dict[str, float]]:
    base = dict(_LEGACY_PARAM_BASE)
    out: list[dict[str, float]] = []

    if include_default:
        out.append(dict(base))

    for key, delta in max_diff.items():
        if key not in base:
            continue
        if delta is None or float(delta) <= 0.0:
            continue

        for offset in _symmetric_offsets(float(delta), int(num_diff)):
            trial = dict(base)
            new_value = float(trial[key]) + float(offset)
            if key in _ATTEMPT_KEYS and new_value <= 0.0:
                continue
            trial[key] = new_value
            out.append(trial)

    return out


@pytest.mark.basic_ci
def test_parse_temp_schedule_spec_with_units_and_stage_names() -> None:
    """Verify schedule string parsing supports units and explicit stage naming."""
    schedule = parse_temp_schedule_spec("1s:500C, 0.5m:523.15K", ["firing", "annealing"])
    compiled = schedule.compile()

    assert len(compiled) == 2
    assert compiled[0].stage == "firing"
    assert compiled[1].stage == "annealing"
    assert compiled[0].t_end_s == pytest.approx(1.0)
    assert compiled[1].t_end_s == pytest.approx(31.0)
    assert compiled[0].T_K == pytest.approx(773.15)
    assert compiled[1].T_K == pytest.approx(523.15)


@pytest.mark.basic_ci
@pytest.mark.slow
def test_manager_compresses_temp_dupes_and_runs_stage_queries() -> None:
    """Verify manager keeps longest duplicate temp sequence and returns usable simulations."""
    solver = SolverConfig(
        backend="petsc",
        rtol=1e-6,
        atol=1e-9,
        petsc_options={"ts_type": "beuler", "ts_adapt_type": "none"},
        max_steps=200000,
    )
    sampling = Sampling(base_out_dt_s=0.01, bootstrap_duration_s=0.005, bootstrap_max_dt_s=1e-4)

    mgr = Campaign(
        structure=DEFAULT_STRUCTURE,
        temp_schedules=[
            "0.03:500C, 0.05:250C",
            "0.01:500C, 0.02:250C",
        ],
        stage_names=["firing", "annealing"],
        sampling=sampling,
        solver=solver,
        auto_run=True,
        compress_temp_dupes=True,
    )

    sims = mgr.simulations
    assert len(sims) == 1

    sim = sims[0]
    assert sim.stage_occurrences("firing") == 1
    assert sim.stage_occurrences("annealing") == 1
    assert sim.stage_bounds("firing", 0) == pytest.approx((0.0, 0.03))
    assert sim.stage_bounds("annealing", 0) == pytest.approx((0.03, 0.08))

    t_vals, y_vals = sim.layer_total(layer="C", stage="annealing", units="cm^-3")
    assert len(t_vals) == len(y_vals)
    assert len(t_vals) > 0


@pytest.mark.basic_ci
def test_sweep_builders_return_legacy_shape_and_stage_names() -> None:
    """Verify sweep builders return (schedule_strings, stage_names) like legacy sweeps."""
    anneal_schedules, anneal_stage_names = make_annealing_sweep(
        [250, 300],
        fire_temp=700,
        fire_s=10,
        anneal_s=20,
        include_room=True,
        room_temp=27,
        room_s=5,
        n_cycles=1,
    )
    assert len(anneal_schedules) == 2
    assert anneal_stage_names == ["firing", "room", "annealing"]

    firing_schedules, firing_stage_names = make_firing_sweep(
        [650, 700],
        anneal_temp=250,
        fire_s=10,
        anneal_s=20,
        include_room=False,
        room_temp=27,
        room_s=5,
        n_cycles=1,
    )
    assert len(firing_schedules) == 2
    assert firing_stage_names == ["firing", "annealing"]

    unfired_schedules, unfired_stage_names = make_unfired_sweep(350, fire_s=10, anneal_s=20)
    assert len(unfired_schedules) == 1
    assert unfired_stage_names == ["firing", "annealing"]


@pytest.mark.basic_ci
@pytest.mark.slow
def test_manager_runs_with_sweep_builder_output() -> None:
    """Verify a manager can run directly from sweep builder output and named stage queries."""
    schedules, stage_names = make_annealing_sweep(
        [250, 300],
        fire_temp=500,
        fire_s=0.01,
        anneal_s=0.02,
        include_room=True,
        room_temp=27,
        room_s=0.005,
    )

    solver = SolverConfig(
        backend="petsc",
        rtol=1e-6,
        atol=1e-9,
        petsc_options={"ts_type": "beuler", "ts_adapt_type": "none"},
        max_steps=200000,
    )
    sampling = Sampling(base_out_dt_s=0.01, bootstrap_duration_s=0.005, bootstrap_max_dt_s=1e-4)

    mgr = Campaign(
        structure=DEFAULT_STRUCTURE,
        temp_schedules=schedules,
        stage_names=stage_names,
        sampling=sampling,
        solver=solver,
        auto_run=True,
        compress_temp_dupes=False,
    )

    sims = mgr.simulations
    assert len(sims) == 2

    sim0 = sims[0]
    assert sim0.stage_occurrences("firing") == 1
    assert sim0.stage_occurrences("room") == 1
    assert sim0.stage_occurrences("annealing") == 1
    assert sim0.stage_bounds("firing", 0) == pytest.approx((0.0, 0.01))
    assert sim0.stage_bounds("room", 0) == pytest.approx((0.01, 0.015))
    assert sim0.stage_bounds("annealing", 0) == pytest.approx((0.015, 0.035))

    t_vals, y_vals = sim0.layer_total(layer="B", stage="annealing", units="cm^-3")
    assert len(t_vals) == len(y_vals)
    assert len(t_vals) > 0


@pytest.mark.basic_ci
def test_make_default_param_sweep_one_at_a_time_and_default_first() -> None:
    """Verify legacy-style additive one-at-a-time parameter sweep behavior."""
    sweep = make_default_param_sweep(
        {
            "A_trap": 0.2,
            "B_trap": 0.1,
        },
        num_diff=2,
        include_default=True,
    )

    assert len(sweep) == 5
    first = sweep[0]
    assert first["A_trap"] == pytest.approx(0.9)
    assert first["B_trap"] == pytest.approx(0.5)

    a_variants = [row for row in sweep[1:] if row["B_trap"] == pytest.approx(0.5)]
    b_variants = [row for row in sweep[1:] if row["A_trap"] == pytest.approx(0.9)]
    assert sorted(item["A_trap"] for item in a_variants) == pytest.approx([0.7, 1.1])
    assert sorted(item["B_trap"] for item in b_variants) == pytest.approx([0.4, 0.6])


@pytest.mark.basic_ci
def test_make_default_param_sweep_skips_nonpositive_attempt_frequency() -> None:
    """Verify attempt-frequency variants are clipped when additive offsets go non-positive."""
    sweep = make_default_param_sweep(
        {
            "A_trap_attemptfreq": 2e12,
        },
        num_diff=2,
        include_default=False,
    )

    assert len(sweep) == 1
    assert sweep[0]["A_trap_attemptfreq"] == pytest.approx(3e12)
