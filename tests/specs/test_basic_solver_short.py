"""Short-duration real solver and live parity checks.

Uses a minimal 500C single-stage scenario with explicit trap/detrap inputs to
validate end-to-end solver behavior quickly and compare against legacy runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

from hdiff.sim import Sampling, Simulation, SolverConfig

try:
    from tests.parity_cases import LAYER_NAMES
    from tests.parity_framework import (
        MeasureSpec,
        build_log_probe_grid,
        compare_traces,
        full_params,
        legacy_like_structure,
        run_legacy_runtime_trace,
        run_new_schedule_trace,
        schedule_single_stage,
    )
except ModuleNotFoundError:
    from parity_cases import LAYER_NAMES
    from parity_framework import (
        MeasureSpec,
        build_log_probe_grid,
        compare_traces,
        full_params,
        legacy_like_structure,
        run_legacy_runtime_trace,
        run_new_schedule_trace,
        schedule_single_stage,
    )


def basic_single_stage_params() -> dict[str, float]:
    """Return the baseline short-case parameters used across short solver/parity tests.

    Intended to be easy to reason about:
    - trap energy 0.5 for all layers,
    - detrap energy 2.0 for layer A, 1.0 for all other layers.
    """
    return {
        "A_trap": 0.5,
        "B_trap": 0.5,
        "C_trap": 0.5,
        "D_trap": 0.5,
        "A_detrap": 2.0,
        "B_detrap": 1.0,
        "C_detrap": 1.0,
        "D_detrap": 1.0,
    }


@pytest.mark.basic_ci
@pytest.mark.slow
def test_short_single_stage_500c_real_solver_runs_and_aligns_queries() -> None:
    """Run a real short 500C solve and validate core query contracts per layer.

    This catches integration/query regressions that synthetic-result tests cannot:
    stage boundaries, monotonic time output, and consistency of mobile vs total
    series returned by the Simulation query API.
    """
    base = basic_single_stage_params()
    full = full_params(base)
    structure = legacy_like_structure(full)
    schedule = schedule_single_stage(stage="firing", temp_C=500.0, duration_s=1.0)
    sampling = Sampling(base_out_dt_s=0.25, bootstrap_duration_s=0.02, bootstrap_max_dt_s=1e-4)
    solver = SolverConfig(
        backend="petsc",
        rtol=1e-5,
        atol=1e-10,
        petsc_options={
            "ts_type": "rosw",
            "ts_adapt_type": "basic",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ts_exact_final_time": "matchstep",
        },
        max_steps=200000,
    )

    sim = Simulation(
        structure=structure,
        schedule=schedule,
        sampling=sampling,
        solver=solver,
    )
    result = sim.run()

    assert len(result.boundaries) == 1
    assert sim.stage_occurrences("firing") == 1
    assert sim.stage_bounds("firing", 0) == (0.0, 1.0)

    for layer in LAYER_NAMES:
        t_mobile, y_mobile = sim.series(layer=layer, kind="mobile", stage="firing", units="solver")
        t_total, y_total = sim.layer_total(layer=layer, stage="firing", units="solver")

        t_mobile_arr = np.asarray(t_mobile, dtype=float)
        t_total_arr = np.asarray(t_total, dtype=float)
        y_mobile_arr = np.asarray(y_mobile, dtype=float)
        y_total_arr = np.asarray(y_total, dtype=float)

        assert t_mobile_arr.size > 1
        assert np.all(np.diff(t_mobile_arr) >= -1e-14)
        assert np.allclose(t_mobile_arr, t_total_arr)
        assert np.all(np.isfinite(y_total_arr))
        assert np.all(y_total_arr >= y_mobile_arr)


@pytest.mark.parity
@pytest.mark.parity_live
@pytest.mark.slow
@pytest.mark.parametrize("layer", LAYER_NAMES)
def test_short_single_stage_500c_live_legacy_parity_per_layer(
    layer: str,
    request: pytest.FixtureRequest,
    parity_probe_points: int,
) -> None:
    """Compare new solver output against live legacy runtime for one layer.

    Each parametrized instance isolates a single layer so failures immediately
    indicate which layer diverged for this short 500C scenario.
    """
    if not request.config.getoption("run_legacy_parity"):
        pytest.skip("use --run-legacy-parity for live legacy parity")

    base = basic_single_stage_params()
    schedule = schedule_single_stage(stage="firing", temp_C=500.0, duration_s=1.0)
    measure = MeasureSpec(layer=layer, kind="total", stage="firing", units="cm^-3")

    ref = run_legacy_runtime_trace(
        schedule_spec="1:500C",
        stage_names=["firing"],
        measure=measure,
        base_params=base,
    )
    sim = run_new_schedule_trace(
        schedule=schedule,
        measure=measure,
        base_params=base,
    )

    probe = build_log_probe_grid(ref.t_s, n_points=parity_probe_points)
    metrics = compare_traces(sim, ref, probe_t_s=probe)

    assert metrics.rel_l2 < 0.25, f"{layer} rel_l2={metrics.rel_l2:.6f} exceeds 0.250000"
    assert metrics.rel_linf < 0.40, f"{layer} rel_linf={metrics.rel_linf:.6f} exceeds 0.400000"
