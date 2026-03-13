"""Fast, deterministic smoke tests for the basic CI lane.

Focuses on schedule compile behavior and query/slicing contracts without
requiring heavy live parity execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from hdiff.result import RunResult, SegmentBoundary
from hdiff.schedule import Schedule, Segment
from hdiff.sim import Sampling, Simulation, SolverConfig
from tests.test_helpers import make_min_structure


@pytest.mark.basic_ci
def test_single_stage_500c_10s_compile_and_bounds() -> None:
    """Verify one simple firing stage compiles into expected absolute bounds.

    This confirms schedule parsing/compilation produces exactly one segment
    from 0s to 10s for a 500C firing-only input.
    """
    schedule = Schedule(
        segments=[
            Segment(duration_s=10.0, stage="firing", T_K=500.0 + 273.15),
        ]
    )
    compiled = schedule.compile()

    assert len(compiled) == 1
    assert compiled[0].stage == "firing"
    assert compiled[0].t_start_s == 0.0
    assert compiled[0].t_end_s == 10.0


@pytest.mark.basic_ci
def test_single_stage_500c_10s_series_alignment_from_result() -> None:
    """Validate stage/layer query behavior using a handmade result (no solver run).

    The test injects a synthetic RunResult and checks that:
    - stage lookup returns correct bounds,
    - mobile and total queries share identical time axes,
    - total concentration is always >= mobile concentration.
    """
    structure = make_min_structure()
    schedule = Schedule(
        segments=[
            Segment(duration_s=10.0, stage="firing", T_K=500.0 + 273.15),
        ]
    )
    sampling = Sampling(
        base_out_dt_s=2.0,
        bootstrap_duration_s=0.5,
        bootstrap_max_dt_s=1e-3,
    )
    solver = SolverConfig(
        backend="petsc",
        rtol=1e-6,
        atol=1e-9,
        petsc_options={"ts_type": "beuler", "ts_adapt_type": "none"},
        max_steps=200000,
    )

    sim = Simulation(
        structure=structure,
        schedule=schedule,
        sampling=sampling,
        solver=solver,
    )
    n_state = sim.layout["n_state"]
    y = np.zeros((3, n_state), dtype=float)
    idx_mobile_a = sim.layout["idx_mobile"]["A"]
    idx_trapped_a = sim.layout["idx_trapped"][("A", "t1")]
    y[:, idx_mobile_a] = np.array([0.2, 0.4, 0.6])
    y[:, idx_trapped_a] = np.array([0.1, 0.1, 0.2])

    result = RunResult(
        cache_key="synthetic",
        schema_version=sim.schema_version,
        spec_json=sim.spec_json(),
        order_json="[]",
        t_s=np.array([0.0, 5.0, 10.0], dtype=float),
        y=y,
        boundaries=[
            SegmentBoundary(
                i_seg=0,
                i_start=0,
                i_end=3,
            )
        ],
    )
    sim.result = result

    assert len(result.boundaries) == 1
    assert sim.stage_occurrences("firing") == 1
    assert sim.stage_bounds("firing", 0) == (0.0, 10.0)

    t_mobile, y_mobile = sim.series(layer="A", kind="mobile", stage="firing", units="solver")
    t_total, y_total = sim.layer_total(layer="A", stage="firing", units="solver")

    t_mobile_arr = np.asarray(t_mobile, dtype=float)
    t_total_arr = np.asarray(t_total, dtype=float)
    y_mobile_arr = np.asarray(y_mobile, dtype=float)
    y_total_arr = np.asarray(y_total, dtype=float)

    assert t_mobile_arr.size > 1
    assert np.all(np.diff(t_mobile_arr) >= -1e-14)
    assert np.allclose(t_mobile_arr, t_total_arr)
    assert np.all(y_total_arr >= y_mobile_arr)


@pytest.mark.basic_ci
def test_runresult_stage_slice_helpers_rezero_and_absolute() -> None:
    """Ensure RunResult slicing helpers return correct rows and time references.

    Checks both modes:
    - rezeroed stage time (starts at 0),
    - absolute time (original global timeline retained).
    """
    result = RunResult(
        cache_key="unit",
        schema_version=1,
        spec_json="{}",
        order_json="[]",
        t_s=np.array([0.0, 2.0, 4.0, 7.0, 10.0], dtype=float),
        y=np.array(
            [
                [0.0, 10.0],
                [1.0, 11.0],
                [2.0, 12.0],
                [3.0, 13.0],
                [4.0, 14.0],
            ],
            dtype=float,
        ),
        boundaries=[
            SegmentBoundary(i_seg=0, i_start=0, i_end=3),
            SegmentBoundary(i_seg=1, i_start=2, i_end=5),
        ],
    )

    compiled = [
        Segment(duration_s=4.0, stage="firing", T_C=500.0),
        Segment(duration_s=6.0, stage="annealing", T_C=250.0),
    ]

    t_stage, y_stage = result.slice_for_stage(compiled, stage="annealing", occurrence=0, rezero_time=True)
    assert np.array_equal(t_stage, np.array([0.0, 3.0, 6.0]))
    assert np.array_equal(y_stage[:, 0], np.array([2.0, 3.0, 4.0]))

    t_abs, y_abs = result.slice_for_stage(compiled, stage="annealing", occurrence=0, rezero_time=False)
    assert np.array_equal(t_abs, np.array([4.0, 7.0, 10.0]))
    assert np.array_equal(y_abs[:, 1], np.array([12.0, 13.0, 14.0]))
