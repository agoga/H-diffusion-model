"""Live single-stage parity checks by layer.

Breaks parity into small per-layer/per-case chunks to isolate divergence and
reduce crash-prone monolithic runs.
"""

from __future__ import annotations

import pytest

try:
    from tests.parity_cases import LAYER_NAMES, SINGLE_STAGE_ANNEAL_CASES
    from tests.parity_framework import (
        MeasureSpec,
        build_log_probe_grid,
        compare_traces,
        run_legacy_runtime_trace,
        run_new_schedule_trace,
        schedule_single_stage,
    )
except ModuleNotFoundError:
    from parity_cases import LAYER_NAMES, SINGLE_STAGE_ANNEAL_CASES
    from parity_framework import (
        MeasureSpec,
        build_log_probe_grid,
        compare_traces,
        run_legacy_runtime_trace,
        run_new_schedule_trace,
        schedule_single_stage,
    )


@pytest.mark.parity
@pytest.mark.parity_live
@pytest.mark.slow
@pytest.mark.parametrize(
    "case_id,temp_C,duration_s",
    SINGLE_STAGE_ANNEAL_CASES,
    ids=[item[0] for item in SINGLE_STAGE_ANNEAL_CASES],
)
@pytest.mark.parametrize("layer", LAYER_NAMES)
def test_single_stage_anneal_layer_total_parity(
    case_id: str,
    temp_C: float,
    duration_s: float,
    layer: str,
    request: pytest.FixtureRequest,
    parity_case_filter: set[str] | None,
    parity_probe_points: int,
) -> None:
    """Compare one single-stage anneal case for one layer (total concentration).

    Parametrization over (case, layer) keeps failures highly local so the
    failing layer/temperature is immediately visible in test output.
    """
    if parity_case_filter is not None and case_id not in parity_case_filter:
        pytest.skip(f"case {case_id} not selected by --parity-case filter")

    if not request.config.getoption("run_legacy_parity"):
        pytest.skip("use --run-legacy-parity for live single-stage parity checks")

    schedule_spec = f"{int(duration_s)}:{int(temp_C)}C"
    measure = MeasureSpec(layer=layer, kind="total", stage="annealing", units="cm^-3")

    ref = run_legacy_runtime_trace(
        schedule_spec=schedule_spec,
        stage_names=["annealing"],
        measure=measure,
        base_params={},
    )

    sim = run_new_schedule_trace(
        schedule=schedule_single_stage(stage="annealing", temp_C=temp_C, duration_s=duration_s),
        measure=measure,
        base_params={},
    )

    probe = build_log_probe_grid(ref.t_s, n_points=parity_probe_points)
    metrics = compare_traces(sim, ref, probe_t_s=probe)

    assert metrics.rel_l2 < 0.60, (
        f"{case_id}/{layer} rel_l2={metrics.rel_l2:.6f} exceeds 0.600000"
    )
    assert metrics.rel_linf < 0.75, (
        f"{case_id}/{layer} rel_linf={metrics.rel_linf:.6f} exceeds 0.750000"
    )
