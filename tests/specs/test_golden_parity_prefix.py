"""Live legacy-vs-new parity tests for the standard two-stage golden cases."""

from __future__ import annotations

import os

import pytest
try:
    from tests.parity_framework import (
        LegacyRuntimeBaselineSource,
        ParityCase,
        assert_case_thresholds,
        build_log_probe_grid,
        compare_traces,
        run_new_trace,
    )
    from tests.parity_cases import DEFAULT_MEASURE, DEFAULT_CASES
except ModuleNotFoundError:
    from parity_framework import (
        LegacyRuntimeBaselineSource,
        ParityCase,
        assert_case_thresholds,
        build_log_probe_grid,
        compare_traces,
        run_new_trace,
    )
    from parity_cases import DEFAULT_MEASURE, DEFAULT_CASES


@pytest.mark.parity
@pytest.mark.parity_live
@pytest.mark.slow
@pytest.mark.parametrize("case", DEFAULT_CASES, ids=[c.name for c in DEFAULT_CASES])
def test_direct_legacy_array_parity_case(
    case: ParityCase,
    request: pytest.FixtureRequest,
    parity_case_filter: set[str] | None,
) -> None:
    """Compare one two-stage golden case directly against live legacy runtime.

    This test is intentionally opt-in because it is slower and depends on
    legacy runtime availability.
    """
    enabled = request.config.getoption("run_legacy_parity") or os.getenv("HDIFF_ENABLE_LEGACY_PARITY", "0") == "1"
    if not enabled:
        pytest.skip("use --run-legacy-parity or HDIFF_ENABLE_LEGACY_PARITY=1")
    if parity_case_filter is not None and case.name not in parity_case_filter:
        pytest.skip(f"case {case.name} not selected by --parity-case filter")

    baseline = LegacyRuntimeBaselineSource()
    ref = baseline.get_trace(case, DEFAULT_MEASURE)
    sim = run_new_trace(case, DEFAULT_MEASURE)
    probe = build_log_probe_grid(ref.t_s, n_points=2048)
    metrics = compare_traces(sim, ref, probe_t_s=probe)
    print(
        f"\n{case.name}: rel_l2={metrics.rel_l2:.6f} "
        f"(thr {case.threshold_rel_l2:.6f}) "
        f"rel_linf={metrics.rel_linf:.6f} "
        f"(thr {case.threshold_rel_linf:.6f})"
    )
    assert_case_thresholds(case, metrics)
