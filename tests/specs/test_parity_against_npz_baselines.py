"""Parity tests against frozen legacy-series NPZ baselines.

Designed for reproducible parity checks without requiring live legacy runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest
try:
    from tests.parity_framework import (
        LegacySeriesNpzBaselineSource,
        ParityCase,
        assert_case_thresholds,
        build_log_probe_grid,
        compare_traces,
        run_new_trace,
    )
    from tests.parity_cases import DEFAULT_CASES, DEFAULT_MEASURE
except ModuleNotFoundError:
    from parity_framework import (
        LegacySeriesNpzBaselineSource,
        ParityCase,
        assert_case_thresholds,
        build_log_probe_grid,
        compare_traces,
        run_new_trace,
    )
    from parity_cases import DEFAULT_CASES, DEFAULT_MEASURE


@pytest.mark.parity
@pytest.mark.parity_npz
@pytest.mark.slow
@pytest.mark.parametrize("case", DEFAULT_CASES, ids=[c.name for c in DEFAULT_CASES])
def test_parity_against_legacy_series_npz_case(
    case: ParityCase,
    parity_case_filter: set[str] | None,
    parity_probe_points: int,
) -> None:
    """Compare one golden case against frozen legacy NPZ baseline.

    This is the reproducible parity lane: no live legacy execution required.
    """
    if parity_case_filter is not None and case.name not in parity_case_filter:
        pytest.skip(f"case {case.name} not selected by --parity-case filter")

    baseline = LegacySeriesNpzBaselineSource(
        baselines_root=Path(__file__).resolve().parents[1] / "golden" / "legacy_series"
    )
    ref = baseline.get_trace(case, DEFAULT_MEASURE)
    sim = run_new_trace(case, DEFAULT_MEASURE)
    probe = build_log_probe_grid(ref.t_s, n_points=parity_probe_points)
    metrics = compare_traces(sim, ref, probe_t_s=probe)
    assert_case_thresholds(case, metrics)
