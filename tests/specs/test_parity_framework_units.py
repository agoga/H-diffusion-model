"""Unit tests for parity utility primitives (slicing, interpolation, metrics)."""

from __future__ import annotations

import json

import numpy as np
try:
    from tests.parity_framework import (
        ComparisonMetrics,
        LegacySeriesNpzBaselineSource,
        MeasureSpec,
        ParityCase,
        SeriesTrace,
        assert_case_thresholds,
        compare_traces,
        extract_series_from_state_matrix,
        interpolate_to_reference,
    )
except ModuleNotFoundError:
    from parity_framework import (
        ComparisonMetrics,
        LegacySeriesNpzBaselineSource,
        MeasureSpec,
        ParityCase,
        SeriesTrace,
        assert_case_thresholds,
        compare_traces,
        extract_series_from_state_matrix,
        interpolate_to_reference,
    )


def test_extract_series_from_state_matrix_stage_slice_and_units() -> None:
    """Verify stage slicing and concentration scaling in extracted traces."""
    t = np.array([0.0, 5.0, 10.0, 15.0, 20.0], dtype=float)
    y = np.zeros((5, 10), dtype=float)
    y[:, 2] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    measure = MeasureSpec(layer="C", kind="trapped", stage="annealing", units="cm^-3")
    t_out, y_out = extract_series_from_state_matrix(
        t_s=t,
        y_state=y,
        measure=measure,
        stage_durations_s=[10.0, 10.0],
        conc_scale=1e3,
    )

    assert np.array_equal(t_out, np.array([0.0, 5.0, 10.0]))
    assert np.array_equal(y_out, np.array([3_000.0, 4_000.0, 5_000.0]))


def test_interpolate_to_reference_requires_monotonic_time() -> None:
    """Ensure interpolation rejects non-monotonic time axes with clear error."""
    trace = SeriesTrace(
        t_s=np.array([0.0, 2.0, 1.0]),
        y=np.array([1.0, 2.0, 3.0]),
        source="unit",
        metadata={},
    )
    try:
        interpolate_to_reference(trace, np.array([0.5, 1.5]))
        assert False, "expected ValueError for non-monotonic time"
    except ValueError as exc:
        assert "monotonic" in str(exc)


def test_compare_traces_metrics_are_stable() -> None:
    """Verify comparison metrics return sane, non-negative values and sample count."""
    ref = SeriesTrace(
        t_s=np.array([0.0, 1.0, 2.0]),
        y=np.array([2.0, 2.0, 2.0]),
        source="ref",
        metadata={},
    )
    sim = SeriesTrace(
        t_s=np.array([0.0, 2.0]),
        y=np.array([1.0, 3.0]),
        source="sim",
        metadata={},
    )
    metrics = compare_traces(sim, ref)
    assert isinstance(metrics, ComparisonMetrics)
    assert metrics.n_samples == 3
    assert metrics.rel_l2 >= 0.0
    assert metrics.rel_linf >= 0.0
    assert metrics.max_abs >= 0.0


def test_assert_case_thresholds_checks_both_norms() -> None:
    """Ensure threshold assertions enforce both rel-L2 and rel-Linf limits."""
    case = ParityCase(
        name="case",
        fire_C=650.0,
        anneal_C=250.0,
        fire_s=10.0,
        anneal_s=100.0,
        threshold_rel_l2=0.10,
        threshold_rel_linf=0.20,
    )
    ok = ComparisonMetrics(rel_l2=0.05, rel_linf=0.10, max_abs=1.0, n_samples=2)
    assert_case_thresholds(case, ok)

    bad = ComparisonMetrics(rel_l2=0.12, rel_linf=0.10, max_abs=1.0, n_samples=2)
    try:
        assert_case_thresholds(case, bad)
        assert False, "expected assertion failure for rel_l2"
    except AssertionError as exc:
        assert "rel_l2" in str(exc)


def test_legacy_series_npz_source_loads_from_index(tmp_path) -> None:
    """Verify NPZ baseline source resolves index entry and loads arrays correctly."""
    root = tmp_path / "golden"
    root.mkdir(parents=True)
    data_path = root / "one_case.npz"
    np.savez_compressed(
        data_path,
        t_s=np.array([0.0, 1.0]),
        y=np.array([10.0, 11.0]),
        metadata_json=json.dumps({"source": "legacy"}),
    )
    (root / "index.json").write_text(json.dumps({"demo": "one_case.npz"}, indent=2))

    src = LegacySeriesNpzBaselineSource(baselines_root=root)
    case = ParityCase(name="demo", fire_C=650.0, anneal_C=250.0, fire_s=10.0, anneal_s=1_000.0)
    trace = src.get_trace(case, MeasureSpec(layer="C", kind="trapped"))

    assert trace.source == "legacy_series_npz"
    assert np.array_equal(trace.t_s, np.array([0.0, 1.0]))
    assert np.array_equal(trace.y, np.array([10.0, 11.0]))
