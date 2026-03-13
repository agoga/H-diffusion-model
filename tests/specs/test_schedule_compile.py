"""Schedule compilation contract tests (ordering, occurrence, determinism)."""

from __future__ import annotations

from hdiff.schedule import Schedule, Segment


def test_compile_has_strictly_increasing_end_times() -> None:
    """Check compiled segment end-times always increase strictly.

    Prevents malformed compiled schedules where time could stall or regress.
    """
    schedule = Schedule(
        segments=[
            Segment(duration_s=10.0, stage="firing", T_K=943.15),
            Segment(duration_s=1000.0, stage="annealing", T_K=523.15),
        ],
        n_cycles=2,
    )
    compiled = schedule.compile()
    t_ends = [segment.t_end_s for segment in compiled]
    assert all(next_t > current_t for current_t, next_t in zip(t_ends, t_ends[1:]))


def test_compile_stage_occurrence_matches_sequence() -> None:
    """Check per-stage occurrence counters are assigned in encounter order.

    Example: firing segments should be numbered 0, 1, ... as they appear.
    """
    schedule = Schedule(
        segments=[
            Segment(duration_s=10.0, stage="firing", T_K=943.15),
            Segment(duration_s=20.0, stage="firing", T_K=950.0),
            Segment(duration_s=1000.0, stage="annealing", T_K=523.15),
        ],
        n_cycles=1,
    )
    compiled = schedule.compile()
    assert [seg.stage_occurrence for seg in compiled if seg.stage == "firing"] == [0, 1]
    assert [seg.stage_occurrence for seg in compiled if seg.stage == "annealing"] == [0]


def test_compile_is_deterministic() -> None:
    """Check compile() is deterministic for unchanged input schedule."""
    schedule = Schedule(
        segments=[
            Segment(duration_s=10.0, stage="firing", T_K=943.15),
            Segment(duration_s=1000.0, stage="annealing", T_K=523.15),
        ],
        n_cycles=3,
    )
    assert schedule.compile() == schedule.compile()


def test_segment_accepts_celsius_input() -> None:
    schedule = Schedule(
        segments=[
            Segment(duration_s=10.0, stage="firing", T_C=670.0),
            Segment(duration_s=1000.0, stage="annealing", T_C=250.0),
        ],
        n_cycles=1,
    )
    compiled = schedule.compile()
    assert compiled[0].T_K == 943.15
    assert compiled[1].T_K == 523.15


def test_segment_rejects_both_kelvin_and_celsius() -> None:
    try:
        Segment(duration_s=10.0, stage="firing", T_K=943.15, T_C=670.0)
    except ValueError as exc:
        assert "only one of T_K or T_C" in str(exc)
    else:
        raise AssertionError("expected ValueError when both T_K and T_C are provided")
