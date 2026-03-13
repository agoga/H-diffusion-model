"""Post-simulation result containers and slice helpers.

``RunResult`` stores the full ODE trajectory as parallel arrays ``t_s`` and
``y`` (shape ``[n_points, n_state]``).  ``SegmentBoundary`` records map each
compiled schedule segment to a half-open row range ``[i_start, i_end)`` into
those arrays.

Slice helpers (``slice_result_for_stage``, ``boundary_for_stage``) provide
convenient access without requiring callers to manage index arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentBoundary:
    """Maps one compiled segment to its row range in ``RunResult.t_s`` / ``y``.

    The slice ``t_s[i_start:i_end]`` covers all output points emitted during
    this segment.  Boundaries are half-open: ``i_end`` points one past the
    last row belonging to this segment.
    """

    i_seg: int
    """Zero-based index matching ``CompiledSegment.i_seg``."""
    stage: str
    t_start_s: float
    """Absolute start time of this segment (s)."""
    t_end_s: float
    """Absolute end time of this segment (s)."""
    i_start: int
    """First row index (inclusive) in ``RunResult.t_s`` / ``y``."""
    i_end: int
    """One-past-last row index (exclusive) in ``RunResult.t_s`` / ``y``."""


@dataclass(frozen=True)
class RunResult:
    """Complete output of one simulation run.

    Attributes:
        t_s: 1-D array of output time points (s), length ``n_points``.
        y: 2-D array of state values, shape ``(n_points, n_state)``.
            Column ordering follows ``build_state_layout``: all trapped
            populations (A..E per layer) then all mobile concentrations
            (one per layer in stack order).  Values are in *solver units*;
            multiply by ``Structure.conc_scale`` to obtain cm^-3.
        boundaries: Segment boundary records, one per compiled segment.
        cache_key: SHA-256 hex digest of ``spec_json``.
        schema_version: Integer schema version from the originating simulation.
        spec_json: Canonical JSON spec used to produce this result.
    """

    t_s: np.ndarray
    y: np.ndarray
    boundaries: list[SegmentBoundary]
    cache_key: str
    schema_version: int
    spec_json: str


def boundary_for_stage(
    boundaries: list[SegmentBoundary],
    *,
    stage: str,
    occurrence: int = 0,
) -> SegmentBoundary:
    """Return the ``SegmentBoundary`` for the given stage label and occurrence index."""
    matches = [boundary for boundary in boundaries if boundary.stage == stage]
    if occurrence < 0 or occurrence >= len(matches):
        raise ValueError(f"stage occurrence out of range for stage={stage}: {occurrence}")
    return matches[occurrence]


def slice_result_by_boundary(
    result: RunResult,
    boundary: SegmentBoundary,
    *,
    rezero_time: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(t, y)`` arrays for one segment boundary.

    Args:
        result: Full simulation output.
        boundary: Segment boundary specifying the row range.
        rezero_time: If ``True`` (default), subtract ``t[0]`` so time
            starts at zero within the returned slice.

    Returns:
        ``(t, y)`` where ``t`` has shape ``(n,)`` and ``y`` has shape
        ``(n, n_state)``.
    """
    i_start = int(boundary.i_start)
    i_end = int(boundary.i_end)
    if i_start < 0 or i_end > result.t_s.shape[0] or i_end <= i_start:
        raise ValueError("invalid boundary indices for result arrays")

    t = np.asarray(result.t_s[i_start:i_end], dtype=float)
    y = np.asarray(result.y[i_start:i_end, :], dtype=float)
    if rezero_time and t.size > 0:
        t = t - t[0]
    return t, y


def slice_result_for_stage(
    result: RunResult,
    *,
    stage: str,
    occurrence: int = 0,
    rezero_time: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(t, y)`` for a named stage, re-zeroing time by default."""
    boundary = boundary_for_stage(result.boundaries, stage=stage, occurrence=occurrence)
    return slice_result_by_boundary(result, boundary, rezero_time=rezero_time)


def boundaries_to_jsonable(boundaries: list[SegmentBoundary]) -> list[dict[str, int | float | str]]:
    return [
        {
            "i_seg": boundary.i_seg,
            "stage": boundary.stage,
            "t_start_s": boundary.t_start_s,
            "t_end_s": boundary.t_end_s,
            "i_start": boundary.i_start,
            "i_end": boundary.i_end,
        }
        for boundary in boundaries
    ]


def boundaries_from_jsonable(items: list[dict[str, int | float | str]]) -> list[SegmentBoundary]:
    return [
        SegmentBoundary(
            i_seg=int(item["i_seg"]),
            stage=str(item["stage"]),
            t_start_s=float(item["t_start_s"]),
            t_end_s=float(item["t_end_s"]),
            i_start=int(item["i_start"]),
            i_end=int(item["i_end"]),
        )
        for item in items
    ]
