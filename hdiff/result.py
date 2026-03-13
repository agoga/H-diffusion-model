"""Post-simulation result containers and query helpers.

``RunResult`` stores the full ODE trajectory as parallel arrays ``t_s`` and
``y`` (shape ``[n_points, n_state]``). ``SegmentBoundary`` maps compiled
segment index to trajectory row ranges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
    i_start: int
    """First row index (inclusive) in ``RunResult.t_s`` / ``y``."""
    i_end: int
    """One-past-last row index (exclusive) in ``RunResult.t_s`` / ``y``."""


@dataclass
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

    cache_key: str
    schema_version: int
    spec_json: str
    order_json: str
    cache_dir: str | None = None
    t_s: np.ndarray | None = None
    y: np.ndarray | None = None
    boundaries: list[SegmentBoundary] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.t_s is not None and self.y is not None

    def set_data(
        self,
        *,
        t_s: np.ndarray,
        y: np.ndarray,
        boundaries: list[SegmentBoundary],
    ) -> None:
        self.t_s = np.asarray(t_s, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.boundaries = list(boundaries)

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        root = Path(self.cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{self.cache_key}.npz"

    def try_load(self) -> bool:
        if self.completed:
            return True

        path = self._cache_path()
        if path is None or not path.is_file():
            return False

        try:
            with np.load(path, allow_pickle=False) as data:
                completed = int(data["completed"].item())
                if completed != 1:
                    return False

                stored_spec_json = str(data["spec_json"].item())
                stored_cache_key = str(data["cache_key"].item())
                stored_order_json = str(data["order_json"].item())
                if stored_spec_json != self.spec_json:
                    return False
                if stored_cache_key != self.cache_key:
                    return False
                if stored_order_json != self.order_json:
                    return False

                boundaries_json = str(data["boundaries_json"].item())
                boundaries = boundaries_from_jsonable(json.loads(boundaries_json))
                self.set_data(
                    t_s=np.asarray(data["t_s"]),
                    y=np.asarray(data["y"]),
                    boundaries=boundaries,
                )
                return True
        except Exception:
            return False

    def save(self, *, timestep: float, reason: str = "completed", completed: int = 1) -> Path | None:
        if not self.completed:
            raise ValueError("cannot save an incomplete result")

        path = self._cache_path()
        if path is None:
            return None

        boundaries_json = json.dumps(
            boundaries_to_jsonable(self.boundaries),
            sort_keys=True,
            separators=(",", ":"),
        )
        np.savez_compressed(
            path,
            t_s=self.t_s,
            y=self.y,
            timestep=float(timestep),
            reason=str(reason),
            completed=int(completed),
            schema_version=int(self.schema_version),
            cache_key=str(self.cache_key),
            spec_json=str(self.spec_json),
            order_json=str(self.order_json),
            boundaries_json=boundaries_json,
        )
        return path

    def stage_occurrences(self, compiled: list[object], stage: str) -> int:
        return sum(
            1
            for boundary in self.boundaries
            if 0 <= boundary.i_seg < len(compiled) and getattr(compiled[boundary.i_seg], "stage", None) == stage
        )

    def boundary_for_stage(
        self,
        compiled: list[object],
        *,
        stage: str,
        occurrence: int = 0,
    ) -> SegmentBoundary:
        matches = [
            boundary
            for boundary in self.boundaries
            if 0 <= boundary.i_seg < len(compiled) and getattr(compiled[boundary.i_seg], "stage", None) == stage
        ]
        if occurrence < 0 or occurrence >= len(matches):
            raise ValueError(f"stage occurrence out of range for stage={stage}: {occurrence}")
        return matches[occurrence]

    def stage_bounds(
        self,
        compiled: list[object],
        *,
        stage: str,
        occurrence: int = 0,
    ) -> tuple[float, float]:
        boundary = self.boundary_for_stage(compiled, stage=stage, occurrence=occurrence)
        if boundary.i_seg < 0 or boundary.i_seg >= len(compiled):
            raise ValueError(f"compiled segment missing for i_seg={boundary.i_seg}")
        segment = compiled[boundary.i_seg]
        return float(getattr(segment, "t_start_s")), float(getattr(segment, "t_end_s"))

    def slice_by_boundary(
        self,
        boundary: SegmentBoundary,
        *,
        rezero_time: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.t_s is None or self.y is None:
            raise ValueError("result has no loaded trajectory data")

        i_start = int(boundary.i_start)
        i_end = int(boundary.i_end)
        if i_start < 0 or i_end > self.t_s.shape[0] or i_end <= i_start:
            raise ValueError("invalid boundary indices for result arrays")

        t = np.asarray(self.t_s[i_start:i_end], dtype=float)
        y = np.asarray(self.y[i_start:i_end, :], dtype=float)
        if rezero_time and t.size > 0:
            t = t - t[0]
        return t, y

    def slice_for_stage(
        self,
        compiled: list[object],
        *,
        stage: str,
        occurrence: int = 0,
        rezero_time: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        boundary = self.boundary_for_stage(compiled, stage=stage, occurrence=occurrence)
        return self.slice_by_boundary(boundary, rezero_time=rezero_time)

def boundaries_to_jsonable(boundaries: list[SegmentBoundary]) -> list[dict[str, int]]:
    return [
        {
            "i_seg": boundary.i_seg,
            "i_start": boundary.i_start,
            "i_end": boundary.i_end,
        }
        for boundary in boundaries
    ]


def boundaries_from_jsonable(items: list[dict[str, int | float | str]]) -> list[SegmentBoundary]:
    return [
        SegmentBoundary(
            i_seg=int(item["i_seg"]),
            i_start=int(item["i_start"]),
            i_end=int(item["i_end"]),
        )
        for item in items
    ]
