"""Schedule and sampling dataclasses, plus schedule builder/parser utilities.

A ``Schedule`` is a list of isothermal ``Segment`` objects, optionally repeated
``n_cycles`` times.  ``Schedule.compile()`` produces a flat list of
``CompiledSegment`` objects with absolute wall-clock times, which is what the
solver consumes.

A ``Sampling`` controls how densely the ODE integrator emits output points:

* ``bootstrap_duration_s`` + ``bootstrap_max_dt_s``: at the start of each
  segment the solver is constrained to small steps so rapid early transients
  are resolved before the step-size adaptor is allowed to coarsen.
* ``base_out_dt_s``: output grid spacing for the main (post-bootstrap) phase.
  This value is fed to the PETSc step monitor via interpolation; it does *not*
  directly constrain the internal solver step.

Schedule string format:
    ``"<duration><unit>:<temperature><unit>, ..."``, e.g.
    ``"600s:750C, 8000000s:250C"``.
    Duration units: s (default), m, h.  Temperature units: C (default), K.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Segment:
    """One isothermal step in a thermal schedule."""

    duration_s: float
    """Duration of this step in seconds (must be > 0)."""
    stage: str
    """Label used to retrieve results, e.g. ``'firing'``, ``'annealing'``."""
    T_K: float | None = None
    """Constant temperature in Kelvin.  Mutually exclusive with ``T_C``/``T_program``."""
    T_C: float | None = None
    """Constant temperature in Celsius.  Converted to ``T_K`` internally."""
    T_program: object | None = None
    """Reserved; temperature-ramp support is not implemented in v1."""

    def __post_init__(self) -> None:
        # Canonicalize Celsius input so downstream code only reads T_K.
        if self.T_K is not None and self.T_C is not None:
            raise ValueError("provide only one of T_K or T_C")
        if self.T_K is None and self.T_C is not None:
            object.__setattr__(self, "T_K", float(self.T_C) + 273.15)
            object.__setattr__(self, "T_C", None)
        self.validate()

    def validate(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("segment duration_s must be > 0")
        if not self.stage:
            raise ValueError("segment stage must be non-empty")
        if (self.T_K is None) == (self.T_program is None):
            raise ValueError("provide exactly one of T_K/T_C or T_program")
        if self.T_K is not None and self.T_K <= 0.0:
            raise ValueError("segment T_K must be > 0")


@dataclass(frozen=True)
class CompiledSegment:
    """An expanded segment with absolute wall-clock times.

    Produced by ``Schedule.compile()``; not constructed directly by callers.
    """

    i_seg: int
    """Zero-based segment index within the full compiled list."""
    stage: str
    stage_occurrence: int
    """How many prior occurrences of the same stage label preceded this one."""
    t_start_s: float
    """Absolute start time (seconds from simulation start)."""
    t_end_s: float
    """Absolute end time (seconds from simulation start)."""
    T_K: float | None
    T_program: object | None


@dataclass(frozen=True)
class Sampling:
    """Controls output density from the ODE integrator.

    The solver always runs with adaptive time-stepping.  These parameters only
    affect *which* time points are written to the ``RunResult`` arrays.
    """

    base_out_dt_s: float
    """Target output spacing (s) during the main phase of each segment.  The
    monitor interpolates to emit a point every this many seconds, independent
    of the solver's internal step size."""
    bootstrap_duration_s: float
    """Length of the high-cadence bootstrap window at the start of each
    segment (s).  Set to 0 to disable bootstrapping."""
    bootstrap_max_dt_s: float
    """Maximum internal solver step (and output spacing) during the bootstrap
    window (s).  Should be much smaller than ``base_out_dt_s``."""

    def validate(self) -> None:
        if self.base_out_dt_s <= 0.0:
            raise ValueError("base_out_dt_s must be > 0")
        if self.bootstrap_duration_s < 0.0:
            raise ValueError("bootstrap_duration_s must be >= 0")
        if self.bootstrap_max_dt_s <= 0.0:
            raise ValueError("bootstrap_max_dt_s must be > 0")


@dataclass(frozen=True)
class Schedule:
    """An ordered list of isothermal segments, optionally cycled."""

    segments: list[Segment]
    """Base segment list — repeated ``n_cycles`` times by ``compile()``."""
    n_cycles: int = 1

    def validate(self) -> None:
        if not self.segments:
            raise ValueError("schedule segments must be non-empty")
        if self.n_cycles <= 0:
            raise ValueError("n_cycles must be >= 1")
        for segment in self.segments:
            segment.validate()

    def compile(self) -> list[CompiledSegment]:
        """Expand ``n_cycles`` repetitions into a flat, time-stamped segment list."""
        self.validate()
        compiled: list[CompiledSegment] = []
        t_cursor = 0.0
        stage_counts: dict[str, int] = defaultdict(int)
        i_seg = 0
        for _ in range(self.n_cycles):
            for segment in self.segments:
                stage_occurrence = stage_counts[segment.stage]
                stage_counts[segment.stage] += 1
                t_start = t_cursor
                t_end = t_start + segment.duration_s
                compiled.append(
                    CompiledSegment(
                        i_seg=i_seg,
                        stage=segment.stage,
                        stage_occurrence=stage_occurrence,
                        t_start_s=t_start,
                        t_end_s=t_end,
                        T_K=segment.T_K,
                        T_program=segment.T_program,
                    )
                )
                i_seg += 1
                t_cursor = t_end
        return compiled


def compiled_segment_to_dict(segment: CompiledSegment) -> dict[str, Any]:
    return {
        "i_seg": segment.i_seg,
        "stage": segment.stage,
        "stage_occurrence": segment.stage_occurrence,
        "t_start_s": segment.t_start_s,
        "t_end_s": segment.t_end_s,
        "T_K": segment.T_K,
        "T_program": str(segment.T_program) if segment.T_program is not None else None,
    }
