"""Schedule dataclasses and schedule builder/parser utilities.

A ``Schedule`` is a list of isothermal ``Segment`` objects, optionally repeated
``n_cycles`` times.  ``Schedule.compile()`` produces a flat list of
``CompiledSegment`` objects with absolute wall-clock times, which is what the
solver consumes.

Sampling policy is intentionally owned by ``sim.py``.

Schedule string format:
    ``"<duration><unit>:<temperature><unit>, ..."``, e.g.
    ``"600s:750C, 8000000s:250C"``.
    Duration units: s (default), m, h.  Temperature units: C (default), K.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


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


_NUM_RE = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_DUR_RE = re.compile(rf"^\s*({_NUM_RE})\s*([smhSMH]?)\s*$")
_TEMP_RE = re.compile(rf"^\s*({_NUM_RE})\s*([kKcC]?)\s*$")


def _parse_duration_s(token: str) -> float:
    match = _DUR_RE.match(token)
    if not match:
        raise ValueError(f"invalid duration token: {token}")
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"", "s"}:
        return value
    if unit == "m":
        return value * 60.0
    if unit == "h":
        return value * 3600.0
    raise ValueError(f"invalid duration unit: {token}")


def _parse_temp_K(token: str) -> float:
    match = _TEMP_RE.match(token)
    if not match:
        raise ValueError(f"invalid temperature token: {token}")
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"", "k"}:
        return value
    if unit == "c":
        return value + 273.15
    raise ValueError(f"invalid temperature unit: {token}")


def parse_temp_schedule_spec(spec: str, stage_names: list[str] | None = None) -> Schedule:
    """Parse a schedule string into a ``Schedule``.

    Format: comma-separated ``<duration><unit>:<temperature><unit>`` pairs, e.g.
    ``"600s:750C, 8000000s:250C"``.  Duration units: ``s`` (default), ``m``,
    ``h``.  Temperature units: ``C`` (default), ``K``.
    """
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise ValueError("schedule spec must include at least one segment")

    segments: list[Segment] = []
    for index, part in enumerate(parts):
        if ":" not in part:
            raise ValueError(f"invalid schedule segment (missing ':'): {part}")
        duration_token, temp_token = [piece.strip() for piece in part.split(":", 1)]
        stage = (
            stage_names[index]
            if stage_names is not None and index < len(stage_names)
            else f"stage_{index}"
        )
        segments.append(
            Segment(
                duration_s=_parse_duration_s(duration_token),
                stage=stage,
                T_K=_parse_temp_K(temp_token),
            )
        )

    if stage_names is not None and len(stage_names) != len(segments):
        raise ValueError(
            f"stage_names length ({len(stage_names)}) must equal schedule segments ({len(segments)})"
        )
    return Schedule(segments=segments)


def build_temp_schedule_spec(
    *,
    fire_C: int | float,
    anneal_C: int | float,
    fire_s: int | float,
    anneal_s: int | float,
    include_room: bool = False,
    room_C: int | float = 0,
    room_s: int | float = 0,
    n_cycles: int = 1,
) -> str:
    """Build a schedule spec string for fire/anneal (optionally with room dwell)."""
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")

    cycle_parts: list[str] = [f"{float(fire_s):g}:{float(fire_C):g}C"]
    if include_room and float(room_s) > 0.0:
        cycle_parts.append(f"{float(room_s):g}:{float(room_C):g}C")
    cycle_parts.append(f"{float(anneal_s):g}:{float(anneal_C):g}C")
    return ", ".join(cycle_parts * int(n_cycles))


def make_unfired_sweep(
    anneal_temp: int | float,
    *,
    fire_temp: int | float = 25,
    fire_s: int | float,
    anneal_s: int | float,
) -> tuple[list[str], list[str]]:
    schedules = [
        build_temp_schedule_spec(
            fire_C=float(fire_temp),
            anneal_C=float(anneal_temp),
            fire_s=float(fire_s),
            anneal_s=float(anneal_s),
            include_room=False,
        )
    ]
    return schedules, ["firing", "annealing"]


def make_annealing_sweep(
    anneal_temps: Iterable[int | float],
    *,
    fire_temp: int | float,
    fire_s: int | float,
    anneal_s: int | float,
    include_room: bool,
    room_temp: int | float,
    room_s: int | float,
    n_cycles: int = 1,
) -> tuple[list[str], list[str]]:
    schedules = [
        build_temp_schedule_spec(
            fire_C=float(fire_temp),
            anneal_C=float(anneal_temp),
            fire_s=float(fire_s),
            anneal_s=float(anneal_s),
            include_room=include_room,
            room_C=float(room_temp),
            room_s=float(room_s),
            n_cycles=n_cycles,
        )
        for anneal_temp in anneal_temps
    ]
    stage_names = ["firing", "room", "annealing"] if include_room else ["firing", "annealing"]
    return schedules, stage_names


def make_firing_sweep(
    firing_temps: Iterable[int | float],
    *,
    anneal_temp: int | float,
    fire_s: int | float,
    anneal_s: int | float,
    include_room: bool,
    room_temp: int | float,
    room_s: int | float,
    n_cycles: int = 1,
) -> tuple[list[str], list[str]]:
    schedules = [
        build_temp_schedule_spec(
            fire_C=float(firing_temp),
            anneal_C=float(anneal_temp),
            fire_s=float(fire_s),
            anneal_s=float(anneal_s),
            include_room=include_room,
            room_C=float(room_temp),
            room_s=float(room_s),
            n_cycles=n_cycles,
        )
        for firing_temp in firing_temps
    ]
    stage_names = ["firing", "room", "annealing"] if include_room else ["firing", "annealing"]
    return schedules, stage_names


__all__ = [
    "CompiledSegment",
    "Schedule",
    "Segment",
    "build_temp_schedule_spec",
    "compiled_segment_to_dict",
    "make_annealing_sweep",
    "make_firing_sweep",
    "make_unfired_sweep",
    "parse_temp_schedule_spec",
]
