"""Orchestration utilities: schedule builders and sweep runners.

Public surface
--------------
* ``parse_temp_schedule_spec`` / ``build_temp_schedule_spec`` — convert between
    the schedule string format and ``Schedule`` objects.
* ``make_annealing_sweep`` / ``make_firing_sweep`` — generate lists of schedule
    strings for temperature sweeps.
* ``Campaign`` — high-level manager for a named set of schedules with
        a fixed structure; provides stage-indexed series access.
    Use ``Campaign.sweep_annealing(...)`` / ``Campaign.sweep_firing(...)`` /
    ``Campaign.sweep_unfired(...)`` to create and run sweeps directly.
* ``run_many`` / ``product_sweep`` — lower-level helpers kept for explicit
    custom sweeps (for example structure-parameter grid runs).
"""

from __future__ import annotations

import re
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable

from .cache import CacheStore
from .defaults import DEFAULT_SAMPLING, DEFAULT_SOLVER, DEFAULT_STRUCTURE
from .schedule import Sampling, Schedule, Segment
from .sim import Simulation, SolverConfig
from .structure import Structure

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

    Args:
        spec: Schedule specification string.
        stage_names: Optional list of stage labels.  Length must equal the
            number of segments if provided; defaults to ``stage_0``, ``stage_1``.

    Returns:
        A ``Schedule`` with ``n_cycles=1``.
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
    """Build a schedule string for a fire-then-anneal (optionally cycled) sequence.

    Returns a string in the format accepted by ``parse_temp_schedule_spec``.
    If ``include_room=True`` and ``room_s > 0``, a room-temperature dwell is
    inserted between firing and annealing.
    """
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
    """Build schedule strings for a sweep over annealing temperatures.

    Returns ``(schedules, stage_names)`` where ``schedules`` is a list of
    spec strings (one per annealing temperature) and ``stage_names`` is the
    corresponding list of stage labels.
    """
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
    """Build schedule strings for a sweep over firing temperatures.

    Returns ``(schedules, stage_names)`` — same shape as ``make_annealing_sweep``.
    """
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


def default_sampling() -> Sampling:
    return DEFAULT_SAMPLING


def default_solver() -> SolverConfig:
    return DEFAULT_SOLVER


def run_many(
    structure: Structure,
    schedules: dict[str, Schedule],
    sampling: Sampling,
    solver: SolverConfig,
    y0: list[float] | None = None,
    cache: Any | None = None,
    schema_version: int = 1,
) -> dict[str, Simulation]:
    """Run (or cache-load) one ``Simulation`` per schedule.

    Args:
        structure: Shared physical structure for all schedules.
        schedules: Mapping of schedule-id → ``Schedule``.
        sampling: Shared sampling settings.
        solver: Shared solver settings.
        y0: Optional initial state; defaults to all-zeros.
        cache: Optional ``CacheStore``; if ``None`` results are not persisted.
        schema_version: Written into each ``RunResult``.

    Returns:
        Dict of schedule-id → completed ``Simulation`` (with ``result`` set).
    """
    out: dict[str, Simulation] = {}
    for schedule_id, schedule in schedules.items():
        sim = Simulation(
            structure=structure,
            schedule=schedule,
            sampling=sampling,
            solver=solver,
            y0=y0,
            schema_version=schema_version,
        )
        out[schedule_id] = sim
        if cache is not None:
            sim.run(cache=cache)
    return out


def product_sweep(
    base_structure: Structure,
    axes: dict[str, list[Any]],
    apply: Callable[[Structure, dict[str, Any]], Structure],
    schedules: dict[str, Schedule],
    sampling: Sampling,
    solver: SolverConfig,
    y0: list[float] | None = None,
    cache: Any | None = None,
    schema_version: int = 1,
) -> dict[tuple[str, str], Simulation]:
    """Run a full Cartesian product of parameter axes × schedules.

    ``apply(base_structure, point)`` must return a *new* ``Structure`` (it may
    not return ``base_structure`` unchanged — this is validated).

    Returns a dict keyed by ``(sweep_id, schedule_id)`` where ``sweep_id`` is a
    human-readable string like ``"Ea_A=2.5,Ea_B=1.3"``.
    """
    axis_names = list(axes.keys())
    axis_values = [axes[name] for name in axis_names]
    out: dict[tuple[str, str], Simulation] = {}

    for values in product(*axis_values):
        point = dict(zip(axis_names, values))
        sweep_id = ",".join(f"{name}={point[name]}" for name in axis_names)
        structure = apply(base_structure, point)
        if structure is base_structure:
            raise ValueError("apply must return a modified Structure copy")

        sims = run_many(
            structure=structure,
            schedules=schedules,
            sampling=sampling,
            solver=solver,
            y0=y0,
            cache=cache,
            schema_version=schema_version,
        )
        for schedule_id, sim in sims.items():
            out[(sweep_id, schedule_id)] = sim
    return out


def stage_temperature_C(sim: Simulation, stage: str, occurrence: int = 0) -> float:
    """Return one stage temperature (C) from a simulation schedule."""

    compiled = sim.schedule.compile()
    segs = [seg for seg in compiled if seg.stage == stage]
    if occurrence < 0 or occurrence >= len(segs):
        raise ValueError(f"stage occurrence out of range for stage={stage}: {occurrence}")
    seg = segs[occurrence]
    if seg.T_K is None:
        raise ValueError(f"stage={stage} has no isothermal temperature")
    return float(seg.T_K - 273.15)


def filter_simulations_by_stage_temperature(
    campaign: "Campaign",
    *,
    stage: str,
    target_temp_C: float,
    tolerance_C: float = 1e-6,
    occurrence: int = 0,
) -> list[Simulation]:
    """Select simulations where one stage temperature matches target within tolerance."""

    matched: list[Simulation] = []
    for sim in campaign.simulations:
        try:
            temp_C = stage_temperature_C(sim, stage, occurrence)
        except Exception:
            continue
        if abs(temp_C - float(target_temp_C)) <= float(tolerance_C):
            matched.append(sim)
    return matched


def _compress_temp_dupes(
    schedules: list[tuple[str, Schedule]],
) -> list[tuple[str, Schedule]]:
    by_temp_seq: dict[tuple[float, ...], tuple[str, Schedule, float]] = {}
    for schedule_id, schedule in schedules:
        compiled = schedule.compile()
        key = tuple(float(segment.T_K) for segment in compiled if segment.T_K is not None)
        duration = float(sum(segment.duration_s for segment in schedule.segments))
        keep = by_temp_seq.get(key)
        if keep is None or duration > keep[2]:
            by_temp_seq[key] = (schedule_id, schedule, duration)
    return [(schedule_id, schedule) for schedule_id, schedule, _ in by_temp_seq.values()]


class Campaign:
    """Configured runner for schedule sweeps over one structure/solver setup."""

    def __init__(
        self,
        *,
        structure: Structure | None = None,
        temp_schedules: Iterable[str] | None = None,
        stage_names: list[str] | None = None,
        results_dir: str | None = None,
        sampling: Sampling | None = None,
        solver: SolverConfig | None = None,
        y0: list[float] | None = None,
        schema_version: int = 1,
        auto_run: bool = True,
        compress_temp_dupes: bool = True,
    ) -> None:
        self.stage_names = list(stage_names) if stage_names is not None else None
        self._sampling = default_sampling() if sampling is None else sampling
        self._solver = default_solver() if solver is None else solver
        self._cache = CacheStore(Path(results_dir)) if results_dir is not None else None
        self._y0 = y0
        self._schema_version = schema_version
        self._structure = DEFAULT_STRUCTURE if structure is None else structure
        self._simulations: list[Simulation] = []
        self._schedules: list[str] = []

        if temp_schedules is not None:
            self.add_schedule_specs(
                temp_schedules,
                stage_names=self.stage_names,
                compress_temp_dupes=compress_temp_dupes,
                run=auto_run,
            )

    def add_schedule_specs(
        self,
        temp_schedules: Iterable[str],
        *,
        stage_names: list[str] | None = None,
        compress_temp_dupes: bool = True,
        run: bool = True,
        id_prefix: str = "sched",
    ) -> list[Simulation]:
        """Parse schedule specs and append simulations to this campaign."""

        names = self.stage_names if stage_names is None else list(stage_names)
        start_index = len(self._schedules)
        parsed: list[tuple[str, Schedule]] = []
        for offset, spec in enumerate(temp_schedules):
            schedule_id = f"{id_prefix}_{start_index + offset}"
            parsed.append((schedule_id, parse_temp_schedule_spec(str(spec), names)))
        if compress_temp_dupes:
            parsed = _compress_temp_dupes(parsed)

        added: list[Simulation] = []
        for schedule_id, schedule in parsed:
            sim = Simulation(
                structure=self._structure,
                schedule=schedule,
                sampling=self._sampling,
                solver=self._solver,
                y0=self._y0,
                schema_version=self._schema_version,
            )
            self._schedules.append(schedule_id)
            self._simulations.append(sim)
            added.append(sim)

        if run:
            for sim in added:
                sim.run(cache=self._cache)
        return list(added)

    def sweep_annealing(
        self,
        anneal_temps: Iterable[int | float],
        *,
        fire_temp: int | float,
        fire_s: int | float,
        anneal_s: int | float,
        include_room: bool = False,
        room_temp: int | float = 25,
        room_s: int | float = 0,
        n_cycles: int = 1,
        run: bool = True,
        compress_temp_dupes: bool = False,
        id_prefix: str = "anneal",
    ) -> list[Simulation]:
        """Create and optionally run an annealing-temperature sweep."""

        specs, names = make_annealing_sweep(
            anneal_temps,
            fire_temp=fire_temp,
            fire_s=fire_s,
            anneal_s=anneal_s,
            include_room=include_room,
            room_temp=room_temp,
            room_s=room_s,
            n_cycles=n_cycles,
        )
        return self.add_schedule_specs(
            specs,
            stage_names=names,
            compress_temp_dupes=compress_temp_dupes,
            run=run,
            id_prefix=id_prefix,
        )

    def sweep_firing(
        self,
        firing_temps: Iterable[int | float],
        *,
        anneal_temp: int | float,
        fire_s: int | float,
        anneal_s: int | float,
        include_room: bool = False,
        room_temp: int | float = 25,
        room_s: int | float = 0,
        n_cycles: int = 1,
        run: bool = True,
        compress_temp_dupes: bool = False,
        id_prefix: str = "firing",
    ) -> list[Simulation]:
        """Create and optionally run a firing-temperature sweep."""

        specs, names = make_firing_sweep(
            firing_temps,
            anneal_temp=anneal_temp,
            fire_s=fire_s,
            anneal_s=anneal_s,
            include_room=include_room,
            room_temp=room_temp,
            room_s=room_s,
            n_cycles=n_cycles,
        )
        return self.add_schedule_specs(
            specs,
            stage_names=names,
            compress_temp_dupes=compress_temp_dupes,
            run=run,
            id_prefix=id_prefix,
        )

    def sweep_unfired(
        self,
        anneal_temp: int | float,
        *,
        fire_temp: int | float = 25,
        fire_s: int | float,
        anneal_s: int | float,
        run: bool = True,
        compress_temp_dupes: bool = False,
        id_prefix: str = "unfired",
    ) -> list[Simulation]:
        """Create and optionally run a single unfired schedule."""

        specs, names = make_unfired_sweep(
            anneal_temp,
            fire_temp=fire_temp,
            fire_s=fire_s,
            anneal_s=anneal_s,
        )
        return self.add_schedule_specs(
            specs,
            stage_names=names,
            compress_temp_dupes=compress_temp_dupes,
            run=run,
            id_prefix=id_prefix,
        )

    @property
    def simulations(self) -> list[Simulation]:
        return list(self._simulations)

    @property
    def results(self) -> list[Simulation]:
        return self.simulations

    def run_all(self) -> list[Simulation]:
        for sim in self._simulations:
            sim.run(cache=self._cache)
        return self.simulations

    def by_stage_temperature(
        self,
        *,
        stage: str,
        target_temp_C: float,
        tolerance_C: float = 1e-6,
        occurrence: int = 0,
    ) -> list[Simulation]:
        """Return runs whose stage temperature matches ``target_temp_C``."""

        return filter_simulations_by_stage_temperature(
            self,
            stage=stage,
            target_temp_C=target_temp_C,
            tolerance_C=tolerance_C,
            occurrence=occurrence,
        )


__all__ = [
    "Campaign",
    "build_temp_schedule_spec",
    "default_sampling",
    "default_solver",
    "filter_simulations_by_stage_temperature",
    "make_annealing_sweep",
    "make_firing_sweep",
    "make_unfired_sweep",
    "parse_temp_schedule_spec",
    "product_sweep",
    "run_many",
    "stage_temperature_C",
]
