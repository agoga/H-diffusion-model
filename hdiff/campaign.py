"""Orchestration utilities: schedule builders and sweep runners.

Public surface
--------------
* ``Campaign`` — high-level manager for a named set of schedules with
        a fixed structure; provides stage-indexed series access.
* ``run_many`` / ``product_sweep`` — lower-level helpers kept for explicit
    custom sweeps (for example structure-parameter grid runs).

Schedule grammar and sweep template builders live in ``schedule.py``.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable

from .defaults import DEFAULT_SAMPLING, DEFAULT_SOLVER, DEFAULT_STRUCTURE
from . import schedule as schedule_api
from .schedule import Schedule
from .sim import Sampling, Simulation, SolverConfig
from .structure import Structure



def run_many(
    structure: Structure,
    schedules: dict[str, Schedule],
    sampling: Sampling,
    solver: SolverConfig,
    y0: list[float] | None = None,
    cache_dir: str | Path | None = None,
    schema_version: int = 1,
) -> dict[str, Simulation]:
    """Run (or cache-load) one ``Simulation`` per schedule.

    Args:
        structure: Shared physical structure for all schedules.
        schedules: Mapping of schedule-id → ``Schedule``.
        sampling: Shared sampling settings.
        solver: Shared solver settings.
        y0: Optional initial state; defaults to all-zeros.
        cache_dir: Optional directory used for NPZ cache files.
            If ``None`` results are not persisted.
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
            cache_dir=cache_dir,
            schema_version=schema_version,
        )
        out[schedule_id] = sim
        if cache_dir is not None:
            sim.run()
    return out


def product_sweep(
    base_structure: Structure,
    axes: dict[str, list[Any]],
    apply: Callable[[Structure, dict[str, Any]], Structure],
    schedules: dict[str, Schedule],
    sampling: Sampling,
    solver: SolverConfig,
    y0: list[float] | None = None,
    cache_dir: str | Path | None = None,
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
            cache_dir=cache_dir,
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


def find_simulation(
    campaign: "Campaign",
    stage_temps: list[tuple[str, float]],
    *,
    tolerance_C: float = 1e-6,
    occurrence: int = 0,
) -> Simulation | None:
    """Find the simulation that matches all provided stage temperatures.

    Args:
        campaign: Campaign to search.
        stage_temps: List of ``(stage_name, target_temp_C)`` pairs that must
            all match.  For example::

                campaign.find(
                    [("firing", 670), ("annealing", 250)]
                )

        tolerance_C: Maximum deviation (°C) allowed for each stage.
        occurrence: Which occurrence of each stage to check (0-indexed).

    Returns:
        The matching :class:`~hdiff.sim.Simulation`, or ``None`` if no match
        is found.  When multiple simulations satisfy all constraints the one
        with the longest total schedule duration is returned — this is
        consistent with how campaign deduplication works.
    """
    candidates: list[Simulation] = []
    for sim in campaign.simulations:
        if all(
            abs(stage_temperature_C(sim, stage, occurrence) - float(target_C)) <= float(tolerance_C)
            for stage, target_C in stage_temps
        ):
            candidates.append(sim)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Tie-break: longest total schedule duration.
    def _total_duration(s: Simulation) -> float:
        return float(sum(seg.duration_s for seg in s.schedule.segments))
    return max(candidates, key=_total_duration)


def other_stage_name(sim: Simulation, stage: str) -> str | None:
    """Return the first stage name in ``sim``'s schedule that is not ``stage``.

    Useful for two-phase schedules (firing + annealing) where you want to
    identify the complementary stage from a known one.

    Returns ``None`` if ``stage`` is the only stage in the schedule.
    """
    names: list[str] = []
    for seg in sim.schedule.compile():
        if seg.stage not in names:
            names.append(seg.stage)
    for name in names:
        if name != stage:
            return name
    return None


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
        self._sampling = DEFAULT_SAMPLING if sampling is None else sampling
        self._solver = DEFAULT_SOLVER if solver is None else solver
        self._results_dir = None if results_dir is None else str(Path(results_dir))
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
            parsed.append((schedule_id, schedule_api.parse_temp_schedule_spec(str(spec), names)))
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
                cache_dir=self._results_dir,
                schema_version=self._schema_version,
            )
            self._schedules.append(schedule_id)
            self._simulations.append(sim)
            added.append(sim)

        if run:
            for sim in added:
                sim.run()
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

        specs, names = schedule_api.make_annealing_sweep(
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

        specs, names = schedule_api.make_firing_sweep(
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

        specs, names = schedule_api.make_unfired_sweep(
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
            sim.run()
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

    def find(
        self,
        stage_temps: list[tuple[str, float]],
        *,
        tolerance_C: float = 1e-6,
        occurrence: int = 0,
    ) -> Simulation | None:
        """Find the simulation matching all ``(stage, temp_C)`` constraints.

        Equivalent to :func:`find_simulation` called on this campaign.
        When multiple simulations match, the one with the longest total schedule
        duration is returned.  Returns ``None`` if no match is found.

        Example::

            sim = campaign.find([("firing", 670), ("annealing", 250)])
        """

        return find_simulation(
            self,
            stage_temps,
            tolerance_C=tolerance_C,
            occurrence=occurrence,
        )


__all__ = [
    "Campaign",
    "filter_simulations_by_stage_temperature",
    "find_simulation",
    "other_stage_name",
    "product_sweep",
    "run_many",
    "stage_temperature_C",
]
