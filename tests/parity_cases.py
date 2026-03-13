"""Central catalog of parity case definitions and selection helpers.

Keeps test case identities in one place so CLI filtering, parametrization,
and documentation stay consistent across parity test modules.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from tests.parity_framework import MeasureSpec, ParityCase
except ModuleNotFoundError:
    from parity_framework import MeasureSpec, ParityCase


@dataclass(frozen=True)
class HarnessConfig:
    measure: MeasureSpec
    cases: tuple[ParityCase, ...]


DEFAULT_MEASURE = MeasureSpec(layer="C", kind="trapped", stage="annealing", units="cm^-3")

DEFAULT_CASES: tuple[ParityCase, ...] = (
    ParityCase(name="anneal_225", fire_C=650.0, anneal_C=225.0, fire_s=10.0, anneal_s=8_000_000.0),
    ParityCase(name="anneal_250", fire_C=650.0, anneal_C=250.0, fire_s=10.0, anneal_s=8_000_000.0),
    ParityCase(name="anneal_350", fire_C=650.0, anneal_C=350.0, fire_s=10.0, anneal_s=8_000_000.0),
    ParityCase(name="fire_750", fire_C=750.0, anneal_C=250.0, fire_s=10.0, anneal_s=8_000_000.0),
)

SINGLE_STAGE_ANNEAL_CASES: tuple[tuple[str, float, float], ...] = (
    ("anneal_250_short", 250.0, 5_000.0),
    ("anneal_350_short", 350.0, 5_000.0),
)

LAYER_NAMES: tuple[str, ...] = ("A", "B", "C", "D", "E")

SMOKE_CASES: tuple[ParityCase, ...] = (
    DEFAULT_CASES[1],
    DEFAULT_CASES[3],
)


def default_harness_config() -> HarnessConfig:
    return HarnessConfig(measure=DEFAULT_MEASURE, cases=DEFAULT_CASES)


def case_ids(cases: tuple[ParityCase, ...] = DEFAULT_CASES) -> tuple[str, ...]:
    two_stage_ids = tuple(case.name for case in cases)
    single_stage_ids = tuple(case_id for case_id, _temp_C, _duration_s in SINGLE_STAGE_ANNEAL_CASES)
    return two_stage_ids + single_stage_ids


def select_cases(
    *,
    names: set[str] | None,
    cases: tuple[ParityCase, ...] = DEFAULT_CASES,
) -> tuple[ParityCase, ...]:
    if not names:
        return cases
    return tuple(case for case in cases if case.name in names)
