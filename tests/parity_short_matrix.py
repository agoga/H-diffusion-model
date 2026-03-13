"""Definitions for short NPZ parity matrix cases used in standard testing.

These cases are intentionally short to keep runtime practical while still
covering multiple stages, all layers, and multiple parameter sets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShortScheduleCase:
    case_id: str
    fire_C: float
    anneal_C: float
    fire_s: float
    anneal_s: float


SHORT_SCHEDULE_CASES: tuple[ShortScheduleCase, ...] = (
    ShortScheduleCase(case_id="short_500_250", fire_C=500.0, anneal_C=250.0, fire_s=0.05, anneal_s=0.15),
    ShortScheduleCase(case_id="short_650_350", fire_C=650.0, anneal_C=350.0, fire_s=0.05, anneal_s=0.15),
)

SHORT_PARAM_SETS: dict[str, dict[str, float]] = {
    "user_basic": {
        "A_trap": 0.5,
        "B_trap": 0.5,
        "C_trap": 0.5,
        "D_trap": 0.5,
        "A_detrap": 2.0,
        "B_detrap": 1.0,
        "C_detrap": 1.0,
        "D_detrap": 1.0,
    },
    "paper_like": {
        "A_detrap": 2.5,
        "B_detrap": 1.3,
        "C_detrap": 1.45,
        "D_detrap": 1.2,
        "A_trap": 0.9,
        "B_trap": 0.50,
        "C_trap": 0.50,
        "D_trap": 0.50,
        "A_detrap_attemptfreq": 1e12,
        "B_detrap_attemptfreq": 1e12,
        "C_detrap_attemptfreq": 1e12,
        "D_detrap_attemptfreq": 1e12,
        "A_trap_attemptfreq": 1e13,
        "B_trap_attemptfreq": 1e13,
        "C_trap_attemptfreq": 1e13,
        "D_trap_attemptfreq": 5e12,
    },
}

STAGES: tuple[str, ...] = ("firing", "annealing")
LAYERS: tuple[str, ...] = ("A", "B", "C", "D", "E")


def matrix_key(*, param_id: str, case_id: str, stage: str, layer: str) -> str:
    return f"{param_id}__{case_id}__{stage}__{layer}"
