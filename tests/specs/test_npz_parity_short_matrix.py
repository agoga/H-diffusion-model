"""Short NPZ parity matrix for standard testing.

Covers:
- multiple schedule stages (firing + annealing),
- all layers (A..E),
- multiple parameter sets,
while keeping schedule durations very short for practical runtime.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from hdiff.sim import Simulation
from tests.parity_framework import (
    compare_traces,
    full_params,
    legacy_initial_state,
    legacy_like_structure,
    sampling_config,
    schedule_two_stage,
    SeriesTrace,
    MeasureSpec,
    solver_config,
)
from tests.parity_short_matrix import (
    LAYERS,
    SHORT_PARAM_SETS,
    SHORT_SCHEDULE_CASES,
    STAGES,
    matrix_key,
)

BASELINES_ROOT = Path(__file__).resolve().parents[1] / "golden" / "legacy_series_short"
INDEX_PATH = BASELINES_ROOT / "index.json"

if not INDEX_PATH.exists():
    pytest.skip(
        "short NPZ baselines missing; run tests/build_short_npz_parity_baselines.py",
        allow_module_level=True,
    )

INDEX = json.loads(INDEX_PATH.read_text())


@lru_cache(maxsize=None)
def _run_sim(param_id: str, case_id: str) -> Simulation:
    case = next(item for item in SHORT_SCHEDULE_CASES if item.case_id == case_id)
    full = full_params(SHORT_PARAM_SETS[param_id])
    sim = Simulation(
        structure=legacy_like_structure(full),
        schedule=schedule_two_stage(case.fire_C, case.anneal_C, case.fire_s, case.anneal_s),
        sampling=sampling_config(),
        solver=solver_config(),
        y0=legacy_initial_state(),
    )
    sim.run()
    return sim


@pytest.mark.basic_ci
@pytest.mark.parity
@pytest.mark.parity_npz
@pytest.mark.parametrize("param_id", sorted(SHORT_PARAM_SETS.keys()))
@pytest.mark.parametrize("case_id", [case.case_id for case in SHORT_SCHEDULE_CASES])
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("layer", LAYERS)
def test_short_npz_parity_matrix(param_id: str, case_id: str, stage: str, layer: str) -> None:
    """Compare one (params, case, stage, layer) entry against frozen legacy NPZ."""
    key = matrix_key(param_id=param_id, case_id=case_id, stage=stage, layer=layer)
    rel = INDEX.get(key)
    assert rel is not None, f"missing baseline index entry for key={key}"

    data = np.load(BASELINES_ROOT / rel, allow_pickle=True)
    ref = SeriesTrace(
        t_s=np.asarray(data["t_s"], dtype=float),
        y=np.asarray(data["y"], dtype=float),
        source="legacy_series_short_npz",
        metadata={},
    )

    sim = _run_sim(param_id, case_id)
    t_new, y_new = sim.series(layer=layer, kind="total", stage=stage, units="cm^-3")
    new_trace = SeriesTrace(
        t_s=np.asarray(t_new, dtype=float),
        y=np.asarray(y_new, dtype=float),
        source="hdiff",
        metadata={"param_id": param_id, "case_id": case_id},
    )

    metrics = compare_traces(new_trace, ref, probe_t_s=np.asarray(ref.t_s, dtype=float))
    assert metrics.rel_l2 < 0.40, (
        f"{key} rel_l2={metrics.rel_l2:.6f} exceeds 0.400000"
    )
    assert metrics.rel_linf < 0.55, (
        f"{key} rel_linf={metrics.rel_linf:.6f} exceeds 0.550000"
    )
