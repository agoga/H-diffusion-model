"""Ad-hoc parity harness helper for interactive/local diagnostics.

Not part of the standard CI lane; intended for manual case runs and plotting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
try:
    from tests.parity_framework import (
        LegacyRuntimeBaselineSource,
        LegacySeriesNpzBaselineSource,
        MeasureSpec,
        ParityCase,
        compare_traces,
        full_params,
        legacy_initial_state,
        paper_param_set,
        run_new_trace,
        sampling_config,
        solver_config,
    )
    from tests.parity_cases import DEFAULT_CASES
except ModuleNotFoundError:
    from parity_framework import (
        LegacyRuntimeBaselineSource,
        LegacySeriesNpzBaselineSource,
        MeasureSpec,
        ParityCase,
        compare_traces,
        full_params,
        legacy_initial_state,
        paper_param_set,
        run_new_trace,
        sampling_config,
        solver_config,
    )
    from parity_cases import DEFAULT_CASES

REPO_ROOT = Path(__file__).resolve().parents[1]


def rel_l2(sim: np.ndarray, ref: np.ndarray) -> float:
    denom = max(np.linalg.norm(ref), 1e-30)
    return float(np.linalg.norm(sim - ref) / denom)


def case_input_payload(case: ParityCase) -> dict[str, Any]:
    base = paper_param_set()
    full = full_params(base)
    return {
        "case": {
            "name": case.name,
            "fire_C": case.fire_C,
            "anneal_C": case.anneal_C,
            "fire_s": case.fire_s,
            "anneal_s": case.anneal_s,
            "threshold_rel_l2": case.threshold_rel_l2,
            "threshold_rel_linf": case.threshold_rel_linf,
        },
        "sampling": {
            "base_out_dt_s": sampling_config().base_out_dt_s,
            "bootstrap_duration_s": sampling_config().bootstrap_duration_s,
            "bootstrap_max_dt_s": sampling_config().bootstrap_max_dt_s,
        },
        "solver": {
            "backend": solver_config().backend,
            "rtol": solver_config().rtol,
            "atol": solver_config().atol,
            "petsc_options": solver_config().petsc_options,
            "max_steps": solver_config().max_steps,
        },
        "y0": legacy_initial_state(),
        "base_param_overrides": base,
        "expanded_params": full,
    }


def print_case_inputs(case: ParityCase) -> None:
    payload = case_input_payload(case)
    print(json.dumps(payload, indent=2, sort_keys=True))


def get_case_by_name(case_name: str) -> ParityCase:
    for case in DEFAULT_CASES:
        if case.name == case_name:
            return case
    valid = ", ".join(case.name for case in DEFAULT_CASES)
    raise KeyError(f"unknown case {case_name!r}; valid cases: {valid}")


def run_case(
    case: ParityCase,
    *,
    verbose: bool = False,
    baseline_mode: str = "legacy-runtime",
) -> dict[str, Any]:
    if verbose:
        print_case_inputs(case)

    measure = MeasureSpec(layer="C", kind="trapped", stage="annealing", units="cm^-3")
    if baseline_mode == "legacy-runtime":
        baseline_source = LegacyRuntimeBaselineSource(base_params=paper_param_set())
    elif baseline_mode == "legacy-npz":
        baseline_source = LegacySeriesNpzBaselineSource(baselines_root=REPO_ROOT / "tests" / "golden" / "legacy_series")
    else:
        raise ValueError(f"unsupported baseline_mode: {baseline_mode}")

    ref = baseline_source.get_trace(case, measure)
    sim = run_new_trace(
        case,
        measure,
        base_params=paper_param_set(),
    )
    metrics = compare_traces(sim, ref)
    y_interp = np.interp(ref.t_s, sim.t_s, sim.y)

    return {
        "case": case,
        "t_legacy": ref.t_s,
        "y_legacy": ref.y,
        "t_new": sim.t_s,
        "y_new": sim.y,
        "y_new_on_legacy_t": y_interp,
        "rel_l2": metrics.rel_l2,
        "rel_linf": metrics.rel_linf,
        "baseline_source": ref.source,
    }


def run_and_compare(
    case_name: str,
    *,
    verbose: bool = False,
    baseline_mode: str = "legacy-runtime",
) -> dict[str, Any]:
    return run_case(
        get_case_by_name(case_name),
        verbose=verbose,
        baseline_mode=baseline_mode,
    )


def print_case_metrics(result: dict[str, Any]) -> None:
    case: ParityCase = result["case"]
    print(
        f"{case.name}: rel_l2={result['rel_l2']:.6f} "
        f"rel_linf={result['rel_linf']:.6f} "
        f"baseline={result['baseline_source']}"
    )


def plot_case_result(result: dict[str, Any]):
    from hdiff.viz import make_parity_figure

    fig, axes = make_parity_figure([result])
    return fig, axes[0, 0]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        valid = ", ".join(case.name for case in DEFAULT_CASES)
        print(f"usage: python -m tests.run_case <case-name> [legacy-runtime|legacy-npz]")
        print(f"valid cases: {valid}")
        return 2

    case_name = args[0]
    baseline_mode = args[1] if len(args) > 1 else "legacy-runtime"
    result = run_and_compare(case_name, verbose=True, baseline_mode=baseline_mode)
    print_case_metrics(result)
    return 0
