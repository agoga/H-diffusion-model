"""Pytest configuration for the diffusion-model test harness.

Defines shared CLI options/markers for parity lanes and basic CI lanes,
plus path setup so tests can import project and test support modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    from tests.parity_cases import case_ids
except ModuleNotFoundError:
    from parity_cases import case_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

for path in (REPO_ROOT, TESTS_DIR):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--parity-case",
        action="append",
        default=[],
        help="Run only selected parity case id(s). Can be passed multiple times.",
    )
    parser.addoption(
        "--parity-probe-points",
        action="store",
        default="512",
        help="Number of probe points used in NPZ parity interpolation grid.",
    )
    parser.addoption(
        "--run-legacy-parity",
        action="store_true",
        default=False,
        help="Enable live legacy runtime parity tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "basic_ci: fast, foundational CI checks")
    config.addinivalue_line("markers", "parity: parity harness tests")
    config.addinivalue_line("markers", "parity_npz: parity tests against frozen NPZ baselines")
    config.addinivalue_line("markers", "parity_live: parity tests against live legacy runtime")
    config.addinivalue_line("markers", "slow: slower-running tests")


@pytest.fixture(scope="session")
def parity_case_filter(pytestconfig: pytest.Config) -> set[str] | None:
    raw = [str(v).strip() for v in pytestconfig.getoption("parity_case") if str(v).strip()]
    if not raw:
        return None
    chosen = set(raw)
    valid = set(case_ids())
    unknown = chosen.difference(valid)
    if unknown:
        raise pytest.UsageError(
            f"Unknown parity case id(s): {sorted(unknown)}. Valid ids: {sorted(valid)}"
        )
    return chosen


@pytest.fixture(scope="session")
def parity_probe_points(pytestconfig: pytest.Config) -> int:
    raw = str(pytestconfig.getoption("parity_probe_points")).strip()
    try:
        n_points = int(raw)
    except ValueError as exc:
        raise pytest.UsageError(f"--parity-probe-points must be an integer, got: {raw}") from exc
    if n_points < 3:
        raise pytest.UsageError("--parity-probe-points must be >= 3")
    return n_points

