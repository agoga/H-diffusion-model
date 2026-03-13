"""Run standard test lanes from one top-level entrypoint.

This script provides obvious, stable commands for the reorganized test harness:
- basic-ci: fast per-change checks
- parity-npz: reproducible parity check against frozen legacy baselines
- parity-live: live parity against legacy runtime (opt-in and heavier)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standard diffusion-model test lanes.")
    parser.add_argument(
        "lane",
        choices=["basic-ci", "parity-npz", "parity-live"],
        help="Which test lane to execute.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    parser.add_argument(
        "--parity-case",
        action="append",
        default=[],
        help="Optional parity case filter. Can be provided multiple times.",
    )
    parser.add_argument(
        "--parity-probe-points",
        default="256",
        help="Probe points for parity interpolation grid.",
    )
    args = parser.parse_args()

    py = args.python

    if args.lane == "basic-ci":
        cmd = [
            py,
            "-m",
            "pytest",
            "-q",
            "tests/specs/test_basic_ci_smoke.py",
            "tests/specs/test_basic_solver_short.py",
            "tests/specs/test_campaign_manager.py",
            "tests/specs/test_npz_parity_short_matrix.py",
            "-m",
            "basic_ci",
        ]
        return run(cmd)

    if args.lane == "parity-npz":
        cmd = [
            py,
            "-m",
            "pytest",
            "-q",
            "tests/specs/test_parity_against_npz_baselines.py",
            "--parity-probe-points",
            str(args.parity_probe_points),
        ]
        for case_id in args.parity_case:
            cmd.extend(["--parity-case", case_id])
        return run(cmd)

    cmd = [
        py,
        "-m",
        "pytest",
        "-q",
        "-m",
        "parity_live",
        "--run-legacy-parity",
        "--parity-probe-points",
        str(args.parity_probe_points),
    ]
    for case_id in args.parity_case:
        cmd.extend(["--parity-case", case_id])
    return run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
