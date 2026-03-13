# Test harness layout

This folder is split by runtime cost and test purpose.

- **Smoke test**: a quick health check that verifies key paths run and return sane values. It is intentionally small and fast.
- **Parity test**: compares new-code output to legacy/reference output for the same inputs.
- **Live parity**: parity test that executes legacy code at runtime.
- **NPZ baseline parity**: parity test that compares to pre-saved legacy outputs (`.npz`) for reproducibility and speed.
- **CI lane**: a subset of tests grouped for a specific purpose (for example fast per-change checks).


## File map

- `conftest.py`: shared pytest options, markers, and parity CLI controls.
- `run_test_lanes.py`: main entrypoint script to run standard test lanes.
- `test_helpers.py`: shared test data/builders used by many specs.
- `specs/test_basic_ci_smoke.py`: deterministic fast checks (compile and slicing/query behavior).
- `specs/test_basic_solver_short.py`: short real-solver scenario plus short live parity scenario.
- `specs/test_npz_parity_short_matrix.py`: standard-lane short NPZ parity matrix over stages/layers/parameter sets.
- `specs/test_structure_geometry.py`: structure + geometry contracts.
- `specs/test_schedule_compile.py`: schedule compile contracts.
- `specs/test_layout_and_cache.py`: state layout and cache schema contracts.
- `specs/test_simulation_solver_and_queries.py`: heavier integration contracts (restart/cache/query).
- `parity_framework.py`: parity adapters, baseline loaders, interpolation/metrics.
- `parity_cases.py`: shared parity case catalog and IDs.
- `specs/test_parity_against_npz_baselines.py`: reproducible parity against frozen NPZ baselines.
- `specs/test_golden_parity_prefix.py`: live legacy parity for two-stage golden cases.
- `specs/test_parity_single_stage_layers.py`: live single-stage parity split by layer.
- `parity_harness.py`: manual/interactive parity helper, not standard CI.
- `build_legacy_series_baselines.py`: utility to regenerate frozen NPZ baselines.

## Run lanes

Top-level runner:

```bash
conda activate diff
python tests/run_test_lanes.py basic-ci
python tests/run_test_lanes.py parity-npz --parity-case anneal_250 --parity-probe-points 256
python tests/run_test_lanes.py parity-live --parity-case anneal_250_short --parity-probe-points 128
```

Fast per-change lane:

```bash
conda activate diff
python -m pytest -q -m basic_ci
```

Repro parity lane (no live legacy runtime):

```bash
python -m pytest -q tests/specs/test_parity_against_npz_baselines.py --parity-case anneal_250 --parity-probe-points 256
```

Live legacy parity lane (on demand):

```bash
python -m pytest -q -m parity_live --run-legacy-parity --parity-probe-points 128
```

## Documentation style in test files

Use short module docstrings at the top of each file to explain:
- test lane role,
- expected runtime profile,
- whether file is CI lane or manual utility.

Add a short function docstring when the test name alone does not make the target behavior obvious.

Prefer docstrings over long inline comments inside test bodies.
