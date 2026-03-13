# Parity harness

Run these tests in small chunks.

## Environment setup

Use the project Conda environment before running parity tests:

```bash
conda activate diff
python -m pytest --version
```

If `pytest` is missing, you are likely in the wrong environment.

## Test groups

- `parity_npz`: new code vs frozen `tests/golden/legacy_series` baselines
- `parity_live`: new code vs live legacy runtime (`legacy/simulation_manager.py`)

## Baseline modes

- `legacy-runtime`: runs the legacy solver directly. Slowest option.
- `legacy-npz`: reads frozen traces from `tests/golden/legacy_series`. Fastest option for local iteration.

The interactive harness in `tests/parity_harness.py` and the CLI in `python -m tests.run_case` both accept these baseline mode names.

## Common options

- `--parity-case <id>`: select one or more case IDs
- `--parity-probe-points <N>`: interpolation probe density (default `512`)
- `--run-legacy-parity`: enable live legacy parity tests

Valid case IDs include:
- Two-stage: `anneal_225`, `anneal_250`, `anneal_350`, `fire_750`
- Single-stage: `anneal_250_short`, `anneal_350_short`

## Common commands

Run one small frozen-baseline check:

```bash
python -m pytest -q tests/specs/test_parity_against_npz_baselines.py --parity-case anneal_250 --parity-probe-points 256
```

Run one live two-stage parity case:

```bash
python -m pytest -q tests/specs/test_golden_parity_prefix.py --run-legacy-parity --parity-case anneal_250
```

Run one named case through the interactive harness wrapper:

```bash
python -m tests.run_case anneal_350 legacy-npz
```

Run one live single-stage all-layer concentration parity case:

```bash
python -m pytest -q tests/specs/test_parity_single_stage_layers.py --run-legacy-parity --parity-case anneal_250_short
```

Run one specific layer test node:

```bash
python -m pytest -q "tests/specs/test_parity_single_stage_layers.py::test_single_stage_anneal_layer_total_parity[anneal_250_short-A]" --run-legacy-parity
```

## Notes

- Start with one case and one file.
- Increase case count only after a green run.
- Keep `--parity-probe-points` low (e.g. `256`) for smoke checks and raise it only for deeper validation.
- Add `-s` to `pytest` when you want the live parity tests to print per-case `rel_l2` and `rel_linf` metrics.
