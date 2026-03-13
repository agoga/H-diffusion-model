# Basic CI test lane

Use this lane for the short check set after a code change.

## Terms

- **Smoke test**: fast test that checks core behavior works at all.
- **Deterministic test**: test that avoids solver/runtime variability and should produce stable outputs every run.
- **Parity test**: test that compares new output to legacy/reference output for the same inputs.

## Environment

```bash
conda activate diff
python -c "import sys; print(sys.executable)"
```

## Fast lane

```bash
python -m pytest -q -m basic_ci
```

This covers:
- basic schedule compilation behavior (`500C` for `10s`, single stage)
- basic stage/series alignment using a deterministic `RunResult`
- `RunResult` boundary slicing behavior
- short NPZ parity matrix (all layers, both stages, multiple short schedules, multiple parameter sets)

## Short real-solver smoke (same basic input family)

Uses a single-stage `500C` short run (`1s`) with:
- trap Ea = `0.5` for all layers
- detrap Ea = `2.0` for layer A, `1.0` for B/C/D/E

```bash
python -m pytest -q tests/specs/test_basic_solver_short.py::test_short_single_stage_500c_real_solver_runs_and_aligns_queries
```

## Extended lane (still focused, excludes parity)

```bash
python -m pytest -q -m "basic_ci or not parity"
```

## Parity lane (heavier)

Run separately when needed:

```bash
python -m pytest -q tests/specs/test_parity_against_npz_baselines.py --parity-case anneal_250 --parity-probe-points 256
```

Short live legacy-vs-new parity for the same 500C short case:

```bash
python -m pytest -q tests/specs/test_basic_solver_short.py::test_short_single_stage_500c_live_legacy_parity_per_layer --run-legacy-parity --parity-probe-points 128
```
