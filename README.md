# H-diffusion model

Hydrogen diffusion and trapping model for layered solar-cell stacks.
The current code centers on typed structure definitions, cached schedule runs,
and stage-sliced plotting.

## Quick start

```bash
conda activate diff
python -m pytest tests/ -q
```

Run one schedule directly:

```python
from hdiff.defaults import DEFAULT_SAMPLING, DEFAULT_SOLVER, DEFAULT_STRUCTURE
from hdiff.schedule import Schedule, Segment
from hdiff.sim import Simulation

schedule = Schedule(
  segments=[
    Segment(duration_s=600.0, stage="firing", T_C=750.0),
    Segment(duration_s=8_000_000.0, stage="annealing", T_C=250.0),
  ],
)
sim = Simulation(
    structure=DEFAULT_STRUCTURE,
    schedule=schedule,
    sampling=DEFAULT_SAMPLING,
    solver=DEFAULT_SOLVER,
  cache_dir="sim_data",
)
result = sim.run()

t_s, y = sim.layer_total("SiO$_x$", stage="annealing")  # cm^-3
```

Adjust a default without rebuilding the whole structure:

```python
from hdiff.defaults import DEFAULT_STRUCTURE

custom = (
  DEFAULT_STRUCTURE
  .with_material("csi", trap_id="t1", trap_nu=5e13)
  .with_transport(prefactor=2e-3)
)
```

## Repository layout

```
hdiff/                 — simulation package (new clean-slate code)
  boundary.py          — boundary-law models (closed/robin) and BC context
  structure.py         — physical stack dataclasses (layers, traps, transport)
  schedule.py          — schedule dataclasses + schedule parser/sweep templates
  sim.py               — Simulation + Sampling + SolverConfig; PETSc/TS integration
  result.py            — RunResult; SegmentBoundary; query + cache persistence
  campaign.py          — orchestration over many simulations

legacy/                — archived reference scripts/notebooks (not active API)

tests/
  specs/               — pytest unit + parity test files
  parity_framework.py  — shared parity utilities (run_new_trace, baseline source)
  parity_cases.py      — canonical parity case definitions (anneal_225, fire_750 …)
  parity_harness.py    — golden NPZ loader for offline parity checks
  tests.ipynb          — interactive diagnostic notebook

sim_data/              — cache directory for new-code NPZ results
origin_exports/        — CSV exports used for quick visual debugging only
docs/
  AGENT_PACKET.md      — design notes from the initial rewrite
```

## Physics

The model treats each layer as one finite-volume cell.  The state vector has
`n_state = n_traps_total + n_layers` components:

```
[trapped_A, trapped_B, …, trapped_E,   mobile_A, …, mobile_E]
```

**Trap kinetics** (per layer, per trap):

$$\frac{dC_\text{trapped}}{dt} = k_\text{trap}(T)\,C_\text{mobile}\,(C_\text{cap} - C_\text{trapped}) - k_\text{detrap}(T)\,C_\text{trapped}$$

**Inter-layer diffusion flux** (between adjacent FV cells):

$$J_{i \to i+1} = D(T)\,\frac{C_i - C_{i+1}}{d_\text{interface}}$$

All rates follow an Arrhenius law $k(T) = \nu \exp(-E_a / k_B T)$.  Solver
concentrations are dimensionless (physical = solver × `conc_scale = 1e22` cm⁻³).

## Solver

PETSc/TS with Rosenbrock-W adaptive stepping (`ts_type = rosw`).
Each schedule segment is integrated in two phases:

1. **Bootstrap** — short initial window with step capped at `bootstrap_max_dt_s`
   to resolve rapid transients.
2. **Main** — free adaptive stepping; a `ts.setMonitor` callback interpolates
   output onto a regular `base_out_dt_s` grid.

For current default runtime parameters, use `hdiff.defaults.DEFAULT_SAMPLING`
and `hdiff.defaults.DEFAULT_SOLVER` (source of truth: `hdiff/defaults.py`).

## Caching

Results are stored as compressed NPZ files in `sim_data/` (or any `cache_dir`
you pass to `Simulation`).  Each file is keyed by the SHA-256 hex digest of a
canonical JSON spec that covers structure, schedule, sampling, solver tolerances,
and initial conditions.  Any change to those inputs produces a new key and a new
file.

The `completed` scalar in each NPZ must be `1` for the record to be loaded;
partial runs (e.g. from interrupted jobs) are silently skipped.

## Default materials and overrides

The default stack lives in `hdiff.defaults` as reusable `Material` objects:

- `DEFAULT_ALOX`
- `DEFAULT_POLY`
- `DEFAULT_SIOX`
- `DEFAULT_CSI`
- `DEFAULT_STRUCTURE`

Override a trap in one material:

```python
from hdiff.defaults import DEFAULT_CSI

custom_csi = DEFAULT_CSI.with_trap("t1", trap_nu=5e13, trap_density=2e-5)
```

Override a material inside a full structure:

```python
from hdiff.defaults import DEFAULT_STRUCTURE

custom_structure = DEFAULT_STRUCTURE.with_material(
  "csi",
  trap_id="t1",
  trap_nu=5e13,
  detrap_Ea_eV=1.0,
)
```

Use `hdiff.campaign.Campaign` as the primary orchestration path for sweeps.

## Layer stack

The canonical stack definition is `hdiff.defaults.DEFAULT_STRUCTURE` in
`hdiff/defaults.py`.

## Sweep workflows

Use `Campaign` sweep helpers directly (`sweep_annealing`, `sweep_firing`, `sweep_unfired`):

```python
from hdiff.campaign import Campaign
from hdiff.defaults import DEFAULT_STRUCTURE

campaign = Campaign(
  structure=DEFAULT_STRUCTURE,
  results_dir="sim_data",
  auto_run=False,
)

campaign.sweep_annealing(
  anneal_temps=[200, 225, 250, 300, 350],
  fire_temp=750,
  fire_s=600,
  anneal_s=8_000_000,
  include_room=False,
  run=True,
)

matched = campaign.by_stage_temperature(stage="firing", target_temp_C=750.0)
sim = matched[0]
t_s, y_total = sim.layer_total("SiO$_x$", stage="annealing")
```

## Plotting framework

Use `hdiff.viz` for composable plotting. The API is split into:

- panel functions (`plot_trace_overlay`, `plot_abs_error`, `plot_rel_error`, `plot_layer_stage_sweep`, `plot_all_layers_for_stage`, `plot_sweep_heatmap`)
- figure builders (`make_parity_figure`, `make_all_layers_over_phases_figure`)

To avoid README drift, plotting examples are maintained in code and notebooks:

- `hdiff/main.ipynb` (interactive workflow)
- `tests/specs/test_viz_framework.py` (minimal callable examples)
- `hdiff/viz.py` (API/docstrings for panel and figure builders)

Common entry points:

- `plot_layer_stage_sweep(...)`
- `make_areal_density_figure(...)`
- `make_parity_figure(...)`
- `plot_peak_time_vs_stage_temperature(...)`
- `plot_sweep_heatmap(...)`

## Running tests

```bash
conda activate diff

# All tests (fast + parity against cached NPZ baselines):
python -m pytest tests/ -q

# Only fast CI-safe tests:
python -m pytest tests/ -q -m basic_ci

# With solver progress output:
python -m pytest tests/ -q -s -m basic_ci
```

## Environment

The simulation stack requires `petsc4py` + PETSc.  Analysis dependencies:
`numpy`, `matplotlib`, `pandas`, `scipy`.

```bash
conda activate diff          # provides petsc4py, numpy, matplotlib, pandas, scipy
```
