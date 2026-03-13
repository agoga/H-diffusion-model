# H-diffusion model

Hydrogen diffusion and trapping model for layered solar-cell stacks.
The current code centers on typed structure definitions, cached schedule runs,
and stage-sliced plotting.

## Quick start

```bash
conda activate diff
python -m pytest tests/ -q          # 71 unit + parity tests
```

Run one schedule directly:

```python
from hdiff.cache import CacheStore
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
)
result = sim.run(cache=CacheStore("sim_data"))

t, y = sim.series("A", "total", stage="annealing")  # cm^-3
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
  structure.py         — physical stack dataclasses (layers, traps, transport)
  schedule.py          — schedule / sampling dataclasses + compile()
  sim.py               — Simulation + SolverConfig; PETSc/TS integration
  result.py            — RunResult; SegmentBoundary; slice helpers
  cache.py             — CacheStore: NPZ on-disk cache keyed by spec SHA-256
  campaign.py          — schedule builders, batch runners, legacy-shaped helpers

legacy/                — frozen reference copy of the original fourstates.py code
  fourstates.py        — original PETSc solver (do not import from new code)
  helpers.py           — legacy parameter defaults
  simulation_manager.py
  sweeps.py / vizkit.py / nrel_exp.py

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

### Default sampling used by the direct quickstart

| Parameter             | Default  |
|-----------------------|----------|
| `base_out_dt_s`       | 10 s     |
| `bootstrap_duration_s`| 10 s     |
| `bootstrap_max_dt_s`  | 1e-4 s   |

### Default solver

| Parameter | Default |
|-----------|---------|
| `rtol`    | 1e-5    |
| `atol`    | 1e-10   |
| backend   | PETSc Rosenbrock-W (`ts_type=rosw`) |

## Caching

Results are stored as compressed NPZ files in `sim_data/` (or any directory you
pass to `CacheStore`).  Each file is keyed by the SHA-256 hex digest of a
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

`hdiff.campaign` still contains legacy-shaped helpers for sweep scripts and parity code,
but the main path is typed structures plus immutable overrides.

## Layer stack

| Label | Material | Thickness |
|-------|----------|-----------|
| A     | SiNx (bulk nitride)      | 100 nm   |
| B     | SiNx (interface nitride) | 290 nm   |
| C     | SiOx                     | 1.5 nm   |
| D     | Si (emitter)             | 608.5 nm |
| E     | Si (base)                | 50 µm    |

## Sweep workflows

Generate schedule lists with `make_annealing_sweep` / `make_firing_sweep`, then
run them with direct `Simulation` construction or `run_many`:

```python
from hdiff.campaign import make_annealing_sweep, parse_temp_schedule_spec, run_many
from hdiff.defaults import DEFAULT_SAMPLING, DEFAULT_SOLVER, DEFAULT_STRUCTURE

schedules, stage_names = make_annealing_sweep(
    anneal_temps=[200, 225, 250, 300, 350],
    fire_temp=750, fire_s=600,
    anneal_s=8_000_000,
    include_room=False, room_temp=25, room_s=0,
)

  schedule_map = {
    f"anneal_{temp}": parse_temp_schedule_spec(spec, stage_names)
    for temp, spec in zip([200, 225, 250, 300, 350], schedules, strict=True)
  }

  runs = run_many(
    DEFAULT_STRUCTURE,
    schedule_map,
    DEFAULT_SAMPLING,
    DEFAULT_SOLVER,
  )
```

## Plotting framework

Use `hdiff.viz` for composable plotting. The API is split into:

- panel functions (`plot_trace_overlay`, `plot_abs_error`, `plot_rel_error`, `plot_layer_stage_sweep`, `plot_all_layers_for_stage`, `plot_sweep_heatmap`)
- figure builders (`make_parity_figure`, `make_all_layers_over_phases_figure`)

### Core plot 1: one layer during one stage for all simulations matching a fixed stage temperature

```python
import matplotlib.pyplot as plt
from hdiff.viz import plot_layer_stage_sweep

fig, ax = plt.subplots(figsize=(7.2, 4.8))
plot_layer_stage_sweep(
  ax,
  simulations=mgr.simulations,
  match_stage="firing",      # fix this stage temperature
  target_temp_C=750.0,
  layer="C",
  kind="trapped",
  plot_stage="annealing",    # plot this stage
  x_units="hours",
)
```

### Core plot 2: all layers over each phase for one simulation

```python
from hdiff.viz import make_all_layers_over_phases_figure

sim = mgr.simulations[0]
fig, axes = make_all_layers_over_phases_figure(
  sim,
  stages=["firing", "annealing"],
  kind="total",
  x_units="hours",
)
```

### Other common plots carried over from legacy workflows

- parity overlays and error panels: `make_parity_figure(results)`
- peak-time vs stage-temperature scatter: `plot_peak_time_vs_stage_temperature(...)`
- sweep metric heatmaps from tabular results: `plot_sweep_heatmap(...)`

### Notebook quickstart

The current example notebook is `hdiff/main.ipynb`.

```python
# See hdiff/main.ipynb for the direct-structure version used in this repo.
```

```python
# Cell 2: core plot #1
# One layer concentration during annealing for all runs with firing fixed at 750C
import matplotlib.pyplot as plt
from hdiff.viz import plot_layer_stage_sweep

fig, ax = plt.subplots(figsize=(8.0, 5.0))
plot_layer_stage_sweep(
  ax,
  simulations=simulations,
  match_stage="firing",
  target_temp_C=750.0,
  layer="C",
  kind="trapped",
  plot_stage="annealing",
  x_units="hours",
)
fig.tight_layout()
```

```python
# Cell 3: core plot #2
# All layers over each phase for one selected simulation
from hdiff.viz import make_all_layers_over_phases_figure

first_sim = simulations[0]
fig, axes = make_all_layers_over_phases_figure(
  first_sim,
  stages=["firing", "annealing"],
  kind="total",
  x_units="hours",
)
fig.tight_layout()
```

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
