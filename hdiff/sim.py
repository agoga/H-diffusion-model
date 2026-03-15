"""Core simulation engine for the H-diffusion model.

Implements a finite-volume (FV) ODE for hydrogen transport across a layered
stack.  The ODE is integrated with PETSc/TS (Rosenbrock-W adaptive stepping)
via ``petsc4py``.

State vector layout (``build_state_layout``)
--------------------------------------------
The state vector has ``n_state`` components ordered as::

    trapped_layer0_trapA, trapped_layer0_trapB, ...,
    trapped_layer1_trapA, ...,
    mobile_layer0, mobile_layer1, ..., mobile_layerN

i.e. all trapped populations first (in layer × trap order), then all mobile
populations.  This order is preserved in ``RunResult.y`` columns.

FV discretisation (``_build_rhs``)
------------------------------------
Each layer is treated as one finite-volume cell.  The RHS has two contributions:

1. **Trap kinetics** (per layer, per trap)::

       dC_trapped/dt = k_trap * C_mobile * max(cap - C_trapped, 0) - k_detrap * C_trapped
       dC_mobile/dt -= above

2. **Inter-layer diffusion flux**::

       flux_{i→i+1} = D_mobile * (C_mobile[i] - C_mobile[i+1]) / d_interface
       dC_mobile[i]/dt -= flux * inv_thickness[i]
       dC_mobile[i+1]/dt += flux * inv_thickness[i+1]

   where ``d_interface`` is the centre-to-centre distance (mean of adjacent
   cell widths) and ``inv_thickness`` is the reciprocal cell width.

All concentrations are in *solver units* (physical cm^-3 divided by
``Structure.conc_scale``).

Integration strategy (``_integrate_piecewise``)
-------------------------------------------------
Each compiled segment is integrated in up to two phases:

1. **Bootstrap phase** — covers the first ``Sampling.bootstrap_duration_s``
   seconds with the internal step capped at ``bootstrap_max_dt_s``.  This
   resolves steep initial transients before the adaptor is allowed to coarsen.

2. **Main phase** — ``ts.solve()`` runs to the segment end with free adaptive
   stepping.  A ``ts.setMonitor`` callback interpolates the solution onto a
   regular ``base_out_dt_s`` grid for compact output.

Each phase is a single ``ts.solve()`` call (two per segment total).

Caching (``Simulation.run``)
------------------------------
Each simulation owns one ``RunResult`` that can persist itself to disk.  If the
result is incomplete, ``run()`` first attempts ``result.try_load()`` from
``cache_dir``.  On a miss, the ODE is integrated and the result writes itself.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .result import RunResult
from .result import SegmentBoundary
from .schedule import Schedule, compiled_segment_to_dict
from .boundary import BoundaryFaceContext
from .structure import Structure


@dataclass(frozen=True)
class SolverConfig:
    """PETSc/TS solver settings.

    The backend is always ``'petsc'`` in v1.  Tolerances are passed to
    ``ts.setTolerances``; ``petsc_options`` are applied via ``PETSc.Options``
    before ``ts.setFromOptions``.

    Set ``verbose=True`` to print per-segment progress (start, bootstrap done,
    segment done with wall-clock time) to stdout.
    """

    backend: str = "petsc"
    rtol: float = 1e-7
    atol: float = 1e-9
    petsc_options: dict[str, str] | None = None
    max_steps: int | None = None
    verbose: bool = False

    def validate(self) -> None:
        if not self.backend:
            raise ValueError("solver backend must be non-empty")
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError("solver tolerances must be > 0")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("solver max_steps must be > 0 when provided")


@dataclass(frozen=True)
class Sampling:
    """Controls output density from the ODE integrator."""

    base_out_dt_s: float
    bootstrap_duration_s: float
    bootstrap_max_dt_s: float

    def validate(self) -> None:
        if self.base_out_dt_s <= 0.0:
            raise ValueError("base_out_dt_s must be > 0")
        if self.bootstrap_duration_s < 0.0:
            raise ValueError("bootstrap_duration_s must be >= 0")
        if self.bootstrap_max_dt_s <= 0.0:
            raise ValueError("bootstrap_max_dt_s must be > 0")


def _to_canonical_data(value: Any) -> Any:
    if is_dataclass(value):
        return _to_canonical_data(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_canonical_data(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_canonical_data(v) for v in value]
    if isinstance(value, float):
        return float(value)
    return value


def build_state_layout(structure: Structure) -> dict[str, Any]:
    """Build the index mapping between state-vector columns and physical quantities.

    Returns a dict with:

    * ``order`` — list of ``(kind, layer_name, trap_id_or_None)`` tuples in
      column order (trapped species first, then mobile).
    * ``order_jsonable`` — same as ``order`` but with lists instead of tuples.
    * ``idx_trapped`` — ``{(layer_name, trap_id): column_index}``.
    * ``idx_mobile`` — ``{layer_name: column_index}``.
    * ``n_state`` — total number of state variables.
    """
    structure.validate()
    order: list[tuple[str, str, str | None]] = []
    idx_trapped: dict[tuple[str, str], int] = {}
    idx_mobile: dict[str, int] = {}

    index = 0
    for layer in structure.layers:
        material = structure.materials[layer.material_id]
        for trap in material.traps:
            order.append(("trapped", layer.name, trap.id))
            idx_trapped[(layer.name, trap.id)] = index
            index += 1

    for layer in structure.layers:
        order.append(("mobile", layer.name, None))
        idx_mobile[layer.name] = index
        index += 1

    order_jsonable = [list(item) for item in order]
    return {
        "order": order,
        "order_jsonable": order_jsonable,
        "idx_trapped": idx_trapped,
        "idx_mobile": idx_mobile,
        "n_state": index,
    }


@dataclass
class Simulation:
    """Fully-specified simulation: structure + schedule + sampling + solver settings.

    Construct with all required fields, then call ``run()`` to integrate.
    All structural validation happens in ``__post_init__``; calling ``run()``
    on an invalid object raises before the PETSc solver is initialised.

    After ``run()`` completes (or a cache hit), the result is stored in
    ``self.result`` and accessible via ``series()``, ``layer_total()``, etc.
    """

    structure: Structure
    schedule: Schedule
    sampling: Sampling
    solver: SolverConfig
    y0: list[float] | None = None
    cache_dir: str | Path | None = None
    schema_version: int = 1
    result: RunResult | None = None

    def __post_init__(self) -> None:
        self.structure.validate()
        self.schedule.validate()
        self.sampling.validate()
        self.solver.validate()
        self.geom = self.structure.build_fv_geometry()
        self.layout = build_state_layout(self.structure)
        self.compiled = self.schedule.compile()
        if self.y0 is not None and len(self.y0) != self.layout["n_state"]:
            raise ValueError("y0 length does not match n_state")
        if self.result is None:
            self.result = RunResult(
                cache_key=self.cache_key(),
                schema_version=self.schema_version,
                spec_json=self.spec_json(),
                order_json=json.dumps(
                    self.layout["order_jsonable"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                cache_dir=None if self.cache_dir is None else str(self.cache_dir),
            )

    def build_spec_dict(self) -> dict[str, Any]:
        solver_options = self.solver.petsc_options or {}
        y0_marker = {"kind": "zeros"} if self.y0 is None else {"kind": "values", "values": list(self.y0)}
        return {
            "schema_version": self.schema_version,
            "structure": _to_canonical_data(self.structure),
            "schedule": {
                "segments": _to_canonical_data(self.schedule.segments),
                "n_cycles": self.schedule.n_cycles,
                "compiled": [compiled_segment_to_dict(segment) for segment in self.compiled],
            },
            "sampling": _to_canonical_data(self.sampling),
            "solver": {
                "backend": self.solver.backend,
                "rtol": self.solver.rtol,
                "atol": self.solver.atol,
                "petsc_options": _to_canonical_data(solver_options),
                "max_steps": self.solver.max_steps,
            },
            "y0": y0_marker,
            "layout_order": self.layout["order_jsonable"],
        }

    def spec_json(self) -> str:
        spec_dict = self.build_spec_dict()
        return json.dumps(spec_dict, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def cache_key(self) -> str:
        payload = self.spec_json().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _arrhenius(self, nu: float, Ea_eV: float, T_K: float) -> float:
        kb_eV_per_K = 8.617333262145e-5
        return nu * np.exp(-Ea_eV / (kb_eV_per_K * T_K))

    def _rates_for_segment(self, T_K: float) -> tuple[dict[tuple[str, str], tuple[float, float, float]], float]:
        """Compute Arrhenius rates for all traps and mobile diffusivity at ``T_K``.

        Returns ``(trap_rates, D_mobile)`` where ``trap_rates`` maps
        ``(layer_name, trap_id)`` to ``(k_trap, k_detrap, trap_density)``.
        """
        trap_rates: dict[tuple[str, str], tuple[float, float, float]] = {}
        for layer in self.structure.layers:
            material = self.structure.materials[layer.material_id]
            for trap in material.traps:
                k_trap = self._arrhenius(trap.trap_kin.nu, trap.trap_kin.Ea_eV, T_K)
                k_detrap = self._arrhenius(trap.detrap_kin.nu, trap.detrap_kin.Ea_eV, T_K)
                trap_rates[(layer.name, trap.id)] = (k_trap, k_detrap, trap.trap_density)

        D_mobile = self._arrhenius(
            self.structure.transport.prefactor,
            self.structure.transport.hop_Ea_eV,
            T_K,
        )
        return trap_rates, D_mobile

    def _build_rhs(self, ctx: dict[str, Any]):
        """Build the RHS function for PETSc.

        This returns the function PETSc calls to compute dy/dt from the current
        state y.

        The physics in this ODE comes from three pieces:
        1. trapping and detrapping inside each layer
        2. diffusion of mobile hydrogen between neighboring layers
        3. flux through the left boundary
        """
        # Shortcuts to precomputed indexing / geometry info.
        layout = self.layout
        layer_names = self.geom["layer_names"]
        inv_thickness = self.geom["inv_thickness_cm"]
        d_interfaces = self.geom["d_interface_cm"]

        def rhs(ts: Any, t: float, u: Any, f: Any) -> int:
            # Current state vector.
            y = u.getArray(readonly=True)

            # This will hold dy/dt.
            dy = np.zeros_like(y)

            # Segment-dependent rates already computed outside this function.
            trap_rates: dict[tuple[str, str], tuple[float, float, float]] = ctx["trap_rates"]
            D_mobile: float = ctx["D_mobile"]

            # ------------------------------------------------------------
            # 1. Local trap kinetics in each layer
            # ------------------------------------------------------------
            # Each layer has one mobile-H population and possibly several traps.
            #
            # For each trap:
            #   trap_flux   = k_trap   * mobile * available_capacity
            #   detrap_flux = k_detrap * trapped
            #
            # Net positive means mobile H is getting trapped.
            # That same amount is removed from the mobile population so mass is
            # just moving between mobile and trapped states within the layer.
            for layer in self.structure.layers:
                mobile_idx = layout["idx_mobile"][layer.name]
                mobile_value = y[mobile_idx]
                material = self.structure.materials[layer.material_id]

                for trap in material.traps:
                    trap_idx = layout["idx_trapped"][(layer.name, trap.id)]
                    trapped_value = y[trap_idx]
                    k_trap, k_detrap, capacity = trap_rates[(layer.name, trap.id)]

                    # Remaining room in the trap.
                    available = max(capacity - trapped_value, 0.0)

                    trap_flux = k_trap * mobile_value * available
                    detrap_flux = k_detrap * trapped_value
                    net = trap_flux - detrap_flux

                    dy[trap_idx] += net
                    dy[mobile_idx] -= net

            # ------------------------------------------------------------
            # 2. Diffusion between neighboring layers
            # ------------------------------------------------------------
            # Each layer is one FV cell.
            #
            # We compute the mobile-H flux across each interface from the
            # concentration difference between adjacent cells.
            #
            # Positive flux means stuff flows from the left cell to the right one.
            # That lowers the left cell concentration and raises the right cell
            # concentration.
            for i in range(len(layer_names) - 1):
                left_layer = layer_names[i]
                right_layer = layer_names[i + 1]
                left_idx = layout["idx_mobile"][left_layer]
                right_idx = layout["idx_mobile"][right_layer]

                grad_term = (y[left_idx] - y[right_idx]) / d_interfaces[i]
                flux = D_mobile * grad_term

                dy[left_idx] -= flux * inv_thickness[i]
                dy[right_idx] += flux * inv_thickness[i + 1]

            # ------------------------------------------------------------
            # 3. Left boundary flux
            # ------------------------------------------------------------
            # Ask the boundary-condition object how much mobile H is flowing into
            # the domain through the left face right now.
            #
            # Closed boundary -> returns 0
            # Robin boundary  -> returns k * (reservoir - cell_conc)
            #
            # The BC returns a face flux, so we convert that into the first-cell
            # concentration derivative by dividing by the first-cell thickness.
            left_idx = layout["idx_mobile"][layer_names[0]]

            left_ctx = BoundaryFaceContext(
                t_s=float(t),
                T_K=float(ctx["T_K"]),
                C_cell=float(y[left_idx]),
                dx_cm=float(1.0 / inv_thickness[0]),
                D_mobile_cm2_s=float(D_mobile),
            )

            J_left_in = self.structure.bc.left_flux_into_domain(left_ctx)
            dy[left_idx] += J_left_in * inv_thickness[0]

            # Hand dy/dt back to PETSc.
            f.setArray(dy)
            return 0

        return rhs

    def _integrate_piecewise(self, y0: np.ndarray) -> RunResult:
        """Integrate the ODE across all compiled segments, returning a full ``RunResult``.

        The PETSc TS object is created once and reused across all segments.
        ``rhs_ctx`` is mutated between segments to update temperatures without
        re-registering the RHS function.

        For each segment:
        1. **Bootstrap phase** — ``ts.solve()`` with step cap = ``bootstrap_max_dt_s``.
        2. **Main phase** — ``ts.solve()`` with free adaptive stepping; a monitor
           function interpolates to emit output at ``base_out_dt_s`` intervals.

        The result's ``boundaries`` list records ``[i_start, i_end)`` row ranges
        for each segment, enabling stage-specific slicing later.
        """
        if self.solver.backend != "petsc":
            raise ValueError("only solver backend='petsc' is supported in v1")

        from petsc4py import PETSc

        n_state = self.layout["n_state"]
        u = PETSc.Vec().createSeq(n_state)
        f = PETSc.Vec().createSeq(n_state)
        u.setArray(y0.copy())
        u.assemble()

        rhs_ctx: dict[str, Any] = {
            "trap_rates": {},
            "D_mobile": 0.0,
            "T_K": 0.0,
        }
        rhs = self._build_rhs(rhs_ctx)

        ts = PETSc.TS().create()
        ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
        ts.setRHSFunction(rhs, f)
        ts.setTolerances(rtol=self.solver.rtol, atol=self.solver.atol)
        ts.setTime(0.0)
        ts.setTimeStep(self.sampling.bootstrap_max_dt_s)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
        if self.solver.max_steps is not None:
            ts.setMaxSteps(self.solver.max_steps)

        options = PETSc.Options()
        options["ts_type"] = "rosw"
        options["ts_adapt_type"] = "basic"
        options["ksp_type"] = "preonly"
        options["pc_type"] = "lu"
        if self.solver.petsc_options:
            for key, value in self.solver.petsc_options.items():
                options[str(key)] = str(value)
        ts.setFromOptions()
        ts.setSolution(u)

        times: list[float] = [0.0]
        states: list[np.ndarray] = [y0.copy()]
        boundaries: list[SegmentBoundary] = []

        bootstrap_duration = max(self.sampling.bootstrap_duration_s, 0.0)
        base_out_dt = self.sampling.base_out_dt_s
        boot_out_dt = self.sampling.bootstrap_max_dt_s

        # Shared monitor context: mutated before each solve() call to switch output cadence.
        mon_ctx: dict[str, Any] = {
            "out_dt": boot_out_dt,
            "next_out_t": boot_out_dt,
        }

        def _monitor(ts_obj: Any, step: int, t_now: float, u_vec: Any) -> None:
            t_now = float(t_now)
            out_dt = mon_ctx["out_dt"]
            next_t = mon_ctx["next_out_t"]
            eps = 1e-12
            can_interp = hasattr(ts_obj, "interpolate")

            # Emit interpolated samples for any grid points passed since last step.
            while next_t < t_now - eps:
                if can_interp:
                    u_tmp = u_vec.duplicate()
                    try:
                        ts_obj.interpolate(next_t, u_tmp)
                        vals = u_tmp.getArray(readonly=True).copy()
                    except Exception:
                        vals = u_vec.getArray(readonly=True).copy()
                else:
                    vals = u_vec.getArray(readonly=True).copy()
                if (not times) or abs(times[-1] - next_t) > eps:
                    times.append(float(next_t))
                    states.append(vals)
                next_t += out_dt

            # Emit current step if it lands on or past a grid point.
            if next_t <= t_now + eps:
                vals_now = u_vec.getArray(readonly=True).copy()
                if (not times) or abs(times[-1] - t_now) > eps:
                    times.append(float(t_now))
                    states.append(vals_now)
                next_t = t_now + out_dt

            mon_ctx["next_out_t"] = next_t

        ts.setMonitor(_monitor)

        for segment in self.compiled:
            if segment.T_K is None:
                raise NotImplementedError("T_program support is not implemented in v1")
            trap_rates, D_mobile = self._rates_for_segment(segment.T_K)
            rhs_ctx["trap_rates"] = trap_rates
            rhs_ctx["D_mobile"] = D_mobile
            rhs_ctx["T_K"] = float(segment.T_K)

            i_start = len(times) - 1
            seg_start = segment.t_start_s
            seg_end = segment.t_end_s
            t_now = float(ts.getTime())
            if abs(t_now - seg_start) > 1e-9:
                raise RuntimeError("solver time is out of sync with compiled segment start")

            seg_t0_wall = time.monotonic()
            _verbose = self.solver.verbose
            if _verbose:
                print(
                    f"  [sim] segment '{segment.stage}': {seg_end - seg_start:.3g}s @ {segment.T_K - 273.15:.0f}C",
                    flush=True,
                )

            # Bootstrap phase: one solve(), fine output grid, seed dt capped.
            t_boot_end = min(seg_end, seg_start + bootstrap_duration)
            if t_now < t_boot_end - 1e-15:
                mon_ctx["out_dt"] = boot_out_dt
                mon_ctx["next_out_t"] = t_now + boot_out_dt
                ts.setTimeStep(self.sampling.bootstrap_max_dt_s)
                ts.setMaxTime(float(t_boot_end))
                ts.solve(u)
                t_now = float(ts.getTime())
                vals = u.getArray(readonly=True).copy()
                if abs(times[-1] - t_now) > 1e-12:
                    times.append(t_now)
                    states.append(vals)
            if _verbose:
                print(f"  [sim]   bootstrap done, t={t_now:.4g}s, pts={len(times)}", flush=True)

            # Main phase: one solve(), PETSc adapts freely, monitor emits at base_out_dt grid.
            if t_now < seg_end - 1e-15:
                mon_ctx["out_dt"] = base_out_dt
                mon_ctx["next_out_t"] = t_now + base_out_dt
                ts.setMaxTime(float(seg_end))
                ts.solve(u)
                t_now = float(ts.getTime())
                vals = u.getArray(readonly=True).copy()
                if abs(times[-1] - t_now) > 1e-12:
                    times.append(t_now)
                    states.append(vals)

            if _verbose:
                elapsed_seg = time.monotonic() - seg_t0_wall
                print(
                    f"  [sim] segment '{segment.stage}' done: {len(times) - i_start} pts, {elapsed_seg:.1f}s wall",
                    flush=True,
                )

            i_end = len(times)
            boundaries.append(
                SegmentBoundary(
                    i_seg=segment.i_seg,
                    i_start=i_start,
                    i_end=i_end,
                )
            )

        y_matrix = np.vstack(states)
        return RunResult(
            cache_key=self.cache_key(),
            schema_version=self.schema_version,
            spec_json=self.spec_json(),
            order_json=json.dumps(
                self.layout["order_jsonable"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            cache_dir=None if self.cache_dir is None else str(self.cache_dir),
            t_s=np.asarray(times, dtype=float),
            y=np.asarray(y_matrix, dtype=float),
            boundaries=boundaries,
        )

    def run(self) -> RunResult:
        """Run the simulation.

        The simulation-owned ``RunResult`` is returned.  If that result is not
        complete, ``run()`` first attempts to load from cache and only
        integrates when no valid cache file is available.
        """
        if self.result is None:
            raise RuntimeError("simulation result container was not initialized")

        if self.result.completed:
            return self.result

        if self.result.try_load():
            return self.result

        y0 = np.zeros(self.layout["n_state"], dtype=float) if self.y0 is None else np.asarray(self.y0, dtype=float)
        integrated = self._integrate_piecewise(y0=y0)
        self.result.set_data(
            t_s=np.asarray(integrated.t_s),
            y=np.asarray(integrated.y),
            boundaries=list(integrated.boundaries),
        )
        self.result.save(timestep=float(self.sampling.base_out_dt_s))
        return self.result

    def snapshot_end(self, i_seg: int | None = None) -> list[float]:
        if self.result is None:
            raise ValueError("simulation has not been run")
        if not self.result.boundaries:
            raise ValueError("simulation has no segment boundaries")

        boundary = self.result.boundaries[-1] if i_seg is None else next(
            (item for item in self.result.boundaries if item.i_seg == i_seg),
            None,
        )
        if boundary is None:
            raise ValueError(f"segment boundary not found for i_seg={i_seg}")
        return self.result.y[boundary.i_end - 1, :].tolist()

    def stage_occurrences(self, stage: str) -> int:
        if self.result is None:
            raise ValueError("simulation has not been run")
        return self.result.stage_occurrences(self.compiled, stage)

    def stage_indices(self, stage: str, occurrence: int = 0) -> tuple[int, int]:
        if self.result is None:
            raise ValueError("simulation has not been run")
        boundary = self.result.boundary_for_stage(self.compiled, stage=stage, occurrence=occurrence)
        return boundary.i_start, boundary.i_end

    def stage_bounds(self, stage: str, occurrence: int = 0) -> tuple[float, float]:
        if self.result is None:
            raise ValueError("simulation has not been run")
        return self.result.stage_bounds(self.compiled, stage=stage, occurrence=occurrence)

    def _slice_indices(
        self,
        stage: str | None,
        occurrence: int,
        t_window_s: tuple[float, float] | None,
    ) -> np.ndarray:
        if self.result is None:
            raise ValueError("simulation has not been run")

        if stage is None:
            i_start, i_end = 0, len(self.result.t_s)
        else:
            i_start, i_end = self.stage_indices(stage, occurrence)

        times = self.result.t_s[i_start:i_end]
        if t_window_s is None:
            mask_local = np.ones_like(times, dtype=bool)
        else:
            t_lo, t_hi = t_window_s
            mask_local = (times >= t_lo) & (times <= t_hi)

        local_idx = np.nonzero(mask_local)[0]
        return local_idx + i_start

    def _apply_units(self, values: np.ndarray, units: str) -> np.ndarray:
        if units == "solver":
            return values
        if units == "cm^-3":
            return values * self.structure.conc_scale
        raise ValueError(f"unsupported units: {units}")

    def series(
        self,
        layer: str,
        kind: str,
        trap_id: str | None = None,
        stage: str | None = None,
        occurrence: int = 0,
        t_window_s: tuple[float, float] | None = None,
        units: str = "cm^-3",
    ) -> tuple[list[float], list[float]]:
        """Extract a time-series for a given layer and population kind.

        Args:
            layer: Layer name (must be in ``Structure.layers``).
            kind: ``'mobile'``, ``'trapped'``, or ``'total'`` (mobile + all traps).
            trap_id: Required when ``kind='trapped'`` and the layer has > 1 trap.
            stage: If given, restrict to output rows belonging to this stage.
                Time is re-zeroed to the stage start.
            occurrence: Which repetition of ``stage`` to use (0-based).
            t_window_s: Optional ``(t_lo, t_hi)`` filter applied after stage
                selection (in seconds relative to stage start if ``stage`` is set).
            units: ``'cm^-3'`` (default) or ``'solver'`` (no conversion).

        Returns:
            ``(t_list, values_list)`` as plain Python lists.
        """
        if self.result is None:
            raise ValueError("simulation has not been run")
        if layer not in self.layout["idx_mobile"]:
            raise ValueError(f"unknown layer: {layer}")
        if kind not in {"mobile", "trapped", "total"}:
            raise ValueError(f"unsupported series kind: {kind}")

        indices = self._slice_indices(stage, occurrence, t_window_s)
        t_vals = self.result.t_s[indices]
        if stage is not None and t_vals.size > 0:
            t_start, _ = self.stage_bounds(stage, occurrence)
            t_vals = t_vals - t_start

        if kind == "mobile":
            state_idx = self.layout["idx_mobile"][layer]
            values = self.result.y[indices, state_idx]
        elif kind == "trapped":
            if trap_id is None:
                layer_obj = next((item for item in self.structure.layers if item.name == layer), None)
                if layer_obj is None:
                    raise ValueError(f"unknown layer: {layer}")
                traps = self.structure.materials[layer_obj.material_id].traps
                if len(traps) != 1:
                    raise ValueError("trap_id is required for kind='trapped' when layer has multiple traps")
                trap_id = traps[0].id
            state_idx = self.layout["idx_trapped"].get((layer, trap_id))
            if state_idx is None:
                raise ValueError(f"unknown trap for layer={layer}: {trap_id}")
            values = self.result.y[indices, state_idx]
        else:
            mobile_idx = self.layout["idx_mobile"][layer]
            values = self.result.y[indices, mobile_idx].copy()
            for (layer_name, trap_name), trap_idx in self.layout["idx_trapped"].items():
                if layer_name == layer:
                    values += self.result.y[indices, trap_idx]

        values = self._apply_units(values, units)
        return t_vals.tolist(), values.tolist()

    def layer_total(
        self,
        layer: str,
        stage: str | None = None,
        occurrence: int = 0,
        units: str = "cm^-3",
    ) -> tuple[list[float], list[float]]:
        return self.series(
            layer=layer,
            kind="total",
            trap_id=None,
            stage=stage,
            occurrence=occurrence,
            t_window_s=None,
            units=units,
        )
