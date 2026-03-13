# simulation_manager.py
# Purpose:
#   - Manage one parameter set across many temperature schedules (strings).
#   - Run-or-load each schedule exactly once via fourstates.run_or_load_sim.
#   - Provide a clean, documented API for querying per-simulation series and
#     lightweight cross-simulation “grid” helpers for plotting.
# Authors: Adam Goga 

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Optional
import numpy as np

import fourstates as fs  # single source of truth for state order & param builder
import helpers as h      # for parsing schedules to canonical numeric form, constants


# ----------------------------- One simulation record -----------------------------

@dataclass
class SimulationResult:
    """
    Container + helpers for a single simulation's data.

    Expected shapes (exactly as fourstates.py saves them):
      - times: (N,)
      - ut:    (N, 10)   # columns = states in fs.STATE_ORDER

    Layer names: "A","B","C","D","E"
    Kinds/species: "trapped","mobile"

    Tricky bits documented inline:
      - Stage windows are derived from the numeric schedule (solver's truth).
      - Stage names are derived from *schedule length* (not computed bounds).
      - Stage slicing avoids double-counting the boundary sample.
      - All temperature APIs exposed in °C for convenience.
      - Series scaling is *non-inplace*; we never mutate the stored arrays.
    """

    # Unique hash/signature emitted by the solver (or a canonical fallback).
    signature: str

    # Parameters actually used for the run (resolved/expanded via fs.build_params).
    params: SimpleNamespace

    # Human-readable schedule string as provided to the manager.
    schedule_spec: str

    # Optional stable formatting derived from the parsed absolute schedule.
    schedule_canon: Optional[str] = None

    # Dict returned by fourstates.run_or_load_sim / simulate
    data: dict[str, Any] = None

    # Cached stage windows (absolute time), computed on first use.
    _bounds: Optional[list[tuple[float, float]]] = None

    # NEW: explicit stage names provided by the manager (preferred over guessing)
    stage_names_spec: Optional[list[str]] = None
    
    @classmethod
    def from_solver_dict(
        cls,
        data: dict[str, Any],
        *,
        stage_names: list[str] | None = None,
        signature: str | None = None,
    ) -> "SimulationResult":
        import numpy as np
        import fourstates as fs

        args_sched = data.get("args_schedule", None)

        # Normalize common np.savez / np.load shapes
        if isinstance(args_sched, np.ndarray):
            # 0-D object array -> unwrap
            if args_sched.ndim == 0:
                args_sched = args_sched.item()
            else:
                args_sched = args_sched.tolist()

        # If it is a single dict, wrap it
        if isinstance(args_sched, dict):
            args_sched = [args_sched]

        # If it is a list/tuple, unwrap any 0-D arrays inside
        if isinstance(args_sched, (list, tuple)):
            norm = []
            for x in args_sched:
                if isinstance(x, np.ndarray) and x.ndim == 0:
                    x = x.item()
                norm.append(x)
            args_sched = norm

        if not isinstance(args_sched, (list, tuple)) or len(args_sched) == 0:
            raise ValueError(
                f"Need non-empty args_schedule to rebuild params. "
                f"Got type={type(args_sched).__name__}, keys={list(data.keys())}"
            )

        if not isinstance(args_sched[0], dict):
            raise ValueError(
                f"args_schedule[0] must be a dict. Got {type(args_sched[0]).__name__}"
            )

        p = fs.build_params(**args_sched[0])
        sig = signature or str(data.get("signature", "adhoc"))

        return cls(
            signature=sig,
            params=p,
            schedule_spec=str(p.temp_schedule),
            schedule_canon=None,
            data=data,
            stage_names_spec=list(stage_names) if stage_names is not None else None,
        )


    # ---- core arrays as saved by fourstates.py ----

    @property
    def times(self) -> np.ndarray:
        """Absolute simulation times t in seconds, shape (N,)."""
        t = np.asarray(self.data["times"])
        if t.ndim != 1:
            raise ValueError(f"times must be (N,), got {t.shape}")
        return t

    @property
    def ut(self) -> np.ndarray:
        """
        State matrix with shape (N, 10).
        Columns follow the canonical order in fourstates.STATE_ORDER:
          trapped A..E, then mobile A..E
        """
        U = np.asarray(self.data["ut"])
        if U.ndim != 2 or U.shape[1] != 10 or U.shape[0] != self.times.shape[0]:
            raise ValueError(
                f"ut must be (N,10) aligned with times, got {U.shape} vs {self.times.shape}"
            )
        return U

    @property
    def schedule_numeric(self) -> list[tuple[float, float]]:
        """
        Canonical numeric schedule used by the solver: list of (t_end_seconds, T_kelvin).

        NOTE:
        - This is the *source of truth* for stage math.
        - Do not use `schedule_spec` for logic; it's display-only.
        """
        sched_raw = self.data.get("temp_schedule", None)
        if isinstance(sched_raw, np.ndarray):
            return sched_raw.tolist()
        return list(sched_raw or [])

    # ---- stage metadata (computed from the numeric schedule) ----

    def _compute_stage_bounds(self) -> list[tuple[float, float]]:
        """
        Convert temp_schedule [(t_end, T[K]), ...] into contiguous, closed-open
        windows over absolute time: [(0,t1), (t1,t2), ...].

        If recorded time extends beyond the last t_k, append a tail window that
        ends at times[-1].

        Tricky bit:
        - We form *closed-open* windows to make slicing unambiguous and avoid
          double-counting boundary samples. Stage 0 includes the left edge;
          later stages are right-open at the left edge when slicing.
        """
        sched = self.schedule_numeric
        bounds: list[tuple[float, float]] = []
        t_prev = 0.0
        for t_end, _Tk in sched:
            t_end = float(t_end)
            bounds.append((t_prev, t_end))
            t_prev = t_end

        t = self.times
        if t.size == 0:
            return bounds or [(0.0, 0.0)]

        t_last = float(t[-1])
        if not bounds:
            bounds = [(0.0, t_last)]
        elif bounds[-1][1] < t_last:
            bounds.append((bounds[-1][1], t_last))
        return bounds

    def _slice_for_stage(self, stage_index: int) -> slice:
        """
        Safe slice into `times` for stage `stage_index`.

        Convention:
        - For all but the last stage: [t0, t1)  (include t0, exclude t1)
        - For the last stage:         [t0, end] (include all remaining points)

        This guarantees:
        - No overlap between consecutive stages.
        - No “lost” points at the boundaries.
        """
        bounds = self.stage_bounds()
        i = int(stage_index)
        if i < 0 or i >= len(bounds):
            return slice(0, 0)

        t = self.times
        t0, t1 = bounds[i]

        # left index: first index with t >= t0
        i0 = int(np.searchsorted(t, t0, side="left"))

        # right index:
        #   - for last stage: go all the way to the end
        #   - otherwise: first index with t >= t1 (exclude exact t1)
        if i == len(bounds) - 1:
            i1 = t.size
        else:
            i1 = int(np.searchsorted(t, t1, side="left"))

        i0 = max(0, min(i0, t.size))
        i1 = max(i0, min(i1, t.size))
        return slice(i0, i1)


    def _normalize_stage(self, stage: int | str | None) -> int | None:
        """
        Accept stage as int or name; return validated int index or None.
        Raises for out-of-range or unknown name.
        """
        if stage is None:
            return None
        if isinstance(stage, int):
            i = int(stage)
            if i < 0 or i >= self.num_stages():
                raise KeyError(
                    f"Stage index {i} out of range (num_stages={self.num_stages()})."
                )
            return i
        if isinstance(stage, str):
            return self.stage_index_by_name(stage)
        raise TypeError(f"stage must be int, str, or None; got {type(stage).__name__}")

    def stage_bounds(self) -> list[tuple[float, float]]:
        """List of absolute time windows [(t0, t1), ...] for each stage."""
        if self._bounds is None:
            self._bounds = self._compute_stage_bounds()
        return self._bounds

    def num_stages(self) -> int:
        """Number of stages derived from temp_schedule (plus tail if present)."""
        return len(self.stage_bounds())

    # Stage naming policy
    #  - 1 stage: ["firing"]
    #  - 2 stages: ["firing", "annealing"]
    #  - 3 stages: ["firing", "room_rest", "annealing"]
    #  - otherwise: ["stage0", "stage1", ...]
    #
    # - Names are based on the *length of the numeric schedule* (i.e., the number
    #   of temperature entries), not on computed stage_bounds(). This ensures that
    #   mapping from "stage index" <-> "stage temperature" is stable and obvious.
    def stage_names(self) -> list[str]:
        """
        Stage naming policy.

        Preferred:
        - If `stage_names_spec` was provided by the manager, just return that.

        Fallback (for backward compatibility):
        - Infer names from the numeric schedule length.
        """
        if self.stage_names_spec is not None:
            return list(self.stage_names_spec)
        #bad but fallback
        return None


    def has_stage(self, name: str) -> bool:
        """Return True if this simulation has a stage by that name."""
        return name in self.stage_names()

    def stage_index_by_name(self, name: str) -> int:
        """Return numeric index of a named stage (raises KeyError if not found)."""
        names = self.stage_names()
        try:
            return names.index(name)
        except ValueError as e:
            raise KeyError(f"Stage '{name}' not found. Available: {names}") from e

    def temperature_for_stage(self, stage: int | str) -> float:
        """
        Get stage temperature in °C.
        Tricky bit: numeric schedule stores Kelvin; we convert to Celsius here so
        that all external users can compare with human-friendly values.
        """

        i = self._normalize_stage(stage)

        sched = self.schedule_numeric
        if i is None or i < 0 or i >= len(sched):
            raise KeyError(f"Stage {stage!r} not found in schedule {sched}")
        Tk = float(sched[i][1])
        return Tk - 273.15

    # ---- per-series accessors (operate on ONE simulation) ----

    def series(
        self,
        *,
        layer: str,
        kind: str,
        stage: int | str | None = None,
        
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (time [s], y_scaled) for the specified (layer, kind).

        - layer: "A".."E"
        - kind:  "trapped" | "mobile"
        - stage: None for full trace, or an int/string stage index.
                 When slicing a stage, time is re-zeroed to start at 0 for that stage.

        Tricky bits:
        - We take *copies* of arrays to avoid mutating self.ut/self.times.
        - Scaling uses helpers.DEFAULT_CONSTANTS["SCALE_Y"] and is non-inplace.
        """
        col = fs.state_index(layer, kind)

        # Work on copies so we never mutate underlying arrays
        t_abs_s = np.asarray(self.times, dtype=float)                 # (N,), copy ok
        y_base  = np.array(self.ut[:, col], dtype=float, copy=True)   # (N,), forced copy

        i = self._normalize_stage(stage)
        if i is None:
            t = t_abs_s
            y = y_base
        else:
            sl = self._slice_for_stage(i)
            t = t_abs_s[sl]
            if t.size:
                t = np.maximum(t - t[0], 0.0)  # re-zero to stage start
            y = y_base[sl].copy()

        # Non-inplace scaling (never mutate the stored arrays)
        scale_y = float(h.DEFAULT_CONSTANTS["SCALE_Y"])
        y = y * scale_y
        return t, y

    def mobile(self, layer: str, stage: int | str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Shortcut for series(..., kind='mobile')."""
        return self.series(layer=layer, kind="mobile", stage=stage)

    def trapped(self, layer: str, stage: int | str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Shortcut for series(..., kind='trapped')."""
        return self.series(layer=layer, kind="trapped", stage=stage)

    def total(self, layer: str, stage: int | str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Sum of mobile + trapped for the layer, on a consistent timebase.
        Tricky bit: we compute both series on the same sliced timebase to keep alignment.
        """
        t_m, y_m = self.mobile(layer, stage=stage)
        t_t, y_t = self.trapped(layer, stage=stage)
        return t_m, (y_m + y_t)

    # ------------------------------------------------------------
    #  Flux time-series (returns (t, F) where F has shape (N, 4))
    # ------------------------------------------------------------
    def fluxes(self, stage: int | str | None = None, units="atoms"):
        """
        Return (t, F) where F has shape (N,4):
            F[:,0] = flux A→B
            F[:,1] = flux B→C
            F[:,2] = flux C→D
            F[:,3] = flux D→E

        units:
            "fraction"  → return raw model flux (fraction/s)
            "atoms"     → return atoms / cm^2 / s (physical flux)
        """
        import helpers as h

        # --- stage slicing ---
        stage_idx = self._normalize_stage(stage)
        t_abs = np.asarray(self.times, float)
        U = np.asarray(self.ut, float)
        sched = self.schedule_numeric

        if stage_idx is None:
            sl = slice(None)
            t = t_abs.copy()
        else:
            sl = self._slice_for_stage(stage_idx)
            t = t_abs[sl].copy()
            if t.size:
                t -= t[0]

        U_slice = U[sl]

        # --- compute fractional flux using helpers.rhs_from_state ---
        F_list = []
        for ti, ui in zip(t_abs[sl], U_slice):
            T = h.T_of_time(float(ti), sched)
            _, fluxes = h.rhs_from_state(ui, self.params, T)
            F_list.append(fluxes)

        F_frac = np.vstack(F_list) if F_list else np.zeros((0, 4))

        # ---------------------------------------------------------------
        #   Units: raw fraction per second
        # ---------------------------------------------------------------
        if units == "fraction":
            return t, F_frac

        # ===============================================================
        #   PHYSICAL UNITS: ATOMS / cm^2 / s
        # ===============================================================

        SCALE_Y = float(h.DEFAULT_CONSTANTS["SCALE_Y"])  # cm^-3 concentration scale

        # ---- compute actual physical layer thicknesses (cm) ----
        # Your SAMPLE_LENGTH = 1000 nm = 1e-4 cm
        

        dxA = fs.SCALE_A * fs.SAMPLE_LENGTH * fs.NM_to_CM     # nm → cm
        dxB = fs.SCALE_B * fs.SAMPLE_LENGTH * fs.NM_to_CM
        dxC = fs.SCALE_C * fs.SAMPLE_LENGTH * fs.NM_to_CM
        dxD = fs.SCALE_D * fs.SAMPLE_LENGTH * fs.NM_to_CM
        dxE = fs.SCALE_E * 1e-4                         # μm → cm

        # Source layers for each interface
        src_thick = np.array([dxA, dxB, dxC, dxD], float)


        # Convert fractional flux → physical atoms / cm^2 / s
        F_atoms = F_frac * SCALE_Y * src_thick

        return t, F_atoms



    # ------------------------------------------------------------
    #  Derivative time-series (returns (t, dUdt) with shape (N,10))
    # ------------------------------------------------------------
    def derivs(self, stage: int | str | None = None):
        """
        Return (t, dUdt) where dUdt has shape (N,10), matching
        the STATE_ORDER = [A,B,C,D,E, HA,HB,HC,HD,HE].

        Uses helpers.rhs_from_state, respecting stage slicing
        and re-zeroing time if stage!=None.
        """
        import helpers as h

        stage_idx = self._normalize_stage(stage)
        t_abs = np.asarray(self.times, float)
        U = np.asarray(self.ut, float)
        sched = self.schedule_numeric

        if stage_idx is None:
            sl = slice(None)
            t = t_abs.copy()
        else:
            sl = self._slice_for_stage(stage_idx)
            t = t_abs[sl].copy()
            if t.size:
                t -= t[0]

        U_slice = U[sl]

        dU_list = []
        for ti, ui in zip(t_abs[sl], U_slice):
            T = h.T_of_time(float(ti), sched)
            derivs, _ = h.rhs_from_state(ui, self.params, T)
            dU_list.append(derivs)

        dU = np.vstack(dU_list) if dU_list else np.zeros((0, 10))
        return t, dU


    # ------------------------------------------------------------
    #  Optional: cumulative flux (integral over time)
    # ------------------------------------------------------------
    def net_flux(self, stage: int | str | None = None):
        """
        Returns cumulative ∫flux dt for each interface.
        Output shape: (4,) corresponding to AB, BC, CD, DE.

        A positive value means net flow toward the right (A→B etc).
        """
        t, F = self.fluxes(stage=stage)
        if len(t) < 2:
            return np.zeros(4)
        return np.trapz(F, t, axis=0)

    def dHdt(
        self, *, layer: str, kind: str = "mobile", stage: int | str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Time derivative of the requested series using np.gradient with edge_order=2.
        Returns (t, dy/dt).
        """
        t, y = self.series(layer=layer, kind=kind, stage=stage)
        if t.size < 2:
            return t, np.zeros_like(t)
        return t, np.gradient(y, t, edge_order=2)
    
    # inside class SimulationResult
    def total_delta_from_t0(
        self, layer: str, stage: int | str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        ΔH for `layer` relative to the *start of the requested stage* (or t=0 if stage=None).

        Returns (t_stage, y_total - y_total[stage_start]).
        Time is already re-zeroed by series()/total() when stage is not None.
        """
        # The commented version would find the delta from t0 in the firing stages
        # s0_t, s0_y_tot = self.total(layer, stage=0)
        # if s0_y_tot.size and y_tot.size:
        #     y0 = float(s0_y_tot[0])   # baseline = initial value *in this stage*
        #     return t, (y_tot - y0)
        # Series for the requested window (already stage-sliced & re-zeroed if stage!=None)
        t, y_tot = self.total(layer, stage=stage)
        if y_tot.size:
            y0 = float(y_tot[0])   # baseline = initial value *in this stage*
            return t, (y_tot - y0)
        return t, y_tot


# ----------------------- Manager: many schedules, one param set -----------------------
class SimulationManager:
    """
    Manage one parameter set across many temperature schedules.

    - We collapse duplicates by *temperature sequence*.
    - For any set of stage temperatures (ignoring durations), we keep only
      the schedule with the *longest total duration*.
    - This prevents re-running equivalent temperature programs that only differ
      by shorter/partial time coverage, and ensures we run/load exactly once.

    - Memoization key includes the canonical schedule string + PETSc options.
    - Stage temperature comparisons exposed in °C with a small tolerance.
    - Public grid helpers return {schedule_str: (t, y)} maps for easy plotting.
    """


    def __init__(
        self,
        base_params: SimpleNamespace | dict[str, Any],
        temp_schedules: Iterable[str],
        *,
        stage_names: Optional[Iterable[str]] = None,  # NEW
        results_dir: Optional[str] = None,
        petsc_options: Optional[dict[str, Any]] = None,
        u0=None,
        tol: float = 0.1,
        verbose: bool = True,
    ):
        # Build a fully-resolved parameter namespace once
        self._params = fs.build_params(
            **(vars(base_params) if isinstance(base_params, SimpleNamespace) else base_params)
        )
        self._results_dir = results_dir
        self._petsc = petsc_options or {}
        self._verbose = verbose
        self._tol = float(tol)
        self._u0 = u0
        # explicit stage names for all runs in this manager
        self._stage_names: Optional[list[str]] = (
            list(stage_names) if stage_names is not None else None
        )

        # In-process memo for loaded results (keyed by canonical schedule + PETSc opts)
        self._mem: dict[tuple[str, tuple[tuple[str, str], ...]], SimulationResult] = {}

        # Keep only the longest schedule for each temperature sequence
        self._schedules: list[str] = self._compress_temp_dupes(list(temp_schedules))

        self._results: list[SimulationResult] = []
        for s in self._schedules:
            self._results.append(self._ensure(s))

    

    # ----------------- basic accessors -----------------
    @property
    def stage_names(self) -> Optional[list[str]]:
        """Explicit stage names shared by all runs in this manager, if provided."""
        return list(self._stage_names) if self._stage_names is not None else None

    @property
    def params(self) -> SimpleNamespace:
        """Resolved parameter namespace used for runs."""
        return self._params

    @property
    def schedules(self) -> list[str]:
        """List of *kept* schedule strings after deduping by temperature sequence."""
        return list(self._schedules)

    def __contains__(self, schedule: str) -> bool:
        """True if a schedule with the same temperature sequence is kept."""
        sig = self._temps_signature(schedule)
        return any(self._temps_signature(s) == sig for s in self._schedules)

    def __len__(self) -> int:
        """Number of kept schedules."""
        return len(self._schedules)

    # ----------------- dedupe helpers ------------------

    @staticmethod
    def _temps_signature(schedule: str) -> tuple[float, ...]:
        """
        Tuple of stage temperatures in °C (rounded) — defines equivalence class.

        Rationale:
        - Two schedules are considered equivalent if they contain the *same sequence*
          of stage temperatures (ignoring durations). We round to a few decimals to
          be robust against harmless float noise in parsing.
        """
        _segs, abs_sched = h.parse_temp_schedule(schedule)
        return tuple(round(float(Tk) - 273.15, 6) for _t, Tk in abs_sched)

    @staticmethod
    def _total_duration(schedule: str) -> float:
        """Total run end time in seconds for the schedule."""
        _segs, abs_sched = h.parse_temp_schedule(schedule)
        return float(abs_sched[-1][0]) if abs_sched else 0.0

    def _compress_temp_dupes(self, schedules: list[str]) -> list[str]:
        """
        Keep only the longest schedule for each temperature sequence.

        Tricky bit (the “schedule fixing” rule):
        - If multiple schedules share the same *sequence of temperatures* across stages,
          we keep the one with the *largest total duration* and drop the rest.
        - This prevents redundant runs and makes plotting less cluttered.
        """
        best: dict[tuple[float, ...], tuple[float, str]] = {}
        for s in schedules:
            sig = self._temps_signature(s)
            dur = self._total_duration(s)
            if (sig not in best) or (dur > best[sig][0]):
                best[sig] = (dur, s)
        return [s for (_dur, s) in best.values()]

    # Public adders (respect dedupe rule)
    def add_schedule(self, schedule: str) -> bool:
        """
        Add a schedule if it is a new temperature sequence or longer than the
        existing kept schedule with the same temperature sequence.

        Returns True if the candidate schedule is kept (added or replaced).
        """
        sig_new = self._temps_signature(schedule)
        dur_new = self._total_duration(schedule)

        idx = None
        for i, s in enumerate(self._schedules):
            if self._temps_signature(s) == sig_new:
                idx = i
                break

        if idx is None:
            self._schedules.append(schedule)
            return True

        if dur_new > self._total_duration(self._schedules[idx]):
            self._schedules[idx] = schedule
            return True

        return False

    def add_schedules(self, schedules: Iterable[str]) -> int:
        """Add multiple schedules, returning the count that were kept."""
        n = 0
        for s in schedules:
            n += 1 if self.add_schedule(s) else 0
        return n

    # ----------------- caching / loading -----------------

    @staticmethod
    def _canonical_schedule_str(schedule: str) -> str:
        """
        Stable (t_end[s], T[K]) formatting to key the memo cache.

        Example: "10.000000s@973.15K|610.000000s@623.15K"
        """
        _segs, abs_sched = h.parse_temp_schedule(schedule)
        return "|".join(f"{float(t_end):.6f}s@{float(Tk):.2f}K" for t_end, Tk in abs_sched)

    def _record_for(self, schedule: str) -> dict[str, Any]:
        """Create the solver 'rec' payload from params + schedule."""
        rec = dict(vars(self._params))
        rec["temp_schedule"] = schedule
        return rec

    def _ensure(self, schedule: str) -> SimulationResult:
        """
        Materialize a SimulationResult for `schedule` (load/run *once per process*).

        Tricky bit:
        - Memoization key includes canonical schedule string *and* PETSc options
          (as a sorted tuple) to avoid collisions across different solver settings.
        """
        canon = self._canonical_schedule_str(schedule)
        petsc_key = tuple(sorted((str(k), str(v)) for k, v in self._petsc.items()))
        key = (canon, petsc_key)
        if key in self._mem:
            return self._mem[key]

        rec = self._record_for(schedule)
        data, _used_cache = fs.run_or_load_sim(
            rec=rec,
            results_dir=self._results_dir,
            petsc_options=self._petsc,
            verbose=self._verbose,
            u0=self._u0,
        )

        t = np.asarray(data["times"])
        U = np.asarray(data["ut"])
        if t.ndim != 1 or U.ndim != 2 or U.shape[1] != 10 or t.shape[0] != U.shape[0]:
            raise ValueError(f"Unexpected shapes: times {t.shape}, ut {U.shape}")

        sig = str(data.get("signature", canon))
        res = SimulationResult(
            signature=sig,
            params=self._params,
            schedule_spec=schedule,
            schedule_canon=canon,
            data=data,
            stage_names_spec=self._stage_names,   
        )
        self._mem[key] = res
        return res


    def run_all(self) -> list[SimulationResult]:
        """Materialize all kept (deduped) schedules."""
        return [self.ensure(s) for s in self.schedules]

    # ----------------- query helpers -------------------

    def results_with_stage_temperature(
        self, *, stage: int | str, target_T: float
    ) -> list[SimulationResult]:
        """
        Return results whose given stage temperature equals target_T (± self._tol), using °C.
        """
        matched: list[SimulationResult] = []
        for res in self._results:
            try:
                T_stage = res.temperature_for_stage(stage)  # °C
                if abs(T_stage - target_T) <= self._tol:
                    matched.append(res)
            except KeyError:
                continue
        return matched

    def peak_times_for_stageT(
        self,
        *,
        target_T: float,
        stage_sort: str = "firing",
        layer: str = "C",
        kind: str = "trapped",
        stage_for_peak: str = "annealing",
    ) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for res in self._results:
            if not (res.has_stage(stage_for_peak) and res.has_stage(stage_sort)):
                continue
            try:
                T_peak_stage = res.temperature_for_stage(stage_for_peak)  # °C
            except KeyError:
                continue
            if abs(T_peak_stage - target_T) > self._tol:
                continue

            t, y = res.series(layer=layer, kind=kind, stage=stage_for_peak)
            if t.size == 0 or y.size == 0:
                continue
            i_max = int(np.argmax(y))
            T_sort = res.temperature_for_stage(stage_sort)  # °C
            out.append((float(T_sort), float(t[i_max])))

        out.sort(key=lambda xy: xy[0])
        return out

    # ----------------- grids for plotting ---------------

    def series_grid(
        self, *, layer: str, kind: str, stage: int | str | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            schedule: res.series(layer=layer, kind=kind, stage=stage)
            for schedule, res in zip(self._schedules, self._results)
        }

    def mobile_grid(
        self, *, layer: str, stage: int | str | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            schedule: res.mobile(layer, stage=stage)
            for schedule, res in zip(self._schedules, self._results)
        }

    def trapped_grid(
        self, *, layer: str, stage: int | str | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            schedule: res.trapped(layer, stage=stage)
            for schedule, res in zip(self._schedules, self._results)
        }

    def total_grid(
        self, *, layer: str, stage: int | str | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            schedule: res.total(layer, stage=stage)
            for schedule, res in zip(self._schedules, self._results)
        }

    def dHdt_grid(
        self, *, layer: str, kind: str = "mobile", stage: int | str | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            schedule: res.dHdt(layer=layer, kind=kind, stage=stage)
            for schedule, res in zip(self._schedules, self._results)
        }
    
    

        # ----------------- derivative / flux grids -----------------

    def flux_grid(
        self, *, stage: int | str | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Return: { schedule: (t, flux_array) } for all kept simulations.
        flux_array has shape (N, 4): [A→B, B→C, C→D, D→E].
        """
        out = {}
        for schedule, res in zip(self._schedules, self._results):
            t, F = res.fluxes(stage=stage)
            out[schedule] = (t, F)
        return out

    def deriv_grid(
        self, *, stage: int | str | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Return: { schedule: (t, dU) } where dU has shape (N,10).
        """
        out = {}
        for schedule, res in zip(self._schedules, self._results):
            t, dU = res.derivs(stage=stage)
            out[schedule] = (t, dU)
        return out

    def net_flux_grid(
        self, *, stage: int | str | None = None
    ) -> dict[str, np.ndarray]:
        """
        Return a map:
            { schedule : net_flux_vector }
        where net_flux_vector is shape (4,) = ∫ flux dt.
        This is useful for interface transport totals.
        """
        out = {}
        for schedule, res in zip(self._schedules, self._results):
            NF = res.net_flux(stage=stage)
            out[schedule] = NF
        return out
