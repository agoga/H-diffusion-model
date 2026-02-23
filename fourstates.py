#!/usr/bin/env python3
# fourstates.py
"""
Four-state diffusion model (stiff + dense output)
- geometry/constants
- parameter canonicalization & hashing (with schema version)
- caching (npz) save/load
- PETSc simulation

Authors: Adam Goga & Zitong Zhao
"""



from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import base64
import hashlib
import json
import os
import zlib

import numpy as np
from petsc4py import PETSc

import helpers as h

DEBUG=False

NM_to_CM = 1e-7

# =========================
# Geometry / scaling
# =========================
 # Layers A,B,C,D all encompass 1um and layer E is the rest of the c-Si at 50um. D and E are both c-Si
SAMPLE_LENGTH: int = 1000 
SCALE_A: float = 100 / SAMPLE_LENGTH
SCALE_B: float = 290 / SAMPLE_LENGTH
SCALE_C: float = 1.5 / SAMPLE_LENGTH
SCALE_D: float = 1 - SCALE_A - SCALE_B - SCALE_C
SCALE_E: float = 50  # 50 um bulk



# =========================
# Versioning & cache paths
# =========================
_DEFAULT_RESULTS_DIR: str = str((Path(__file__).parent / "sim_data").resolve())

SCHEMA_VERSION: str = "1"  # bump whenever math/outputs change
_PARAM_DEFAULTS: dict[str, Any] = dict(
    temp_schedule="10:950K",
    diffusion_rate=1e-3,
    hopping=0.5,
    # --- Defect Densities (per unit volume) ---
    N_A=1.0,
    N_B=1e-3,
    N_C=1e-1,#1e-1 default for paper
    N_D=1e-5,
    N_E=1e-5,
    # --- Trap / detrap energies (eV) ---
    A_detrap=2.40,
    B_detrap=1.2,
    C_detrap=1.45,
    D_detrap=1.1,
    A_trap=0.9,
    B_trap=0.50,
    C_trap=0.50,
    D_trap=0.50,
    # --- Attempt frequencies (s⁻¹) ---
    A_detrap_attemptfreq=1e12,
    B_detrap_attemptfreq=1e12,
    C_detrap_attemptfreq=1e12,
    D_detrap_attemptfreq=1e12,
    A_trap_attemptfreq  =1e12,
    B_trap_attemptfreq  =1e12,
    C_trap_attemptfreq  =1e12,
    D_trap_attemptfreq  =1e12,
    # --- Timestep/Saving controls ---
    timestep=1e-8,
    output_dt=10.0,
    # “hot micro-window” bootstrap controls
    bootstrap_seconds=11,
    bootstrap_dt_max=1e-4,
    bootstrap_output_dt=1e-4,
    bootstrap_seed_dt=1e-4,
)

# ---- Canonical state layout (change schema_version if these change) -----------------------
# Order: trapped A..E, then mobile A..E (matches RHS unpack and everywhere else)
STATE_LAYERS: tuple[str, ...] = ("A", "B", "C", "D", "E")
STATE_KINDS: tuple[str, ...] = ("trapped", "mobile")

STATE_ORDER: tuple[tuple[str, str], ...] = tuple(
    [(L, "trapped") for L in STATE_LAYERS] + [(L, "mobile") for L in STATE_LAYERS]
)

STATE_INDEX: dict[tuple[str, str], int] = {pair: i for i, pair in enumerate(STATE_ORDER)}
UT_INDEX: dict[str, int] = {f"{L}_{k}": i for i, (L, k) in enumerate(STATE_ORDER)}


# =========================
# Public API
# =========================

def simulate(
    params: SimpleNamespace,
    *,
    petsc_options: dict[str, Any] | None = None,
    results_dir: str | None = None,
    verbose: bool = False,
    u0: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """Run the simulator end-to-end, with automatic caching.
    u0: Optional initial condition vector (length 10) to override default or cached initial state."""
    rec = build_sim_record(params=params, petsc_options=petsc_options,u0=u0)
    results_dir_effective = _results_dir(results_dir)

    # cache
    cached = _try_load_cached(rec, results_dir_effective)
    if cached is not None:
        if verbose:
            print(f"[cache hit] {cached['cache_path']}")
        return cached

    # schedule
    segs_new, schedule_abs_new = h.parse_temp_schedule(params.temp_schedule)
    temp_schedule = schedule_abs_new
    args_schedule = [h.params_to_dict(params) for _ in range(len(segs_new))]

    # choose save path
    primary_path, alt_path = _paths_for_record(rec, results_dir_effective)
    save_path = primary_path if not os.path.exists(primary_path) else alt_path

    # PETSc set-up
    u = PETSc.Vec().createSeq(10)
    F = PETSc.Vec().createSeq(10)
    
    ctx: dict[str, Any] = {
        "k": {},
        "N": {"A": params.N_A, "B": params.N_B, "C": params.N_C, "D": params.N_D, "E": params.N_E},
        "D": 0.0,
        "d_AB": SCALE_A / 2.0 + SCALE_B / 2.0,
        "d_BC": SCALE_B / 2.0 + SCALE_C / 2.0,
        "d_CD": SCALE_C / 2.0 + SCALE_D / 2.0,
        "d_DE": SCALE_D / 2.0 + SCALE_E / 2.0,
        "T": None,
        "u0": u0,
        "temp_schedule": temp_schedule,
        "args_schedule": args_schedule,
        "history": [],
        "args": params,
        "out_dt": float(params.output_dt),
        "out_dt_active": float(params.output_dt),
        "next_out_t": None,
        "segment_t_start": 0.0,
        "t0_global": 0.0,
    }

    t0 = 0.0
    ctx["t0_global"] = float(t0)
    T0 = h.T_of_time(t0, temp_schedule)
    _update_rates_for_T(ctx, T0)
    

    ts = PETSc.TS().create()
    ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
    ts.setTolerances(rtol=1e-5, atol=1e-10)
    ts.setRHSFunction(_rhs, F, args=(ctx,))
    ts.setTimeStep(params.timestep)

    # default PETSc options (ROS-W + adaptivity)
    opts = PETSc.Options()
    opts["ts_type"] = "rosw"
    opts["ts_adapt_type"] = "basic"
    opts["ts_adapt_safety"] = 0.9
    opts["ts_adapt_clip"] = "0.1,2.0"
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["ts_exact_final_time"] = "matchstep"
    if petsc_options:
        for k, v in petsc_options.items():
            opts[str(k)] = v if isinstance(v, str) else str(v)
    ts.setFromOptions()

    ts.setMonitor(_monitor_dense, args=(ctx,))
    ts.setTime(float(t0))
    _initial_conditions(u, ctx)
    ts.setSolution(u)

    if ctx["next_out_t"] is None:
        ctx["next_out_t"] = _ceil_to_grid(float(t0), float(t0) + 1e-15, ctx["out_dt"])
    if len(ctx["history"]) == 0:
        vals0 = u.getArray(readonly=True).copy()
        ctx["history"].append((float(t0), vals0))

    # integrate piecewise and save checkpoints
    _solve_piecewise(ts, u, ctx, save_path=save_path, sim_record=rec)

    # final overwrite with final state
    times = np.array([x[0] for x in ctx["history"]], dtype=float)
    ut = np.array([x[1] for x in ctx["history"]], dtype=float)
    reason = ts.getConvergedReason()

    _save_npz(
        save_path,
        times=times,
        ut=ut,
        timestep=ts.getTimeStep(),
        reason=reason,
        sim_record=rec,
        temp_schedule=ctx["temp_schedule"],
        args_schedule=ctx["args_schedule"],
        completed=1,  # <-- FINAL
    )

    if verbose:
        print(f"[cache miss] saved → {save_path}")

    return {
        "times": times,
        "ut": ut,
        "timestep": float(ts.getTimeStep()),
        "reason": int(reason),
        "temp_schedule": ctx["temp_schedule"],
        "args_schedule": ctx["args_schedule"],
        "cache_hit": False,
        "cache_path": save_path,
    }


# =========================
# Rates update & RHS / monitors
# =========================
def _update_rates_for_T(ctx: dict[str, Any], T: float) -> None:
    kb = 8.617333262145e-5
    T = float(T)
    prevT = ctx.get("T", None)
    if (prevT is not None) and (abs(prevT - T) < 1e-14):
        return
    ctx["T"] = T
    args = ctx["args"]
    ctx["D"] = args.diffusion_rate / (SAMPLE_LENGTH * NM_to_CM) ** 2 * np.exp(
        -args.hopping / (kb * T)
    )
    ctx["k"] = {
        "AH": args.A_detrap_attemptfreq * np.exp(-args.A_detrap / (kb * T)),
        "HA": args.A_trap_attemptfreq * np.exp(-args.A_trap / (kb * T)),
        "BH": args.B_detrap_attemptfreq * np.exp(-args.B_detrap / (kb * T)),
        "HB": args.B_trap_attemptfreq * np.exp(-args.B_trap / (kb * T)),
        "CH": args.C_detrap_attemptfreq * np.exp(-args.C_detrap / (kb * T)),
        "HC": args.C_trap_attemptfreq * np.exp(-args.C_trap / (kb * T)),
        "DH": args.D_detrap_attemptfreq * np.exp(-args.D_detrap / (kb * T)),
        "HD": args.D_trap_attemptfreq * np.exp(-args.D_trap / (kb * T)),
    }


def _initial_conditions(u: PETSc.Vec, ctx: dict[str, Any]) -> None:
    u0 = ctx.get("u0", None)

    # 1) explicit initial condition
    if u0 is not None:
        arr = np.asarray(u0, dtype=float).reshape(-1)
        if arr.size != 10:
            raise ValueError(f"u0 must be length 10 (got {arr.size})")
        u.setArray(arr.copy())
        u.assemble()
        return

    # 2) restart from history
    if ctx.get("history"):
        _, x0 = ctx["history"][-1]
        u.setArray(np.array(x0, dtype=float))
        u.assemble()
        return

    # 3) default initial condition
    u.setValues(
        range(10),
        [0.99, 1e-8, 1e-8, 1e-9, 1e-9, 1e-8, 1e-8, 1e-8, 1e-9, 1e-9],
    )
    u.assemble()

def _rhs(ts: PETSc.TS, t: float, u: PETSc.Vec, F: PETSc.Vec, ctx: dict[str, Any]) -> int:
    A, B, C, D, E, HA, HB, HC, HD, HE = u.getArray(readonly=True)
    k = ctx["k"]
    N = ctx["N"]
    D_H = ctx["D"]

    NA_avail = max(N["A"] - A, 0)
    NB_avail = max(N["B"] - B, 0)
    NC_avail = max(N["C"] - C, 0)
    ND_avail = max(N["D"] - D, 0)
    NE_avail = max(N["E"] - E, 0)

    dA = (-k["AH"] * A + k["HA"] * HA * NA_avail)
    dB = (-k["BH"] * B + k["HB"] * HB * NB_avail)
    dC = (-k["CH"] * C + k["HC"] * HC * NC_avail)
    dD = (-k["DH"] * D + k["HD"] * HD * ND_avail)
    dE = (-k["DH"] * E + k["HD"] * HE * NE_avail)

    flux_A_to_B = (HB - HA) / ctx["d_AB"]
    flux_B_to_C = (HC - HB) / ctx["d_BC"]
    flux_C_to_D = (HD - HC) / ctx["d_CD"]
    flux_D_to_E = (HE - HD) / ctx["d_DE"]

    dHA = -dA + D_H * flux_A_to_B / SCALE_A
    dHB = -dB + D_H * (flux_B_to_C - flux_A_to_B) / SCALE_B
    dHC = -dC + D_H * (flux_C_to_D - flux_B_to_C) / SCALE_C
    dHD = -dD + D_H * (flux_D_to_E - flux_C_to_D) / SCALE_D
    dHE = -dE - D_H * flux_D_to_E / SCALE_E
    
    if DEBUG and step_debug_print_allowed(ctx, t):
        print("\n--- RHS DEBUG @ t =", t, "---")
        print("State u:", u.getArray(readonly=True))
        print("Derivs:", [dA, dB, dC, dD, dE, dHA, dHB, dHC, dHD, dHE])
        print("Fluxes:", [flux_A_to_B, flux_B_to_C, flux_C_to_D, flux_D_to_E])

    F.setValues(range(10), [dA, dB, dC, dD, dE, dHA, dHB, dHC, dHD, dHE])
    F.assemble()
    return 0


# =========================
# Piecewise solve
# =========================
def _solve_piecewise(
    ts: PETSc.TS,
    u: PETSc.Vec,
    ctx: dict[str, Any],
    *,
    save_path: str,
    sim_record: dict[str, Any],
) -> None:
    history = ctx["history"]
    sched = ctx["temp_schedule"]
    t0_global = ctx["t0_global"]

    args = ctx["args"]
    # Hard-require params: if they're missing, crash loudly so we don't use shadow defaults
    boot_len = float(args.bootstrap_seconds)
    boot_dtmax = float(args.bootstrap_dt_max)
    boot_outdt = float(args.bootstrap_output_dt)
    boot_seeddt = float(args.bootstrap_seed_dt)

    for i, (t_end, Tk) in enumerate(sched):
        t_start = float(ts.getTime())
        ctx["segment_t_start"] = t_start

        # --- hot micro-window ---
        if boot_len > 0.0:
            t_boot_end = min(t_end, t_start + boot_len)

            # Denser output during the boot window
            ctx["out_dt_active"] = min(boot_outdt, ctx["out_dt"])
            ctx["next_out_t"] = _ceil_to_grid(t0_global, t_start, ctx["out_dt_active"])

            # Cap timestep and duration explicitly
            ts.setTimeStep(min(float(args.timestep), boot_seeddt))
            ts.setMaxTime(float(t_boot_end))

            # run a short solve with the smaller dt cap
            ts.solve(u)

            # Add a sample exactly at the end of boot (dedup-protected)
            t_now = float(ts.getTime())
            if (not history) or abs(history[-1][0] - t_now) > 1e-12:
                history.append((t_now, u.getArray(readonly=True).copy()))

            # clear time limit before continuing
            ts.setMaxTime(PETSc.DECIDE)

        # --- main remainder of the segment ---
        ctx["out_dt_active"] = ctx["out_dt"]
        t_now = float(ts.getTime())
        ctx["next_out_t"] = _ceil_to_grid(t0_global, t_now, ctx["out_dt_active"])

        if t_now < t_end - 1e-15:
            ts.setTimeStep(min(ts.getTimeStep(), max(args.timestep, 1e-6)))
            ts.setMaxTime(float(t_end))
            ts.solve(u)
            t_now = float(ts.getTime())
            if (not history) or abs(history[-1][0] - t_now) > 1e-12:
                history.append((t_now, u.getArray(readonly=True).copy()))

        _save_npz(
            save_path,
            times=[float(x[0]) for x in history],
            ut=[np.array(x[1], dtype=float) for x in history],
            timestep=ts.getTimeStep(),
            reason=ts.getConvergedReason(),
            sim_record=sim_record,
            temp_schedule=ctx["temp_schedule"],
            args_schedule=ctx["args_schedule"],
            completed=0,  # <-- mark as NOT final
        )

        # preload next segment rates
        if i + 1 < len(sched):
            _update_rates_for_T(ctx, float(sched[i + 1][1]))

## 
## Saving and loading 
##
def run_or_load_sim(
    *,
    rec: dict[str, Any],
    results_dir: str | None = None,
    petsc_options: dict[str, Any] | None = None,
    verbose: bool = True,
    u0: Any | None = None,      
) -> tuple[dict[str, Any], bool]:
    p = build_params(**rec)
    simrec = build_sim_record(params=p, petsc_options=petsc_options, u0=u0) 
    rd = _results_dir(results_dir)
    cached = _try_load_cached(simrec, rd)
    if cached is not None:
        if verbose:
            print(f"    ↪ Using cached result: {cached['cache_path']}")
        return cached, True
    if verbose:
        print("    ▶ Running new sim")
    res = simulate(p, petsc_options=petsc_options, results_dir=rd, verbose=False, u0=u0)
    return res, False


def _load_npz(path: str) -> dict[str, Any]:
    """Load and unbox numpy arrays from an npz file."""
    with np.load(path, allow_pickle=True) as data:
        out: dict[str, Any] = {}
        for k in data.files:
            v: Any = data[k]
            if isinstance(v, np.ndarray) and v.ndim == 0:  # unbox ANY scalar array
                try:
                    v = v.item()
                except Exception:
                    pass
            out[k] = v
        return out

def _try_load_cached(rec: dict[str, Any], results_dir: str) -> dict[str, Any] | None:
    primary, alt = _paths_for_record(rec, results_dir)

    for path in (primary, alt):
        if not os.path.exists(path):
            continue

        try:
            data = _load_npz(path)
        except Exception:
            # Corrupt/partial NPZ → treat as cache miss and try next candidate
            try: 
                os.remove(path)
            except Exception: pass
            continue
        # Require final results
        completed = int(data.get("completed", 0))
        reason = int(data.get("reason", -1))
        if completed != 1 or reason <= 0:
            continue

        saved_json: Any = data.get("sim_record_json", None)
        if isinstance(saved_json, np.ndarray) and saved_json.ndim == 0:
            try:
                saved_json = saved_json.item()
            except Exception:
                pass
        if not isinstance(saved_json, (str, np.str_)):
            continue

        try:
            saved = json.loads(saved_json)
        except Exception:
            continue

        if saved == rec:
            return {
                "times": np.array(data["times"], dtype=np.float64),
                "ut": np.array(data["ut"], dtype=np.float64),
                "timestep": float(np.array(data["timestep"], dtype=np.float32)),
                "reason": int(reason),
                "temp_schedule": data.get("temp_schedule"),
                "args_schedule": data.get("args_schedule"),
                "cache_hit": True,
                "cache_path": path,
            }

    return None


def _save_npz(
    path: str,
    *,
    times: Any,
    ut: Any,
    timestep: float,
    reason: int,
    sim_record: dict[str, Any],
    **meta: Any,) -> None:
    """Save simulation results (stacked float32 + compressed; lossless)."""
    sim_record_json = json.dumps(sim_record, sort_keys=True, separators=(",", ":"))
    times_arr = np.asarray(times, dtype=np.float64)  # keep full precision for time
    ut_arr    = np.asarray(ut,    dtype=np.float64)
    ts_arr = np.asarray(timestep, dtype=np.float32)

    np.savez_compressed(
        path,
        times=times_arr,
        ut=ut_arr,
        timestep=ts_arr,
        reason=int(reason),
        sim_record_json=np.array(sim_record_json, dtype=object),
        **meta,
    )

def _stack32(x: Any) -> np.ndarray:
    """Return a dense np.ndarray of dtype float32.
       - If x is a list/tuple of arrays → stack along axis=0
       - If x is already an array → astype(float32)
    """
    if isinstance(x, (list, tuple)):
        return np.stack([np.asarray(a, dtype=np.float32) for a in x], axis=0)
    return np.asarray(x, dtype=np.float32)


##
## parameter record building & hashing
##
def canonical_params_all(params: Any) -> dict[str, Any]:
    """Canonicalize ALL params for cache identity."""
    d = h.params_to_dict(params)
    # Ensure stable JSON: sort keys here or rely on json.dumps(sort_keys=True) later.
    return d


def state_index(layer: str, kind: str) -> int:
    return STATE_INDEX[(layer.upper(), kind)]


def _build_params_core(**overrides: Any) -> SimpleNamespace:
    """Core builder used by both fourstates and helpers (lazy imported there)."""
    merged: dict[str, Any] = {**_PARAM_DEFAULTS, **overrides}
    return SimpleNamespace(**merged)

build_params = _build_params_core  # type: ignore[assignment]

def _collect_module_constants() -> dict[str, Any]:
    """Get module-specific constants separate from general helpers."""
    return {
        "SAMPLE_LENGTH": SAMPLE_LENGTH,
        "SCALE_A": SCALE_A,
        "SCALE_B": SCALE_B,
        "SCALE_C": SCALE_C,
        "SCALE_D": SCALE_D,
        "SCALE_E": SCALE_E,
    }


def build_sim_record(
    *,
    params: Any,
    petsc_options: dict[str, Any] | None = None,
    u0: Any | None = None,                     # NEW
) -> dict[str, Any]:
    """Build a complete simulation record including all parameters and options."""
    rec = {
        "schema_version": SCHEMA_VERSION,
        "constants": h.round_constants(_collect_module_constants(), ndp=5),
        "params": canonical_params_all(params),
        "petsc_options": h.canonical_petsc_options(petsc_options),
    }

    # NEW: include initial condition in cache identity when provided
    if u0 is not None:
        arr = np.asarray(u0, dtype=np.float64).reshape(-1)
        if arr.size != 10:
            raise ValueError(f"u0 must be length 10 (got {arr.size})")
        # round a bit to avoid tiny float noise making new cache files
        rec["u0"] = [float(x) for x in np.round(arr, 15)]

    rec_json = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    return json.loads(rec_json)


def sim_signature(rec: dict[str, Any], *, length: int = 16) -> str:
    """Generate a hash signature for the simulation record."""
    payload = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


def _results_dir(path: str | None) -> str:
    """Ensure and return path to results directory."""
    p = path or _DEFAULT_RESULTS_DIR
    os.makedirs(p, exist_ok=True)
    return p


def _paths_for_record(rec: dict[str, Any], results_dir: str) -> tuple[str, str]:
    """Get primary and alternate paths for storing simulation results. The primary is a short hash-based name, and the alternate includes a longer encoded key for debugging/collision resolution."""
    sig = sim_signature(rec, length=16)
    primary = os.path.join(results_dir, f"sim_{sig}.npz")
    # also store a longer encoded key for possible collisions/debug
    enc = base64.urlsafe_b64encode(
        zlib.compress(
            json.dumps(rec, sort_keys=True, separators=(",", ":")).encode("utf-8"), 9
        )
    ).rstrip(b"=").decode("ascii")
    alt = os.path.join(results_dir, f"sim_{sig}_{enc[:48]}.npz")
    return primary, alt

##
## Helper functions
##
def _ceil_to_grid(t0: float, t: float, h: float) -> float:
    if h <= 0:
        return t
    n = np.ceil((t - t0) / h)
    return float(t0 + n * h)


def _monitor_dense(
    ts: PETSc.TS, step: int, time: float, u: PETSc.Vec, ctx: dict[str, Any]
) -> None:
    history = ctx["history"]
    out_dt = ctx["out_dt_active"]
    next_out_t = ctx["next_out_t"]

    t_now = float(time)
    dt_try = ts.getTimeStep()
    t_prev = max(ctx.get("segment_t_start", 0.0), t_now - (dt_try if dt_try > 0 else 0.0))

    if next_out_t < t_prev - 1e-12:
        next_out_t = t_prev

    eps = 1e-12
    can_interp = hasattr(ts, "interpolate")

    while next_out_t < t_now - eps:
        if can_interp:
            Utmp = u.duplicate()
            try:
                ts.interpolate(next_out_t, Utmp)
                vals = Utmp.getArray(readonly=True).copy()
            except Exception:
                vals = u.getArray(readonly=True).copy()
        else:
            vals = u.getArray(readonly=True).copy()

        if (not history) or abs(history[-1][0] - next_out_t) > 1e-12:
            history.append((float(next_out_t), vals))

        next_out_t += out_dt

    if next_out_t <= t_now + eps:
        vals_now = u.getArray(readonly=True).copy()
        if (not history) or abs(history[-1][0] - t_now) > 1e-12:
            history.append((t_now, vals_now))
        next_out_t = t_now + out_dt

    ctx["next_out_t"] = next_out_t

def step_debug_print_allowed(ctx, t, interval=1.0):
    """
    Only allow debug prints every `interval` seconds of simulated time
    to avoid overwhelming output.
    """
    last = ctx.get("_last_debug_t", -1e18)
    if t - last >= interval:
        ctx["_last_debug_t"] = t
        return True
    return False

