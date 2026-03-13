# helpers.py
# Authors: Adam Goga 
from __future__ import annotations
import json
# import hashlib
import re
from types import SimpleNamespace
import numpy as np
from typing import Dict, List, Tuple, Any, Iterable, Optional, Union

# Duration/temp parsing regexes
_DUR_RE = re.compile(r'^\s*(\d+(?:\.\d*)?)\s*([smhSMH]?)\s*$')
_TMP_RE = re.compile(r'^\s*(\d+(?:\.\d*)?)\s*([kKcC]?)\s*$')

# The keys that affect simulation dynamics and should be included in cache hashing
_CANON_KEYS = [
    "temp_schedule",
    "diffusion_rate", "hopping", "timestep", 
    "A_detrap", "B_detrap", "C_detrap", "D_detrap",
    "A_trap",   "B_trap",   "C_trap",   "D_trap",
    "A_detrap_attemptfreq", "B_detrap_attemptfreq",
    "C_detrap_attemptfreq", "D_detrap_attemptfreq", 
    "A_trap_attemptfreq",   "B_trap_attemptfreq",  
    "C_trap_attemptfreq",   "D_trap_attemptfreq",
]

# Default constants (can be overridden by callers)
DEFAULT_CONSTANTS = {
    # Schedule defaults
    'FIRE_S': 10,                # firing duration (seconds)
    'DEFAULT_FIRE_C': 677,       # default firing temp (°C)
    'DEFAULT_ANNEAL_C': 350,     # default anneal temp (°C) 
    'DEFAULT_ROOM_C': 27,        # room temp (°C)
    
    # Plot/data defaults 
    'UT_INDEX': 2,              # which ut[] column to plot
    'SCALE_Y': 1e22,           # y-axis scaling factor
    'DPI': 200,                # plot resolution
    
    # Stage indices
    'STAGE_ANN': 1,            # anneal stage index (no room)
    'STAGE_ANN_WITH_ROOM': 2   # anneal stage index (with room)
}

# Schedule parsing and building functions
def parse_duration(s: str) -> float:
    """Parse a duration string with optional unit (s/m/h) into seconds."""
    m = _DUR_RE.match(s)
    if not m:
        raise ValueError(f"Bad duration '{s}'")
    val, unit = float(m.group(1)), m.group(2).lower()
    if unit in ('', 's'): 
        return val
    if unit == 'm': 
        return val * 60.0
    if unit == 'h': 
        return val * 3600.0
    raise ValueError(f"Bad duration unit in '{s}'")

def parse_temp(s: str) -> float:
    """Parse a temperature string with optional unit (K/C) into Kelvin."""
    m = _TMP_RE.match(s)
    if not m:
        raise ValueError(f"Bad temperature '{s}'")
    val, unit = float(m.group(1)), m.group(2).lower()
    return val if unit in ('', 'k') else (val + 273.15)  # C -> K

def parse_temp_schedule(spec: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Parse a temperature schedule string into relative and cumulative segments.
    
    Format: "duration1:temp1, duration2:temp2, ..."
    - Durations can have units: s (seconds), m (minutes), h (hours)
    - Temps can have units: K (Kelvin) or C (Celsius)
    
    Returns:
        relative: List of (duration, temp_K) pairs 
        cumulative: List of (end_time, temp_K) pairs with absolute times
    """
    segs = []
    for part in (p for p in spec.split(',') if p.strip()):
        dur_s, temp_s = [x.strip() for x in part.split(':', 1)]
        segs.append((parse_duration(dur_s), parse_temp(temp_s)))
    
    # Build cumulative schedule with absolute times
    cum, t = [], 0.0
    for dur, Tk in segs:
        t += dur
        cum.append((t, Tk))
    return segs, cum  # (relative, cumulative)

def T_of_time(t: float, schedule_abs: List[Tuple[float, float]]) -> float:
    """Get temperature at time t from absolute schedule [(t_end, T_K), ...]."""
    for t_end, T in schedule_abs:
        if t <= t_end + 1e-15:
            return T
    return schedule_abs[-1][1]

def build_schedule(
    fire_C: int,
    anneal_C: int,
    fire_s: int,
    anneal_s: int,
    include_room: bool = False,
    room_C: int = 0,
    room_s: int = 0,
    *,
    n_cycles: int = 1,   # NEW: repeat the whole pattern this many times
) -> str:
    """
    Build a temperature schedule string (e.g., '10:700C, 1e6:350C').

    Durations are relative (seconds) and converted to cumulative internally.

    If n_cycles > 1, the (firing → [room] → anneal) pattern is repeated n_cycles
    times in sequence, with the same durations and temperatures each cycle.
    """
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")

    # One cycle: fire → [room] → anneal
    cycle_parts: list[str] = [f"{int(fire_s)}:{int(fire_C)}C"]
    if include_room and room_s > 0:
        cycle_parts.append(f"{int(room_s)}:{int(room_C)}C")
    cycle_parts.append(f"{int(anneal_s)}:{int(anneal_C)}C")

    # Repeat the cycle n_cycles times
    parts = cycle_parts * int(n_cycles)

    return ", ".join(parts)


def parse_float_list(s: str) -> list[float]:
    """Parse comma-separated floats into a list."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated ints into a list."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def coerce_numeric(s: Any) -> int | float:
    """Try to coerce a value to int first, then float, else return as-is."""
    try:
        i = int(s)
        return i
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

# ---------------------------------------------------------------------
# Filename formatting and stemming
# ---------------------------------------------------------------------

def make_stem_compact(rec: dict, keys: list[str]) -> str:
    """Make a compact filename stem from a record, focusing on the given keys."""
    parts = []
    for k in keys:
        if k not in rec:
            continue
        v = rec[k]
        key_str = short_param_tag(k)
        val_str = format_value_for_dir(k, v)
        parts.append(f"{key_str}={val_str}")
    return "__".join(parts) if parts else "default"

# ---------------------------------------------------------------------
# Parameters processing and hashing
# ---------------------------------------------------------------------

def fmt_float12g(x: Union[float, int]) -> Union[float, str]:
    """
    Return a JSON-safe canonical representation for floats (<=12 significant digits).
    For hashing stability we stringify big/small magnitudes, otherwise keep as float.
    """
    if not isinstance(x, (float, int, np.floating)):
        return x
    s = f"{float(x):.12g}"
    # Keep normal numbers as float to avoid dtype=object arrays everywhere
    try:
        val = float(s)
        return val
    except Exception:
        return s

def params_to_dict(params: Any) -> Dict[str, Any]:
    """Convert parameters (SimpleNamespace or dict) to a normalized dictionary."""
    try:
        d = vars(params).copy()
    except Exception:
        d = dict(params)
    out = {}
    for k, v in d.items():
        if hasattr(v, "item") and callable(getattr(v, "item", None)):
            try: v = v.item()
            except Exception: pass
        if isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = fmt_float12g(float(v))
        else:
            out[k] = v
    return out

# def canonical_params_view(params: Any) -> Dict[str, Any]:
#     """
#     Extract canonical simulation parameters that affect dynamics.
#     Formats numeric values consistently for stable hashing.
#     """
#     d = params_to_dict(params)
#     canon = {}
#     for k in _CANON_KEYS:
#         if k in d:
#             v = d[k]
#             if isinstance(v, (float, np.floating)):
#                 canon[k] = fmt_float12g(v)
#             else:
#                 canon[k] = v
#     return canon

# def param_signature(params: Any, *, length: int = 12) -> str:
#     """Generate a stable hash signature for simulation parameters."""
#     payload = json.dumps(canonical_params_view(params), sort_keys=True, separators=(",", ":"))
#     return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]

def round_constants(d: dict, ndp: int = 5) -> dict:
    """Round float values in a dict to ndp decimal places."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            out[k] = float(f"{v:.{ndp}f}")
        else:
            out[k] = v
    return out

def collect_constants() -> Dict[str, Any]:
    """Get simulation geometry/scaling constants in a versioned format."""
    return {
        "FIRE_S": DEFAULT_CONSTANTS["FIRE_S"],
        "DEFAULT_FIRE_C": DEFAULT_CONSTANTS["DEFAULT_FIRE_C"],
        "DEFAULT_ANNEAL_C": DEFAULT_CONSTANTS["DEFAULT_ANNEAL_C"],
        "DEFAULT_ROOM_C": DEFAULT_CONSTANTS["DEFAULT_ROOM_C"],
        "UT_INDEX": DEFAULT_CONSTANTS["UT_INDEX"],
        "SCALE_Y": DEFAULT_CONSTANTS["SCALE_Y"],
        "DPI": DEFAULT_CONSTANTS["DPI"],
        #OLD
        # "STAGE_ANN": DEFAULT_CONSTANTS["STAGE_ANN"],
        # "STAGE_ANN_WITH_ROOM": DEFAULT_CONSTANTS["STAGE_ANN_WITH_ROOM"]
    }
    
    
# def _thin_xy(x, y, maxpts=1e12):
#     """Downsample x,y to at most maxpts points (float32)."""
#     x = np.asarray(x, dtype=np.float32)
#     y = np.asarray(y, dtype=np.float32)
#     n = x.size
#     if n <= maxpts:
#         return x, y
#     idx = np.linspace(0, n - 1, maxpts, dtype=int)
#     return x[idx], y[idx]


# ---------------------------------------------------------------------
# Pretty tags / titles / file-safe formatting
# ---------------------------------------------------------------------

def short_param_tag(name: str) -> str:
    """
    Compact, human-friendly short code:
      - detrap energies: AH, BH, CH, DH
      - detrap attempt freq: kAH, kBH, kCH, kDH
      - trap attempt freq:   kHA, kHB, kHC, kHD
    Fall back to the original name if unknown.
    """
    n = name.lower()
    letter = None
    for L in ("A","B","C","D"):
        if n.startswith(L.lower()+"_"):
            letter = L
            break

    if "attemptfreq" in n:
        if "detrap" in n:
            return f"k{letter or name[:1].upper()}H"
        if "trap" in n:
            return f"kH{letter or name[:1].upper()}"
        return f"k{letter or name[:1].upper()}"
    if "detrap" in n:
        return f"{letter or name[:1].upper()}H"
    return name


def format_value_for_title(name: str, v: Any) -> str:
    """Readable values for figure titles."""
    try:
        x = float(v)
    except Exception:
        return str(v)
    if "attemptfreq" in name.lower():
        return f"{x:.0e}".replace("e+","e")
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s


def format_value_for_dir(param_key: str, v: Any) -> str:
    """
    File-system-safe, stable token:
      - Floats → compact decimal, '.' -> '_', unless large (>=1e6) or *_attempt/freq → scientific 1eN
      - Ints   → as int
      - Str    → lowercased, spaces removed, '.' -> '_'
    """
    if isinstance(v, str):
        s = v.strip().lower().replace(' ', '')
        return s.replace('.', '_')
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        key_l = str(param_key).lower()
        if abs(float(v)) >= 1e6 or 'attempt' in key_l or 'freq' in key_l:
            s = f"{float(v):.0e}".replace('+', '')
            return s
        s = f"{float(v):g}"
        return s.replace('.', '_')
    return str(v).replace('.', '_')

def schedule_tag(s: str) -> str:
    """Turn '10:700C, 1e6:350C' into '10-700C__1e6-350C'."""
    parts = [p.strip().replace(":", "-").replace(" ", "") for p in s.split(",") if p.strip()]
    return "__".join(parts) if parts else "nosched"

def canonical_petsc_options(opts: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Convert PETSc options dict to canonical string form for hashing."""
    if not opts:
        return {}
    # stringify for hashing determinism
    return {str(k): str(v) for k, v in sorted(opts.items(), key=lambda kv: kv[0])}

# ---------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------

def build_params(**overrides):
    """Thin wrapper that delegates to fourstates' single source of truth."""
    from fourstates import build_params # lazy import avoids circularity
    return build_params(**overrides)

# ---------------------------------------------------------------------
# Time units and axes handling
# ---------------------------------------------------------------------

# def x_unit_and_scale(total_seconds: float, *, force: Optional[str] = None) -> Tuple[str, float]:
#     """
#     Decide time units for x-axis.
#       - force='hours'  → ('hours', 3600.0)
#       - force='minutes'→ ('minutes', 60.0)
#       - None: pick by range (>=3600 s → hours; >=120 s → minutes; else seconds)
#     """
#     if force:
#         f = force.lower()
#         if f == "hours":
#             return "hours", 3600.0
#         if f == "minutes":
#             return "minutes", 60.0
#         if f == "seconds":
#             return "seconds", 1.0
#         raise ValueError(f"Unknown force='{force}'")
#     if total_seconds >= 3600:
#         return "hours", 3600.0
#     if total_seconds >= 120:
#         return "minutes", 60.0
#     return "seconds", 1.0

# ---------------------------------------------------------------------
# Stage extraction for dict results returned by simulate()
# ---------------------------------------------------------------------

def _infer_anneal_stage_index(result: Dict[str, Any], include_room: Optional[bool]) -> int:
    """
    If include_room is None: infer 2 if a room stage (≈ 300K) exists; else 1.
    If include_room is True/False: return 2/1 respectively.
    """
    sched = list(result.get("temp_schedule") or [])
    if include_room is not None:
        return 2 if include_room else 1
    if not sched:
        return 1
    # Treat ~room temp segments (260–320 K) as a "room" stage
    has_room = any((260.0 <= float(Tk) <= 320.0) for _, Tk in sched)
    return 2 if has_room else 1

def extract_stage_curves(result, *, ut_index=2, stage_index=1, include_room=None, scale=1.0):
    import numpy as _np
    # print('calling this version')
    times = _np.asarray(result.get("times", []), dtype=float)
    if times.size == 0:
        return _np.array([]), _np.array([]), 0.0

    ut = _np.asarray(result.get("ut", []), dtype=float)
    if ut.ndim == 1:
        ut = ut[None, :]
    if ut.shape[0] != times.shape[0]:
        return _np.array([]), _np.array([]), 0.0

    sched_raw = result.get("temp_schedule", None)
    sched_list = (sched_raw.tolist() if isinstance(sched_raw, _np.ndarray) else list(sched_raw or []))

    bounds, t_prev = [], 0.0
    for t_end, _Tk in sched_list:
        t_end = float(t_end); bounds.append((t_prev, t_end)); t_prev = t_end
    t_last = float(times[-1])
    if not bounds:
        bounds = [(0.0, t_last)]
    elif bounds[-1][1] < t_last:
        bounds.append((bounds[-1][1], t_last))

    if stage_index < 0 or stage_index >= len(bounds):
        return _np.array([]), _np.array([]), 0.0

    t0, t1 = bounds[stage_index]
    if stage_index == 0:
        i0 = int(_np.searchsorted(times, t0, side="left"))
        i1 = int(_np.searchsorted(times, t1, side="right"))
    else:
        i0 = int(_np.searchsorted(times, t0, side="right"))
        i1 = int(_np.searchsorted(times, t1, side="right"))

    i0 = max(0, min(i0, times.size))
    i1 = max(i0, min(i1, times.size))
    if i1 <= i0:
        return _np.array([]), _np.array([]), float(t1 - t0)

    t_rel = times[i0:i1] - t0
    if t_rel.size:
        # normalize first plotted sample to exactly 0
        t_rel = _np.maximum(t_rel - t_rel[0], 0.0)

    y = ut[i0:i1, ut_index] * float(scale)
    seg_len = float(t1 - t0)
    return t_rel, y, seg_len




def baseline_value(t_h, y, *, t_min=0.1, t_max=0.3):
    """
    Estimate the 'anneal-start plateau' as the median in a short early window.
    Times in hours. For sims, pass t_h = t_sec/3600 (stage='annealing' starts at 0).
    """
    t = np.asarray(t_h, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y) & (t >= t_min) & (t <= t_max)
    if not np.any(m):
        # fallback: first few valid points
        m = np.isfinite(t) & np.isfinite(y)
        idx = np.where(m)[0][:max(3, int(0.02*len(t)))]
        return float(np.median(y[idx])) if idx.size else np.nan
    return float(np.median(y[m]))

def normalize_baseline_peak(
    t_h,
    y,
    *,
    exp: bool = False,
    sim_baseline_idx: int = 10,
    exp_baseline_idx: int = 1,
    t0_window: tuple[float, float] | None = (0.1, 0.3),
    plateau_frac: float = 0.03,      # plateau = values <= plateau_frac * y_peak (pre-peak)
    min_plateau_pts: int = 5,
    **_ignored,
):
    """
    Normalize series using the MEAN of a plateau region and the global max as peak.

      Primary baseline (preferred):
        - Use all pre-peak samples with y <= plateau_frac * y_peak
        - Baseline y0 = mean of those samples

      Fallbacks:
        1) If too few plateau points, average samples in t0_window (hours),
           restricted to pre-peak if possible.
        2) If still too few, use the *first* min_plateau_pts finite samples.
        3) Last resort: the old fixed-index baseline (exp_baseline_idx/sim_baseline_idx).

      Returns: (t, yN, y0, y_peak, i_peak)
        where yN = (y - y0) / (y_peak - y0)
    """
    import numpy as np

    t = np.asarray(t_h, float)
    y = np.asarray(y, float)

    if y.size == 0:
        return t, y.astype(float), float("nan"), float("nan"), -1

    # Handle all-NaN quickly
    if not np.isfinite(y).any():
        return t, y.astype(float), float("nan"), float("nan"), -1

    # Peak
    i_peak = int(np.nanargmax(y))
    y_peak = float(y[i_peak])

    # --------- Primary plateau (pre-peak & small amplitude) ----------
    pre_mask = (np.arange(y.size) <= i_peak) & np.isfinite(y)
    small_mask = y <= plateau_frac * y_peak
    plateau_mask = pre_mask & small_mask
    if int(np.sum(plateau_mask)) >= min_plateau_pts:
        y0 = float(np.mean(y[plateau_mask]))
    else:
        # --------- Fallback 1: average in time window (prefer pre-peak) ----------
        y0 = None
        if t0_window is not None and np.isfinite(t).any():
            tmin, tmax = t0_window
            time_mask = (
                np.isfinite(t) & np.isfinite(y) &
                (t >= float(tmin)) & (t <= float(tmax))
            )
            # prefer pre-peak samples in the window; otherwise any in window
            time_pre = time_mask & pre_mask
            if int(np.sum(time_pre)) >= max(3, min_plateau_pts // 2):
                y0 = float(np.mean(y[time_pre]))
            elif int(np.sum(time_mask)) >= max(3, min_plateau_pts // 2):
                y0 = float(np.mean(y[time_mask]))

        # --------- Fallback 2: first min_plateau_pts finite samples ----------
        if y0 is None:
            idx = np.where(np.isfinite(y))[0][:min_plateau_pts]
            if idx.size >= 1:
                y0 = float(np.mean(y[idx]))
            else:
                y0 = float(np.nanmean(y))  # absolute last resort

        # --------- Fallback 3: old fixed-index rule ----------
        if not np.isfinite(y0):
            idx_fixed = (exp_baseline_idx if exp else sim_baseline_idx)
            idx_fixed = int(np.clip(idx_fixed, 0, y.size - 1))
            y0 = float(y[idx_fixed])

    # Normalize
    span = float(max(y_peak - y0, 1e-300))  # avoid divide-by-zero
    yN = (y - y0) / span

    return t, yN.astype(float), float(y0), float(y_peak), i_peak




def time_to_return_baseline(t_h, yN, ipeak, *, tol=0.05):
    """
    Time (hours) from the peak until the curve returns within ±tol of baseline (i.e., yN <= tol).
    If it never returns, uses the last time.
    """
    t = np.asarray(t_h, float); yN = np.asarray(yN, float)
    tr = t[ipeak:]; yr = yN[ipeak:]
    if tr.size == 0: return np.nan
    idx = np.where(yr <= tol)[0]
    return float(tr[idx[0]] - tr[0]) if idx.size else float(tr[-1] - tr[0])

def peak_time_hours(t_h, ipeak):
    return float(np.asarray(t_h, float)[ipeak])


# ---------------------------------------------------------------------
# Minimal ctx + RHS reconstruction helpers for postprocessing
# ---------------------------------------------------------------------

def make_minimal_ctx(params, fs=None):
    """
    Construct the minimal ctx required for fourstates._rhs.
    This ctx is temperature-agnostic; T-dependent parts are filled later
    by fs._update_rates_for_T(ctx, T).

    This reproduces ONLY the fields that _rhs actually reads.
    """
    if fs is None:
        import fourstates as fs

    ctx = {
        "args": params,            # full parameter namespace
        "k": {},                   # rate constants (filled by update_rates_for_T)
        "D": 0.0,                  # diffusion coefficient (filled by update_rates_for_T)

        # Trap densities (fixed solver constants)
        "N": {
        "A": float(params.N_A),
        "B": float(params.N_B),
        "C": float(params.N_C),
        "D": float(params.N_D),
        "E": float(params.N_E),
    },

        # Geometry distances used in flux computation
        "d_AB": fs.SCALE_A/2 + fs.SCALE_B/2,
        "d_BC": fs.SCALE_B/2 + fs.SCALE_C/2,
        "d_CD": fs.SCALE_C/2 + fs.SCALE_D/2,
        "d_DE": fs.SCALE_D/2 + fs.SCALE_E/2,
    }
    return ctx


def rhs_from_state(u_vec, params, T, *, fs=None):
    """
    Recompute d(u)/dt and interface fluxes for a single state u at temperature T.

    Uses the *actual* solver's _rhs, with a minimal reconstructed ctx and
    fake PETSc Vec wrappers.

    Inputs:
        u_vec  : np.array length 10 (A,B,C,D,E, HA,HB,HC,HD,HE)
        params : the simulation parameter namespace
        T      : temperature in Kelvin

    Returns:
        derivs : np.array length 10, d(u)/dt
        fluxes : np.array length 4, [flux_AB, flux_BC, flux_CD, flux_DE]
    """
    if fs is None:
        import fourstates as fs

    import numpy as _np

    # ---- 1. Build minimal context and update rates for this temperature ----
    ctx = make_minimal_ctx(params, fs=fs)
    fs._update_rates_for_T(ctx, float(T))

    # ---- 2. Fake PETSc Vec wrappers ----
    class FakeVec:
        def __init__(self, arr):
            self._arr = _np.array(arr, float)
        def getArray(self, readonly=False):
            return self._arr
        def setValues(self, idxs, vals):
            for i, v in zip(idxs, vals):
                self._arr[int(i)] = v
        def assemble(self): pass

    u_fake = FakeVec(u_vec)
    F_fake = FakeVec(_np.zeros_like(u_vec))

    # ---- 3. Call the solver RHS (PETSc TS normally does this) ----
    fs._rhs(None, 0.0, u_fake, F_fake, ctx)

    derivs = F_fake.getArray().copy()

        # ---- 4. Compute interface fluxes in *physical units* (atoms / cm^2 / s) ----
    # Raw model-domain mobile concentrations:
    A, B, C, D, E, HA, HB, HC, HD, HE = _np.asarray(u_vec, float)

    # Raw model-domain flux (dimensionless)
    flux_AB_raw = - (HB - HA) / ctx["d_AB"]
    flux_BC_raw = - (HC - HB) / ctx["d_BC"]
    flux_CD_raw = - (HD - HC) / ctx["d_CD"]
    flux_DE_raw = - (HE - HD) / ctx["d_DE"]

    # Physical diffusion coefficient (cm^2/s)
    D_H = ctx["D"]

    # Convert normalized layer fractions into actual physical thickness (cm)
    # 1 nm = 1e-7 cm, 1 μm = 1e-4 cm.
    dxA = fs.SCALE_A * fs.SAMPLE_LENGTH * fs.NM_to_CM     # nm → cm
    dxB = fs.SCALE_B * fs.SAMPLE_LENGTH * fs.NM_to_CM
    dxC = fs.SCALE_C * fs.SAMPLE_LENGTH * fs.NM_to_CM
    dxD = fs.SCALE_D * fs.SAMPLE_LENGTH * fs.NM_to_CM
    dxE = fs.SCALE_E * 1e-4                         # μm → cm

    # Interface midpoint distances (physical Δx)
    dx_AB = 0.5 * (dxA + dxB)
    dx_BC = 0.5 * (dxB + dxC)
    dx_CD = 0.5 * (dxC + dxD)
    dx_DE = 0.5 * (dxD + dxE)

    # True physical flux (atoms / cm^2 / s)
    flux_AB = D_H * flux_AB_raw / dx_AB
    flux_BC = D_H * flux_BC_raw / dx_BC
    flux_CD = D_H * flux_CD_raw / dx_CD
    flux_DE = D_H * flux_DE_raw / dx_DE

    fluxes = _np.array([flux_AB, flux_BC, flux_CD, flux_DE], float)

    return derivs, fluxes
