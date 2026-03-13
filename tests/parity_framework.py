"""Shared parity framework utilities.

This module centralizes legacy/new runner adapters, trace extraction,
interpolation/comparison metrics, and baseline source abstractions used by
all parity tests.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hdiff.schedule import Sampling, Schedule, Segment
from hdiff.sim import Simulation, SolverConfig
from hdiff.defaults import DEFAULT_SAMPLING, DEFAULT_SOLVER, DEFAULT_STRUCTURE, DEFAULT_Y0
from hdiff.structure import (
    Arrhenius,
    BoundaryCondition,
    Layer,
    Material,
    Structure,
    Transport,
    TrapSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = REPO_ROOT / "legacy"

STATE_ORDER = [
    ("trapped", "A", "t1"),
    ("trapped", "B", "t1"),
    ("trapped", "C", "t1"),
    ("trapped", "D", "t1"),
    ("trapped", "E", "t1"),
    ("mobile", "A", None),
    ("mobile", "B", None),
    ("mobile", "C", None),
    ("mobile", "D", None),
    ("mobile", "E", None),
]


@dataclass(frozen=True)
class MeasureSpec:
    layer: str
    kind: str
    trap_id: str | None = None
    stage: str | None = "annealing"
    occurrence: int = 0
    units: str = "cm^-3"


@dataclass(frozen=True)
class ParityCase:
    name: str
    fire_C: float
    anneal_C: float
    fire_s: float
    anneal_s: float
    threshold_rel_l2: float = 0.50
    threshold_rel_linf: float = 0.60


@dataclass(frozen=True)
class SeriesTrace:
    t_s: np.ndarray
    y: np.ndarray
    source: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ComparisonMetrics:
    rel_l2: float
    rel_linf: float
    max_abs: float
    n_samples: int


class BaselineSource(Protocol):
    def get_trace(self, case: ParityCase, measure: MeasureSpec) -> SeriesTrace:
        ...


def _require_legacy_imports() -> tuple[Any, Any]:
    if str(LEGACY_ROOT) not in sys.path:
        sys.path.insert(0, str(LEGACY_ROOT))
    import fourstates as legacy_fs
    import simulation_manager as legacy_sm

    return legacy_fs, legacy_sm


def paper_param_set() -> dict[str, float]:
    tA = DEFAULT_STRUCTURE.materials["alox"].traps[0]
    tB = DEFAULT_STRUCTURE.materials["poly_si"].traps[0]
    tC = DEFAULT_STRUCTURE.materials["siox"].traps[0]
    tD = DEFAULT_STRUCTURE.materials["csi"].traps[0]

    return {
        "A_detrap": float(tA.detrap_kin.Ea_eV),
        "B_detrap": float(tB.detrap_kin.Ea_eV),
        "C_detrap": float(tC.detrap_kin.Ea_eV),
        "D_detrap": float(tD.detrap_kin.Ea_eV),
        "A_trap": float(tA.trap_kin.Ea_eV),
        "B_trap": float(tB.trap_kin.Ea_eV),
        "C_trap": float(tC.trap_kin.Ea_eV),
        "D_trap": float(tD.trap_kin.Ea_eV),
        "A_detrap_attemptfreq": float(tA.detrap_kin.nu),
        "B_detrap_attemptfreq": float(tB.detrap_kin.nu),
        "C_detrap_attemptfreq": float(tC.detrap_kin.nu),
        "D_detrap_attemptfreq": float(tD.detrap_kin.nu),
        "A_trap_attemptfreq": float(tA.trap_kin.nu),
        "B_trap_attemptfreq": float(tB.trap_kin.nu),
        "C_trap_attemptfreq": float(tC.trap_kin.nu),
        "D_trap_attemptfreq": float(tD.trap_kin.nu),
    }


def full_params(base_params: dict[str, float]) -> dict[str, float]:
    legacy_fs, _ = _require_legacy_imports()
    return dict(vars(legacy_fs.build_params(**base_params)))


def legacy_like_structure(p: dict[str, float]) -> Structure:
    def mk_trap(
        capacity_key: str,
        trap_Ea_key: str,
        detrap_Ea_key: str,
        trap_nu_key: str,
        detrap_nu_key: str,
    ) -> TrapSpec:
        return TrapSpec(
            id="t1",
            trap_density=float(p[capacity_key]),
            trap_kin=Arrhenius(nu=float(p[trap_nu_key]), Ea_eV=float(p[trap_Ea_key])),
            detrap_kin=Arrhenius(
                nu=float(p[detrap_nu_key]),
                Ea_eV=float(p[detrap_Ea_key]),
            ),
        )

    materials = {
        "mA": Material("mA", [mk_trap("N_A", "A_trap", "A_detrap", "A_trap_attemptfreq", "A_detrap_attemptfreq")]),
        "mB": Material("mB", [mk_trap("N_B", "B_trap", "B_detrap", "B_trap_attemptfreq", "B_detrap_attemptfreq")]),
        "mC": Material("mC", [mk_trap("N_C", "C_trap", "C_detrap", "C_trap_attemptfreq", "C_detrap_attemptfreq")]),
        "mD": Material("mD", [mk_trap("N_D", "D_trap", "D_detrap", "D_trap_attemptfreq", "D_detrap_attemptfreq")]),
        "mE": Material("mE", [mk_trap("N_E", "D_trap", "D_detrap", "D_trap_attemptfreq", "D_detrap_attemptfreq")]),
    }

    layers = [
        Layer(name="A", thickness_cm=100e-7, material_id="mA"),
        Layer(name="B", thickness_cm=290e-7, material_id="mB"),
        Layer(name="C", thickness_cm=1.5e-7, material_id="mC"),
        Layer(name="D", thickness_cm=608.5e-7, material_id="mD"),
        Layer(name="E", thickness_cm=50e-4, material_id="mE"),
    ]

    transport_prefactor = float(p["diffusion_rate"])
    return Structure(
        materials=materials,
        layers=layers,
        bc=BoundaryCondition(kind="closed_closed", params={}),
        transport=Transport(prefactor=transport_prefactor, hop_Ea_eV=float(p["hopping"])),
        conc_scale=1e22,
    )


def solver_config(*, verbose: bool = False) -> SolverConfig:
    if verbose:
        from dataclasses import replace
        return replace(DEFAULT_SOLVER, verbose=True)
    return DEFAULT_SOLVER


def sampling_config() -> Sampling:
    return DEFAULT_SAMPLING


def legacy_initial_state() -> list[float]:
    return list(DEFAULT_Y0)


def schedule_two_stage(fire_C: float, anneal_C: float, fire_s: float, anneal_s: float) -> Schedule:
    return Schedule(
        segments=[
            Segment(duration_s=float(fire_s), stage="firing", T_K=float(fire_C) + 273.15),
            Segment(duration_s=float(anneal_s), stage="annealing", T_K=float(anneal_C) + 273.15),
        ]
    )


def schedule_single_stage(*, stage: str, temp_C: float, duration_s: float) -> Schedule:
    return Schedule(
        segments=[
            Segment(duration_s=float(duration_s), stage=str(stage), T_K=float(temp_C) + 273.15),
        ]
    )


def stage_window(times_s: np.ndarray, durations_s: list[float], stage_idx: int) -> tuple[np.ndarray, float]:
    if stage_idx < 0 or stage_idx >= len(durations_s):
        raise ValueError(f"invalid stage_idx={stage_idx} for durations size={len(durations_s)}")
    start = float(sum(durations_s[:stage_idx]))
    end = start + float(durations_s[stage_idx])
    mask = (times_s >= start) & (times_s <= end)
    idx = np.nonzero(mask)[0]
    return idx, start


def _state_index(layer: str, kind: str, trap_id: str | None) -> int | None:
    for idx, (state_kind, state_layer, state_trap) in enumerate(STATE_ORDER):
        if state_layer != layer:
            continue
        if kind == "mobile" and state_kind == "mobile":
            return idx
        if kind == "trapped" and state_kind == "trapped" and (trap_id is None or trap_id == state_trap):
            return idx
    return None


def extract_series_from_state_matrix(
    *,
    t_s: np.ndarray,
    y_state: np.ndarray,
    measure: MeasureSpec,
    stage_durations_s: list[float] | None = None,
    conc_scale: float = 1e22,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(t_s, dtype=float)
    states = np.asarray(y_state, dtype=float)
    if states.ndim != 2:
        raise ValueError("y_state must be 2D")
    if states.shape[0] != times.shape[0]:
        raise ValueError("t_s and y_state must align on the first axis")

    if measure.stage is not None:
        if stage_durations_s is None:
            raise ValueError("stage_durations_s is required when measure.stage is provided")
        stage_lookup = {"firing": 0, "annealing": 1}
        if measure.stage not in stage_lookup:
            raise ValueError(f"unsupported stage name: {measure.stage}")
        idx, t0 = stage_window(times, stage_durations_s, stage_lookup[measure.stage])
        times = times[idx] - t0
        states = states[idx]

    if measure.kind == "mobile":
        state_idx = _state_index(measure.layer, "mobile", None)
        if state_idx is None:
            raise ValueError("mobile state index not found")
        values = states[:, state_idx]
    elif measure.kind == "trapped":
        state_idx = _state_index(measure.layer, "trapped", measure.trap_id)
        if state_idx is None:
            raise ValueError("trapped state index not found")
        values = states[:, state_idx]
    elif measure.kind == "total":
        mobile_idx = _state_index(measure.layer, "mobile", None)
        trapped_idx = _state_index(measure.layer, "trapped", measure.trap_id)
        if mobile_idx is None or trapped_idx is None:
            raise ValueError("total state indices not found")
        values = states[:, mobile_idx] + states[:, trapped_idx]
    else:
        raise ValueError(f"unsupported kind: {measure.kind}")

    if measure.units == "cm^-3":
        values = values * conc_scale
    elif measure.units != "solver":
        raise ValueError(f"unsupported units: {measure.units}")

    return times, values


class LegacyRuntimeBaselineSource:
    def __init__(self, *, base_params: dict[str, float] | None = None, results_dir: Path | None = None, verbose: bool = False) -> None:
        self.base_params = paper_param_set() if base_params is None else dict(base_params)
        self.results_dir = REPO_ROOT / "sim_data" if results_dir is None else Path(results_dir)
        self.verbose = verbose

    def get_trace(self, case: ParityCase, measure: MeasureSpec) -> SeriesTrace:
        import time as _time
        _, legacy_sm = _require_legacy_imports()
        schedule = f"{int(case.fire_s)}:{int(case.fire_C)}C, {int(case.anneal_s)}:{int(case.anneal_C)}C"
        if self.verbose:
            print(f"[legacy] '{case.name}'  loading/running: {schedule}", flush=True)
        _t0 = _time.monotonic()
        mgr = legacy_sm.SimulationManager(
            base_params=self.base_params,
            temp_schedules=[schedule],
            stage_names=["firing", "annealing"],
            results_dir=str(self.results_dir),
            verbose=False,
        )
        res = mgr._results[0]
        t_stage, y_stage = res.series(
            layer=measure.layer,
            kind=measure.kind,
            stage=measure.stage,
        )
        _elapsed = _time.monotonic() - _t0
        if self.verbose:
            print(f"[legacy] '{case.name}' done: {len(t_stage)} pts in {_elapsed:.1f}s", flush=True)
        return SeriesTrace(
            t_s=np.asarray(t_stage, dtype=float),
            y=np.asarray(y_stage, dtype=float),
            source="legacy_runtime",
            metadata={"schedule": schedule},
        )


def run_legacy_runtime_trace(
    *,
    schedule_spec: str,
    stage_names: list[str],
    measure: MeasureSpec,
    base_params: dict[str, float] | None = None,
    results_dir: Path | None = None,
) -> SeriesTrace:
    _, legacy_sm = _require_legacy_imports()
    root = REPO_ROOT / "sim_data" if results_dir is None else Path(results_dir)
    params = {} if base_params is None else dict(base_params)
    mgr = legacy_sm.SimulationManager(
        base_params=params,
        temp_schedules=[schedule_spec],
        stage_names=stage_names,
        results_dir=str(root),
        verbose=False,
    )
    res = mgr._results[0]
    if measure.kind == "total":
        t_mobile, y_mobile = res.series(
            layer=measure.layer,
            kind="mobile",
            stage=measure.stage,
        )
        t_trapped, y_trapped = res.series(
            layer=measure.layer,
            kind="trapped",
            stage=measure.stage,
        )
        t_mobile_arr = np.asarray(t_mobile, dtype=float)
        t_trapped_arr = np.asarray(t_trapped, dtype=float)
        if t_mobile_arr.shape != t_trapped_arr.shape or not np.allclose(t_mobile_arr, t_trapped_arr):
            raise ValueError("legacy mobile/trapped timebases do not align for total series")
        t_stage = t_mobile_arr
        y_stage = np.asarray(y_mobile, dtype=float) + np.asarray(y_trapped, dtype=float)
    else:
        t_stage, y_stage = res.series(
            layer=measure.layer,
            kind=measure.kind,
            stage=measure.stage,
        )
    return SeriesTrace(
        t_s=np.asarray(t_stage, dtype=float),
        y=np.asarray(y_stage, dtype=float),
        source="legacy_runtime",
        metadata={"schedule": schedule_spec, "stage_names": list(stage_names)},
    )


class LegacySeriesNpzBaselineSource:
    def __init__(self, *, baselines_root: Path, index_file: Path | None = None) -> None:
        self.baselines_root = Path(baselines_root)
        self.index_file = Path(index_file) if index_file is not None else self.baselines_root / "index.json"
        if not self.index_file.exists():
            raise FileNotFoundError(f"baseline index not found: {self.index_file}")
        self.index = json.loads(self.index_file.read_text())

    def get_trace(self, case: ParityCase, measure: MeasureSpec) -> SeriesTrace:
        del measure
        rel = self.index.get(case.name)
        if rel is None:
            raise KeyError(f"case not found in baseline index: {case.name}")
        p = self.baselines_root / rel
        data = np.load(p, allow_pickle=True)
        meta_raw = data["metadata_json"].item() if "metadata_json" in data.files else "{}"
        metadata = json.loads(meta_raw)
        return SeriesTrace(
            t_s=np.asarray(data["t_s"], dtype=float),
            y=np.asarray(data["y"], dtype=float),
            source="legacy_series_npz",
            metadata=metadata,
        )


class LegacyCacheNpzBaselineSource:
    def __init__(self, *, file_map: dict[str, Path], conc_scale: float = 1e22) -> None:
        self.file_map = {k: Path(v) for k, v in file_map.items()}
        self.conc_scale = float(conc_scale)

    def get_trace(self, case: ParityCase, measure: MeasureSpec) -> SeriesTrace:
        p = self.file_map.get(case.name)
        if p is None:
            raise KeyError(f"no legacy cache file for case={case.name}")
        data = np.load(p, allow_pickle=True)
        times = np.asarray(data["times"], dtype=float)
        ut = np.asarray(data["ut"], dtype=float)
        stage_durations_s = [float(case.fire_s), float(case.anneal_s)]
        t_s, y = extract_series_from_state_matrix(
            t_s=times,
            y_state=ut,
            measure=measure,
            stage_durations_s=stage_durations_s,
            conc_scale=self.conc_scale,
        )
        return SeriesTrace(
            t_s=t_s,
            y=y,
            source="legacy_cache_npz",
            metadata={"file": str(p)},
        )


def run_new_trace(
    case: ParityCase,
    measure: MeasureSpec,
    *,
    base_params: dict[str, float] | None = None,
    verbose: bool = False,
) -> SeriesTrace:
    import time as _time
    base = paper_param_set() if base_params is None else dict(base_params)
    full = full_params(base)
    structure = legacy_like_structure(full)
    samp = sampling_config()
    if verbose:
        print(
            f"[run_new_trace] '{case.name}'  fire={case.fire_C}C/{case.fire_s}s  "
            f"anneal={case.anneal_C}C/{case.anneal_s}s  base_out_dt={samp.base_out_dt_s}s",
            flush=True,
        )
    sim = Simulation(
        structure=structure,
        schedule=schedule_two_stage(case.fire_C, case.anneal_C, case.fire_s, case.anneal_s),
        sampling=samp,
        solver=solver_config(verbose=verbose),
        y0=legacy_initial_state(),
    )
    _t0 = _time.monotonic()
    sim.run()
    _elapsed = _time.monotonic() - _t0
    n_pts = len(sim.result.t_s) if sim.result is not None else '?'
    if verbose:
        print(f"[run_new_trace] '{case.name}' done: {n_pts} pts in {_elapsed:.1f}s", flush=True)
    t_new, y_new = sim.series(
        layer=measure.layer,
        kind=measure.kind,
        trap_id=measure.trap_id,
        stage=measure.stage,
        occurrence=measure.occurrence,
        units=measure.units,
    )
    return SeriesTrace(
        t_s=np.asarray(t_new, dtype=float),
        y=np.asarray(y_new, dtype=float),
        source="hdiff",
        metadata={"case": case.name},
    )


def run_new_schedule_trace(
    *,
    schedule: Schedule,
    measure: MeasureSpec,
    base_params: dict[str, float] | None = None,
    y0: list[float] | None = None,
) -> SeriesTrace:
    base = {} if base_params is None else dict(base_params)
    full = full_params(base)
    structure = legacy_like_structure(full)
    sim = Simulation(
        structure=structure,
        schedule=schedule,
        sampling=sampling_config(),
        solver=solver_config(),
        y0=legacy_initial_state() if y0 is None else list(y0),
    )
    sim.run()
    t_new, y_new = sim.series(
        layer=measure.layer,
        kind=measure.kind,
        trap_id=measure.trap_id,
        stage=measure.stage,
        occurrence=measure.occurrence,
        units=measure.units,
    )
    return SeriesTrace(
        t_s=np.asarray(t_new, dtype=float),
        y=np.asarray(y_new, dtype=float),
        source="hdiff",
        metadata={"stage": measure.stage, "layer": measure.layer, "kind": measure.kind},
    )


def interpolate_to_reference(new_trace: SeriesTrace, ref_t_s: np.ndarray) -> np.ndarray:
    t = np.asarray(new_trace.t_s, dtype=float)
    y = np.asarray(new_trace.y, dtype=float)
    ref_t = np.asarray(ref_t_s, dtype=float)
    if t.size == 0 or y.size == 0:
        raise ValueError("new trace must be non-empty")
    if not np.all(np.diff(t) >= 0):
        raise ValueError("new trace time axis must be monotonic non-decreasing")
    if ref_t.size == 0:
        raise ValueError("reference time axis must be non-empty")
    return np.interp(ref_t, t, y)


def build_log_probe_grid(ref_t_s: np.ndarray, n_points: int = 2048) -> np.ndarray:
    ref_t = np.asarray(ref_t_s, dtype=float)
    if ref_t.size == 0:
        raise ValueError("reference time axis must be non-empty")
    if n_points < 3:
        raise ValueError("n_points must be >= 3")

    t_max = float(np.max(ref_t))
    if t_max <= 0.0:
        return np.linspace(0.0, 0.0, n_points)

    min_pos = ref_t[ref_t > 0.0]
    t_min = float(np.min(min_pos)) if min_pos.size else t_max / 1e6
    t_min = max(t_min, t_max / 1e12, 1e-12)

    geom = np.geomspace(t_min, t_max, n_points - 1)
    grid = np.concatenate(([0.0], geom))
    return np.unique(grid)


def compare_traces(
    sim_trace: SeriesTrace,
    ref_trace: SeriesTrace,
    *,
    probe_t_s: np.ndarray | None = None,
) -> ComparisonMetrics:
    ref_t = np.asarray(ref_trace.t_s, dtype=float) if probe_t_s is None else np.asarray(probe_t_s, dtype=float)
    y_interp = interpolate_to_reference(sim_trace, ref_t)
    ref = np.asarray(ref_trace.y, dtype=float)
    if probe_t_s is not None:
        ref = np.interp(ref_t, np.asarray(ref_trace.t_s, dtype=float), ref)
    delta = y_interp - ref
    rel_l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(ref), 1e-30))
    rel_linf = float(np.max(np.abs(delta)) / max(np.max(np.abs(ref)), 1e-30))
    max_abs = float(np.max(np.abs(delta)))
    return ComparisonMetrics(
        rel_l2=rel_l2,
        rel_linf=rel_linf,
        max_abs=max_abs,
        n_samples=int(ref.shape[0]),
    )


def assert_case_thresholds(case: ParityCase, metrics: ComparisonMetrics) -> None:
    assert metrics.rel_l2 < case.threshold_rel_l2, (
        f"{case.name} rel_l2={metrics.rel_l2:.6f} exceeds {case.threshold_rel_l2:.6f}"
    )
    assert metrics.rel_linf < case.threshold_rel_linf, (
        f"{case.name} rel_linf={metrics.rel_linf:.6f} exceeds {case.threshold_rel_linf:.6f}"
    )


def save_legacy_series_baseline(
    *,
    out_file: Path,
    case: ParityCase,
    measure: MeasureSpec,
    trace: SeriesTrace,
) -> None:
    meta = {
        "case": {
            "name": case.name,
            "fire_C": case.fire_C,
            "anneal_C": case.anneal_C,
            "fire_s": case.fire_s,
            "anneal_s": case.anneal_s,
        },
        "measure": {
            "layer": measure.layer,
            "kind": measure.kind,
            "trap_id": measure.trap_id,
            "stage": measure.stage,
            "occurrence": measure.occurrence,
            "units": measure.units,
        },
        "source": trace.source,
        "source_metadata": trace.metadata,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        t_s=np.asarray(trace.t_s, dtype=float),
        y=np.asarray(trace.y, dtype=float),
        metadata_json=json.dumps(meta, sort_keys=True),
    )
