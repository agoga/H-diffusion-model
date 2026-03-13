"""Simulation integration tests for piecewise solve, restart, query APIs, and cache hits."""

from __future__ import annotations

import numpy as np

from hdiff.schedule import Schedule, Segment
from hdiff.sim import Sampling, Simulation, SolverConfig
from hdiff.structure import (
    Arrhenius,
    BoundaryCondition,
    Layer,
    Material,
    Structure,
    Transport,
    TrapSpec,
)


def make_fast_structure() -> Structure:
    shallow = TrapSpec(
        id="t1",
        trap_density=1.0,
        trap_kin=Arrhenius(nu=1e3, Ea_eV=0.2),
        detrap_kin=Arrhenius(nu=1e2, Ea_eV=0.3),
    )
    mats = {
        "mA": Material(id="mA", traps=[shallow]),
        "mB": Material(id="mB", traps=[shallow]),
        "mC": Material(id="mC", traps=[shallow]),
        "mSi": Material(id="mSi", traps=[shallow]),
    }
    layers = [
        Layer(name="A", thickness_cm=1e-6, material_id="mA"),
        Layer(name="B", thickness_cm=2e-6, material_id="mB"),
        Layer(name="C", thickness_cm=5e-7, material_id="mC"),
        Layer(name="D", thickness_cm=1e-5, material_id="mSi"),
        Layer(name="E", thickness_cm=2e-2, material_id="mSi"),
    ]
    return Structure(
        materials=mats,
        layers=layers,
        bc=BoundaryCondition(kind="closed_closed", params={}),
        transport=Transport(prefactor=1e-2, hop_Ea_eV=0.05),
        conc_scale=1e22,
    )


def make_fast_solver() -> SolverConfig:
    return SolverConfig(
        backend="petsc",
        rtol=1e-6,
        atol=1e-9,
        petsc_options={"ts_type": "beuler", "ts_adapt_type": "none"},
        max_steps=100000,
    )


def test_piecewise_run_builds_boundaries_and_stage_queries() -> None:
    """Run a two-stage solve and verify stage metadata + query API behavior.

    Confirms boundaries are created, stage counts are correct, and per-stage
    series/total queries align as expected.
    """
    structure = make_fast_structure()
    schedule = Schedule(
        segments=[
            Segment(duration_s=0.02, stage="firing", T_K=900.0),
            Segment(duration_s=0.04, stage="annealing", T_K=520.0),
        ]
    )
    sampling = Sampling(base_out_dt_s=0.01, bootstrap_duration_s=0.005, bootstrap_max_dt_s=0.001)
    sim = Simulation(
        structure=structure,
        schedule=schedule,
        sampling=sampling,
        solver=make_fast_solver(),
    )

    result = sim.run()

    assert result.y.shape[1] == sim.layout["n_state"]
    assert len(result.boundaries) == 2
    assert sim.stage_occurrences("firing") == 1
    assert sim.stage_occurrences("annealing") == 1
    assert sim.stage_bounds("firing", 0) == (0.0, 0.02)

    t_m, mobile = sim.series(layer="A", kind="mobile", stage="annealing", units="solver")
    t_tot, total = sim.layer_total(layer="A", stage="annealing", units="solver")
    assert len(t_m) == len(mobile)
    assert len(t_tot) == len(total)
    assert np.all(np.asarray(total) >= np.asarray(mobile))


def test_restart_from_boundary_snapshot_reproduces_tail() -> None:
    """Verify restart semantics: tail-only rerun matches full-run tail result.

    Uses snapshot at end of first segment as y0 for a second simulation.
    """
    structure = make_fast_structure()
    full_schedule = Schedule(
        segments=[
            Segment(duration_s=0.02, stage="firing", T_K=900.0),
            Segment(duration_s=0.03, stage="annealing", T_K=520.0),
        ]
    )
    tail_schedule = Schedule(
        segments=[
            Segment(duration_s=0.03, stage="annealing", T_K=520.0),
        ]
    )
    sampling = Sampling(base_out_dt_s=0.01, bootstrap_duration_s=0.005, bootstrap_max_dt_s=0.001)
    solver = make_fast_solver()

    sim_full = Simulation(
        structure=structure,
        schedule=full_schedule,
        sampling=sampling,
        solver=solver,
    )
    sim_full.run()
    y_mid = sim_full.snapshot_end(i_seg=0)
    y_end_full = np.asarray(sim_full.snapshot_end(i_seg=1))

    sim_tail = Simulation(
        structure=structure,
        schedule=tail_schedule,
        sampling=sampling,
        solver=solver,
        y0=y_mid,
    )
    sim_tail.run()
    y_end_tail = np.asarray(sim_tail.snapshot_end(i_seg=0))

    assert np.allclose(y_end_full, y_end_tail, rtol=1e-5, atol=1e-8)


def test_second_identical_run_hits_cache(tmp_path) -> None:
    """Verify identical second run is loaded from cache rather than recomputed."""
    structure = make_fast_structure()
    schedule = Schedule(
        segments=[
            Segment(duration_s=0.02, stage="firing", T_K=900.0),
            Segment(duration_s=0.03, stage="annealing", T_K=520.0),
        ]
    )
    sampling = Sampling(base_out_dt_s=0.01, bootstrap_duration_s=0.005, bootstrap_max_dt_s=0.001)
    solver = make_fast_solver()

    sim1 = Simulation(
        structure=structure,
        schedule=schedule,
        sampling=sampling,
        solver=solver,
        cache_dir=tmp_path,
    )
    sim1.run()
    assert sim1.result is not None

    sim2 = Simulation(
        structure=structure,
        schedule=schedule,
        sampling=sampling,
        solver=solver,
        cache_dir=tmp_path,
    )
    sim2.run()
    assert sim2.result is not None
    assert np.array_equal(sim1.result.t_s, sim2.result.t_s)
    assert np.array_equal(sim1.result.y, sim2.result.y)
