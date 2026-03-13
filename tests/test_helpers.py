from __future__ import annotations

from hdiff.schedule import Schedule, Segment
from hdiff.sim import Sampling, SolverConfig
from hdiff.structure import (
    Arrhenius,
    BoundaryCondition,
    Layer,
    Material,
    Structure,
    Transport,
    TrapSpec,
)


def make_min_structure() -> Structure:
    trap = TrapSpec(
        id="t1",
        trap_density=1.0,
        trap_kin=Arrhenius(nu=1e13, Ea_eV=0.9),
        detrap_kin=Arrhenius(nu=1e12, Ea_eV=1.1),
    )
    mats = {
        "mA": Material(id="mA", traps=[trap]),
        "mB": Material(id="mB", traps=[trap]),
        "mC": Material(id="mC", traps=[trap]),
        "mSi": Material(id="mSi", traps=[trap]),
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
        transport=Transport(prefactor=1e9, hop_Ea_eV=0.5),
        conc_scale=1e22,
    )


def make_schedule() -> Schedule:
    return Schedule(
        segments=[
            Segment(duration_s=10.0, stage="firing", T_K=943.15),
            Segment(duration_s=1000.0, stage="annealing", T_K=523.15),
        ],
        n_cycles=1,
    )


def make_sampling() -> Sampling:
    return Sampling(
        base_out_dt_s=10.0,
        bootstrap_duration_s=60.0,
        bootstrap_max_dt_s=0.5,
    )


def make_solver() -> SolverConfig:
    return SolverConfig(
        backend="petsc",
        rtol=1e-7,
        atol=1e-9,
        petsc_options={"ts_adapt_type": "none"},
        max_steps=None,
    )
