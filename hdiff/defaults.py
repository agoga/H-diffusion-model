"""Default materials, structures, and solver settings.

Materials are defined once here and reused to assemble the standard structures.
Use `Material.with_trap(...)` for small changes.
"""

from __future__ import annotations

from .boundary import BoundaryCondition
from .sim import Sampling, SolverConfig
from .structure import Arrhenius, Layer, Material, Structure, Transport, TrapSpec


DEFAULT_ALOX = Material(id="alox",
    traps=[
        TrapSpec(
            id="t1",
            trap_density    =   1.0,
            trap_kin        =   Arrhenius(  nu = 1e13, Ea_eV = 0.9    ),
            detrap_kin      =   Arrhenius(  nu = 1e12, Ea_eV = 2.5    ),
        )],)    
    
DEFAULT_POLY = Material(id="poly_si",    
    traps=[    
        TrapSpec(    
            id="t1",    
            trap_density    =   1e-3,    
            trap_kin        =   Arrhenius(  nu = 1e13, Ea_eV = 0.50   ),
            detrap_kin      =   Arrhenius(  nu = 1e12, Ea_eV = 1.3    ),
        )],)    
    
DEFAULT_SIOX = Material(id="siox",    
    traps=[    
        TrapSpec(    
            id="t1",    
            trap_density    =   1e-1,    
            trap_kin        =   Arrhenius(  nu = 1e13, Ea_eV = 0.50   ),
            detrap_kin      =   Arrhenius(  nu = 1e12, Ea_eV = 1.45   ),
        )],)    
    
DEFAULT_CSI = Material(id="csi",    
    traps=[    
        TrapSpec(    
            id="t1",    
            trap_density    =   1e-5,    
            trap_kin        =   Arrhenius(  nu = 5e12, Ea_eV = 0.50   ),
            detrap_kin      =   Arrhenius(  nu = 1e12, Ea_eV = 1.2    ),
        )],)


def _stack_structure(*, alox: Material, poly: Material, siox: Material, csi: Material) -> Structure:
    return Structure(
        materials={
            "alox": alox,
            "poly_si": poly,
            "siox": siox,
            "csi": csi,
        },
        layers=[
            Layer(name=r"AlO$_x$", thickness_cm=100e-7, material_id="alox"),
            Layer(name=r"poly-Si", thickness_cm=290e-7, material_id="poly_si"),
            Layer(name=r"SiO$_x$", thickness_cm=1.5e-7, material_id="siox"),
            Layer(name=r"c-Si (near)", thickness_cm=608.5e-7, material_id="csi"),
            Layer(name=r"c-Si (bulk)", thickness_cm=50e-4, material_id="csi"),
        ],
        bc = BoundaryCondition.closed_closed(),
        transport=Transport(prefactor=1e-3, hop_Ea_eV=0.5),
        conc_scale=1e22,
    )


DEFAULT_STRUCTURE = _stack_structure(
    alox=DEFAULT_ALOX,
    poly=DEFAULT_POLY,
    siox=DEFAULT_SIOX,
    csi=DEFAULT_CSI,
)

# Initial condition in state order:
# trapped [layer0..layer4], then mobile [layer0..layer4].
DEFAULT_Y0: list[float] = [0.99, 1e-8, 1e-8, 1e-9, 1e-9, 1e-8, 1e-8, 1e-8, 1e-9, 1e-9]

# PETSc / Rosenbrock-W solver settings used in the direct quickstart.
DEFAULT_SOLVER = SolverConfig(
    backend =                   "petsc",
    rtol    =                   1e-5,
    atol    =                   1e-10,
    petsc_options={
        "ts_type":              "rosw",
        "ts_adapt_type":        "basic",
        "ksp_type":             "preonly",
        "pc_type":              "lu",
        "ts_exact_final_time":  "matchstep",
    },
)

# Sampling used by the direct quickstart and notebook examples.
DEFAULT_SAMPLING = Sampling(
    base_out_dt_s           =   10.0,
    bootstrap_duration_s    =   10.0,
    bootstrap_max_dt_s      =   1e-4,
)



__all__ = [
    "DEFAULT_ALOX",
    "DEFAULT_POLY",
    "DEFAULT_SIOX",
    "DEFAULT_CSI",
    "DEFAULT_STRUCTURE",
    "DEFAULT_SAMPLING",
    "DEFAULT_SOLVER",
    "DEFAULT_Y0",
]
