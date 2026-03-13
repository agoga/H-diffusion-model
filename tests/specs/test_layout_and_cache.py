"""State-layout stability and cache-schema behavior tests."""

from __future__ import annotations

import json

import numpy as np

from hdiff.cache import CacheStore
from hdiff.defaults import DEFAULT_CSI, DEFAULT_STRUCTURE
from hdiff.result import RunResult, SegmentBoundary
from hdiff.sim import Simulation
from hdiff.structure import Arrhenius, Material, TrapSpec
from tests.test_helpers import make_min_structure, make_sampling, make_schedule, make_solver


def test_layout_is_stable_for_same_structure() -> None:
    """Verify state layout indices are identical for identical structures.

    Protects against accidental non-determinism in layout ordering.
    """
    structure = make_min_structure()
    sim1 = Simulation(
        structure=structure,
        schedule=make_schedule(),
        sampling=make_sampling(),
        solver=make_solver(),
    )
    sim2 = Simulation(
        structure=structure,
        schedule=make_schedule(),
        sampling=make_sampling(),
        solver=make_solver(),
    )
    assert sim1.layout["order"] == sim2.layout["order"]
    assert sim1.layout["idx_trapped"] == sim2.layout["idx_trapped"]
    assert sim1.layout["idx_mobile"] == sim2.layout["idx_mobile"]


def test_multiple_traps_increase_n_state() -> None:
    """Verify adding one extra trap increases state-vector size by one."""
    structure = make_min_structure()
    extra_trap = TrapSpec(
        id="t2",
        trap_density=0.5,
        trap_kin=Arrhenius(nu=1e13, Ea_eV=1.2),
        detrap_kin=Arrhenius(nu=1e12, Ea_eV=1.3),
    )
    mats = dict(structure.materials)
    mats["mC"] = Material(id="mC", traps=[*mats["mC"].traps, extra_trap])
    structure2 = type(structure)(
        materials=mats,
        layers=structure.layers,
        bc=structure.bc,
        transport=structure.transport,
        conc_scale=structure.conc_scale,
    )
    sim = Simulation(
        structure=structure,
        schedule=make_schedule(),
        sampling=make_sampling(),
        solver=make_solver(),
    )
    sim2 = Simulation(
        structure=structure2,
        schedule=make_schedule(),
        sampling=make_sampling(),
        solver=make_solver(),
    )
    assert sim2.layout["n_state"] == sim.layout["n_state"] + 1


def test_cache_key_same_spec_and_changes_on_output_affecting_change() -> None:
    """Verify cache key stability and sensitivity to output-affecting config.

    Same spec must hash the same; changed sampling must hash differently.
    """
    structure = make_min_structure()
    sim1 = Simulation(
        structure=structure,
        schedule=make_schedule(),
        sampling=make_sampling(),
        solver=make_solver(),
    )
    sim2 = Simulation(
        structure=structure,
        schedule=make_schedule(),
        sampling=make_sampling(),
        solver=make_solver(),
    )
    assert sim1.cache_key() == sim2.cache_key()

    changed_sampling = type(make_sampling())(
        base_out_dt_s=11.0,
        bootstrap_duration_s=60.0,
        bootstrap_max_dt_s=0.5,
    )
    sim3 = Simulation(
        structure=structure,
        schedule=make_schedule(),
        sampling=changed_sampling,
        solver=make_solver(),
    )
    assert sim1.cache_key() != sim3.cache_key()


def test_cache_save_load_roundtrip(tmp_path) -> None:
    """Verify NPZ cache save/load preserves arrays and metadata fields."""
    structure = make_min_structure()
    sim = Simulation(
        structure=structure,
        schedule=make_schedule(),
        sampling=make_sampling(),
        solver=make_solver(),
    )
    key = sim.cache_key()
    order_json = json.dumps(sim.layout["order_jsonable"], separators=(",", ":"), sort_keys=True)
    result = RunResult(
        t_s=np.array([0.0, 1.0, 2.0]),
        y=np.zeros((3, sim.layout["n_state"])),
        boundaries=[
            SegmentBoundary(
                i_seg=0,
                stage="firing",
                t_start_s=0.0,
                t_end_s=10.0,
                i_start=0,
                i_end=3,
            )
        ],
        cache_key=key,
        schema_version=sim.schema_version,
        spec_json=sim.spec_json(),
    )
    cache = CacheStore(tmp_path)
    cache.save(
        key=key,
        result=result,
        order_json=order_json,
        timestep=1.0,
        reason="unit-test",
        completed=1,
    )

    loaded, loaded_order_json, loaded_boundaries_json = cache.load(
        key,
        requested_spec_json=sim.spec_json(),
    )

    assert np.array_equal(loaded.t_s, result.t_s)
    assert np.array_equal(loaded.y, result.y)
    assert loaded.cache_key == result.cache_key
    assert loaded.spec_json == result.spec_json
    assert loaded_order_json == order_json
    assert "firing" in loaded_boundaries_json


def test_structure_with_material_updates_one_trap_immutably() -> None:
    """Verify structure-level trap updates return a new structure and preserve original defaults."""

    updated = DEFAULT_STRUCTURE.with_material("csi", trap_id="t1", trap_nu=5e13, trap_density=2e-5)

    assert DEFAULT_STRUCTURE.materials["csi"].traps[0].trap_kin.nu == 5e12
    assert DEFAULT_STRUCTURE.materials["csi"].traps[0].trap_density == 1e-5
    assert updated.materials["csi"].traps[0].trap_kin.nu == 5e13
    assert updated.materials["csi"].traps[0].trap_density == 2e-5
    assert updated.materials["alox"] is DEFAULT_STRUCTURE.materials["alox"]


def test_structure_with_material_accepts_whole_material_replacement() -> None:
    """Verify a whole replacement Material can be injected without rebuilding the structure."""

    custom_csi = DEFAULT_CSI.with_trap("t1", detrap_Ea_eV=1.7)
    updated = DEFAULT_STRUCTURE.with_material("csi", material=custom_csi)

    assert updated.materials["csi"].traps[0].detrap_kin.Ea_eV == 1.7
    assert DEFAULT_STRUCTURE.materials["csi"].traps[0].detrap_kin.Ea_eV == 1.2


def test_structure_with_transport_updates_immutably() -> None:
    """Verify transport updates return a new structure and preserve the original values."""

    updated = DEFAULT_STRUCTURE.with_transport(prefactor=2e-3, hop_Ea_eV=0.6)

    assert DEFAULT_STRUCTURE.transport.prefactor == 1e-3
    assert DEFAULT_STRUCTURE.transport.hop_Ea_eV == 0.5
    assert updated.transport.prefactor == 2e-3
    assert updated.transport.hop_Ea_eV == 0.6
