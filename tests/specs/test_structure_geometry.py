"""Structure and FV-geometry contract tests for layer ordering and dimensions."""

from __future__ import annotations

from tests.test_helpers import make_min_structure


def test_structure_validates_minimal_preset() -> None:
    """Ensure the minimal canned structure passes schema/consistency validation."""
    structure = make_min_structure()
    structure.validate()


def test_geometry_order_and_lengths_match_layers() -> None:
    """Verify geometry arrays preserve layer order and expected lengths/values."""
    structure = make_min_structure()
    geom = structure.build_fv_geometry()

    assert geom["layer_names"] == ["A", "B", "C", "D", "E"]
    assert len(geom["thickness_cm"]) == 5
    assert len(geom["inv_thickness_cm"]) == 5
    assert len(geom["d_interface_cm"]) == 4
    assert geom["thickness_cm"][0] == 1e-6
    assert geom["thickness_cm"][4] == 2e-2
