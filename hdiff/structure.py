"""Physical structure dataclasses for the H-diffusion stack.

Hierarchy::

    Structure
    ├── materials: dict[id → Material]
    │   └── traps: list[TrapSpec]
    │       ├── trap_kin: Arrhenius   (trapping rate k_trap)
    │       └── detrap_kin: Arrhenius (detrapping rate k_detrap)
    ├── layers: list[Layer]           (ordered left→right)
    ├── bc: BoundaryCondition         (only "closed_closed" in v1)
    ├── transport: Transport           (mobile-H diffusivity)
    └── conc_scale: float             (cm^-3; solver units × conc_scale = physical units)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Arrhenius:
    """Arrhenius rate law: k(T) = nu * exp(-Ea_eV / (kB * T))."""

    nu: float
    """Attempt frequency (attempt/s)."""
    Ea_eV: float
    """Activation energy (eV)."""

    def with_updates(self, *, nu: float | None = None, Ea_eV: float | None = None) -> "Arrhenius":
        """Return a copy with selected fields replaced."""

        return replace(
            self,
            nu=self.nu if nu is None else float(nu),
            Ea_eV=self.Ea_eV if Ea_eV is None else float(Ea_eV),
        )


@dataclass(frozen=True)
class TrapSpec:
    """A single hydrogen trap site type within a material."""

    id: str
    """Short identifier, e.g. 'A', 'B'.  Must be unique within a Material."""
    trap_density: float
    """Maximum trapped-H concentration in solver units (i.e. cm^-3 / conc_scale)."""
    trap_kin: Arrhenius
    """Trapping rate parameters.  Flux = k_trap * C_mobile * (capacity - C_trapped)."""
    detrap_kin: Arrhenius
    """Detrapping rate parameters.  Flux = k_detrap * C_trapped."""

    def with_updates(
        self,
        *,
        trap_density: float | None = None,
        trap_nu: float | None = None,
        trap_Ea_eV: float | None = None,
        detrap_nu: float | None = None,
        detrap_Ea_eV: float | None = None,
    ) -> "TrapSpec":
        """Return a copy with selected trap parameters replaced."""

        return replace(
            self,
            trap_density=self.trap_density if trap_density is None else float(trap_density),
            trap_kin=self.trap_kin.with_updates(nu=trap_nu, Ea_eV=trap_Ea_eV),
            detrap_kin=self.detrap_kin.with_updates(nu=detrap_nu, Ea_eV=detrap_Ea_eV),
        )


@dataclass(frozen=True)
class Material:
    """A material type that can be assigned to one or more layers."""

    id: str
    """Unique identifier, must match the key in Structure.materials."""
    traps: list[TrapSpec]
    """Trap site types present in this material (may be empty)."""

    def with_trap(self, trap_id: str, **updates: float) -> "Material":
        """Return a copy with one trap updated by id.

        Supported update keys are the kwargs accepted by ``TrapSpec.with_updates``.
        Example: ``DEFAULT_CSI.with_trap("t1", trap_nu=5e13)``.
        """

        new_traps: list[TrapSpec] = []
        found = False
        for trap in self.traps:
            if trap.id == trap_id:
                new_traps.append(trap.with_updates(**updates))
                found = True
            else:
                new_traps.append(trap)
        if not found:
            raise ValueError(f"trap id not found in material {self.id}: {trap_id}")
        return replace(self, traps=new_traps)


@dataclass(frozen=True)
class Layer:
    """A single physical layer in the stack."""

    name: str
    """Display name, also used as dictionary key throughout the solver (e.g. 'SiNx')."""
    thickness_cm: float
    """Physical thickness in cm.  Used as the FV cell size."""
    material_id: str
    """Key into Structure.materials."""


@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary condition for the outermost layer faces.

    Only ``kind='closed_closed'`` (zero-flux at both surfaces) is supported in v1.
    """

    kind: str = "closed_closed"
    params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Transport:
    """Bulk diffusivity of mobile H: D(T) = prefactor * exp(-hop_Ea_eV / (kB * T))."""

    prefactor: float
    """Pre-exponential diffusivity (cm^2/s)."""
    hop_Ea_eV: float
    """Hopping activation energy (eV)."""


@dataclass(frozen=True)
class Structure:
    """Complete physical description of the layered stack.

    ``conc_scale`` is the numerical conversion factor between the solver's
    dimensionless concentration variable and physical units (cm^-3).  All
    ``trap_density`` values are stored pre-divided by ``conc_scale`` so the
    solver operates on O(1) numbers rather than O(10^22) numbers.
    """

    materials: dict[str, Material]
    layers: list[Layer]
    """Ordered left-to-right; index 0 is the surface-adjacent layer."""
    bc: BoundaryCondition
    transport: Transport
    conc_scale: float = 1e22
    """cm^-3.  Physical concentration = solver_value * conc_scale."""

    def with_material(
        self,
        material_id: str,
        *,
        material: Material | None = None,
        trap_id: str | None = None,
        **trap_updates: float,
    ) -> "Structure":
        """Return a copy with one material replaced or one trap updated.

        Examples:
            structure.with_material("csi", material=DEFAULT_CSI.with_trap("t1", trap_nu=5e13))
            structure.with_material("csi", trap_id="t1", trap_nu=5e13)
        """

        if material_id not in self.materials:
            raise ValueError(f"material id not found in structure: {material_id}")
        if material is not None and trap_updates:
            raise ValueError("provide either material or trap updates, not both")
        if material is not None and trap_id is not None:
            raise ValueError("trap_id cannot be used when replacing the whole material")
        if material is None and trap_id is None:
            raise ValueError("trap_id is required when applying trap updates")

        current = self.materials[material_id]
        new_material = material if material is not None else current.with_trap(str(trap_id), **trap_updates)
        new_materials = dict(self.materials)
        new_materials[material_id] = new_material
        return replace(self, materials=new_materials)

    def with_transport(
        self,
        *,
        prefactor: float | None = None,
        hop_Ea_eV: float | None = None,
    ) -> "Structure":
        """Return a copy with selected transport fields replaced."""

        return replace(
            self,
            transport=replace(
                self.transport,
                prefactor=self.transport.prefactor if prefactor is None else float(prefactor),
                hop_Ea_eV=self.transport.hop_Ea_eV if hop_Ea_eV is None else float(hop_Ea_eV),
            ),
        )

    def validate(self) -> None:
        if self.conc_scale <= 0.0:
            raise ValueError("conc_scale must be > 0")
        if not self.layers:
            raise ValueError("layers must be non-empty")
        if not self.materials:
            raise ValueError("materials must be non-empty")

        material_ids = set(self.materials.keys())
        if len(material_ids) != len(self.materials):
            raise ValueError("material ids must be unique")

        layer_names: set[str] = set()
        for layer in self.layers:
            if not layer.name:
                raise ValueError("layer name must be non-empty")
            if layer.name in layer_names:
                raise ValueError(f"duplicate layer name: {layer.name}")
            layer_names.add(layer.name)
            if layer.thickness_cm <= 0.0:
                raise ValueError(f"layer thickness must be > 0: {layer.name}")
            if layer.material_id not in material_ids:
                raise ValueError(
                    f"layer material_id not found in materials: {layer.material_id}"
                )

        for material in self.materials.values():
            if material.id not in self.materials:
                raise ValueError(f"material key mismatch for id={material.id}")
            trap_ids: set[str] = set()
            for trap in material.traps:
                if not trap.id:
                    raise ValueError(f"empty trap id in material {material.id}")
                if trap.id in trap_ids:
                    raise ValueError(
                        f"duplicate trap id {trap.id} in material {material.id}"
                    )
                trap_ids.add(trap.id)
                if trap.trap_density < 0.0:
                    raise ValueError(
                        f"trap capacity must be >= 0 for {material.id}:{trap.id}"
                    )
                if trap.trap_kin.nu <= 0.0 or trap.detrap_kin.nu <= 0.0:
                    raise ValueError(
                        f"attempt frequencies must be > 0 for {material.id}:{trap.id}"
                    )
                if trap.trap_kin.Ea_eV < 0.0 or trap.detrap_kin.Ea_eV < 0.0:
                    raise ValueError(
                        f"activation energies must be >= 0 for {material.id}:{trap.id}"
                    )

        if self.bc.kind != "closed_closed":
            raise ValueError("only closed_closed boundary condition is supported in v1")

    def build_fv_geometry(self) -> dict[str, list[float] | list[str]]:
        """Pre-compute finite-volume geometry quantities.

        Returns a dict with:

        * ``layer_names`` — ordered layer name strings.
        * ``thickness_cm`` — FV cell widths (cm).
        * ``inv_thickness_cm`` — reciprocals; used for flux divergence.
        * ``d_interface_cm`` — centre-to-centre distances across each
          inter-layer interface (arithmetic mean of adjacent thicknesses);
          used in the diffusion gradient term.
        """
        self.validate()
        layer_names = [layer.name for layer in self.layers]
        thickness_cm = [layer.thickness_cm for layer in self.layers]
        inv_thickness_cm = [1.0 / value for value in thickness_cm]
        d_interface_cm = [
            0.5 * (thickness_cm[index] + thickness_cm[index + 1])
            for index in range(len(thickness_cm) - 1)
        ]
        return {
            "layer_names": layer_names,
            "thickness_cm": thickness_cm,
            "inv_thickness_cm": inv_thickness_cm,
            "d_interface_cm": d_interface_cm,
        }
