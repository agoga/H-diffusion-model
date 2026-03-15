"""Boundary-condition models for the diffusion solver.

This module owns boundary law abstractions and runtime boundary context.
Structure references these dataclasses; simulation evaluates them each RHS call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundaryFaceContext:
    """Runtime inputs for evaluating the left boundary law.

    All concentrations are in solver units.
    Returned flux must have units:

        solver_concentration * cm / s

    Positive flux means hydrogen enters the domain.
    """

    t_s: float
    T_K: float
    C_cell: float
    dx_cm: float
    D_mobile_cm2_s: float


class LeftBoundaryLaw(ABC):
    """Abstract law for the left outer boundary."""

    @abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def flux_into_domain(self, ctx: BoundaryFaceContext) -> float:
        """Return flux into the domain through the left boundary.

        Units:
            solver_concentration * cm / s
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ClosedBoundary(LeftBoundaryLaw):
    """Zero-flux left boundary."""

    kind: str = field(init=False, default="closed")

    def validate(self) -> None:
        return None

    def flux_into_domain(self, ctx: BoundaryFaceContext) -> float:
        return 0.0


@dataclass(frozen=True)
class OpenAirBoundary(LeftBoundaryLaw):
    k_ref_cm_s: float
    T_ref_K: float = 573.15  # 300 C
    E_eff_eV: float = 0.0
    kind: str = field(init=False, default="open_air")

    def validate(self) -> None:
        if not math.isfinite(self.k_ref_cm_s) or self.k_ref_cm_s < 0.0:
            raise ValueError("OpenAirBoundary requires k_ref_cm_s >= 0")
        if not math.isfinite(self.T_ref_K) or self.T_ref_K <= 0.0:
            raise ValueError("OpenAirBoundary requires T_ref_K > 0")
        if not math.isfinite(self.E_eff_eV) or self.E_eff_eV < 0.0:
            raise ValueError("OpenAirBoundary requires E_eff_eV >= 0")

    def k_cm_s(self, T_K: float) -> float:
        kb_eV_per_K = 8.617333262145e-5
        return self.k_ref_cm_s * math.exp(
            -(self.E_eff_eV / kb_eV_per_K) * (1.0 / T_K - 1.0 / self.T_ref_K)
        )

    def flux_into_domain(self, ctx: BoundaryFaceContext) -> float:
        # Open sink, outside concentration = 0
        return -self.k_cm_s(ctx.T_K) * ctx.C_cell

@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary conditions for the stack.

    In v1 only the left boundary is configurable.
    The right boundary is always treated as closed.
    """

    left: LeftBoundaryLaw = field(default_factory=ClosedBoundary)

    @classmethod
    def closed_closed(cls) -> "BoundaryCondition":
        return cls(left=ClosedBoundary())

    @classmethod
    def open_closed(
        cls,
        *,
        k0_cm_s: float,
        E_eff_eV: float,
    ) -> "BoundaryCondition":
        return cls(
            left=OpenAirBoundary(
                k_ref_cm_s=float(k0_cm_s),
                E_eff_eV=float(E_eff_eV),
            )
        )

    def validate(self) -> None:
        self.left.validate()

    def left_flux_into_domain(self, ctx: BoundaryFaceContext) -> float:
        return self.left.flux_into_domain(ctx)


__all__ = [
    "BoundaryCondition",
    "BoundaryFaceContext",
    "ClosedBoundary",
    "LeftBoundaryLaw",
    "OpenAirBoundary",
]
