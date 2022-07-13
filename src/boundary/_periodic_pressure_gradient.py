from __future__ import annotations
import src.simulation as sim
import numpy as np
from ._base import BaseBoundaryCondition, Boundary


class PeriodicPressureGradientBoundaryCondition(BaseBoundaryCondition):
    """
    Models a system with periodic boundary conditions and a pressure gradient between two opposing boundaries.
    """

    def __init__(self, orientation: str, density_gradient: list | np.ndarray, backwards: bool = False):
        """
        Parameters
        ----------
        orientation : str
            The orientation of the gradient. One of:
            - ``h`` or ``horizontal`` for a gradient between the left and right boundaries
            - ``v`` or ``vertical`` for a gradient between the top and bottom boundaries

        density_gradient : list | np.ndarray
            A list/array consisting of two numbers describing the inlet and outlet density.
            The density ρ is related to the pressure p through the ideal gas equation of state:
            ρ=p/c_s^2 with c_s=1/√3 being the speed of sound in lattice units.
        """
        if orientation in ["h", "horizontal"]:
            self._horizontal = True
            self._inlet_boundary = Boundary.LEFT
            self._outlet_boundary = Boundary.RIGHT
        elif orientation in ["v", "vertical"]:
            self._horizontal = False
            self._inlet_boundary = Boundary.TOP
            self._outlet_boundary = Boundary.BOTTOM
        else:
            raise ValueError(f"orientation parameter must be one of 'h', 'horizontal', 'v', 'vertical'")

        super().__init__(self._inlet_boundary | self._outlet_boundary, True)

        self._gradient = density_gradient

        # map each boundary to the indices required for the extra nodes at the inlet/outlet
        self._extension_indices = {
            Boundary.TOP:    (-2, slice(None)),
            Boundary.BOTTOM: (1,  slice(None)),
            Boundary.LEFT:   (slice(None), -2),
            Boundary.RIGHT:  (slice(None),  1),
        }

    def initialize(self, lattice: sim.LatticeBoltzmann):
        """
        Implements the :method:`BaseBoundaryCondition.initialize` method.
        """
        super().initialize(lattice)

        # initialize array of extra nodes with inlet/outlet density
        boundary_size = lattice.Y if self._horizontal else lattice.X
        self._inlet_density = np.full(boundary_size, self._gradient[0])
        self._outlet_density = np.full(boundary_size, self._gradient[1])

    def _calc_equilibrium(self, d: np.ndarray, u: np.ndarray, c: np.ndarray, w: np.ndarray) -> np.ndarray:
        # copied from LatticeBoltzmann.update_equilibrium
        u_dot_c = (u.T @ c.T).T
        u_norm2 = np.linalg.norm(u, axis=0)**2
        return w[..., np.newaxis] * d[np.newaxis, ...] * (
            1. + 3. * u_dot_c + 4.5 * u_dot_c ** 2 - 1.5 * u_norm2[np.newaxis, ...])

    def apply(self, lattice: sim.LatticeBoltzmann):
        """
        Implements the :method:`BaseBoundaryCondition.apply` method.
        """
        super().apply(lattice)

        for b, d in [(self._inlet_boundary, self._inlet_density), (self._outlet_boundary, self._outlet_density)]:
            c, _, y, x = self.boundary_indices[b]
            y2, x2 = self._extension_indices[b]
            eq = self._calc_equilibrium(d, lattice.velocity[:, y2, x2], lattice.channel, lattice.weight)
            lattice.pdf[c, y, x] = eq[c] + lattice.pdf[c, y2, x2] - lattice.pdf_eq[c, y2, x2]
