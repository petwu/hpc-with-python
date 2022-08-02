from __future__ import annotations
import src.simulation as sim
from ._base import BaseBoundary, Boundary


class RigidWall(BaseBoundary):
    """
    Models a rigid wall boundary condition, that has a bounce back behavior at the boundaries.
    """

    def __init__(self, boundaries: str|Boundary):
        """
        Parameters
        ----------
        boundaries : str|Boundary
            See :class:`BaseBoundary`.
        """
        super().__init__(boundaries, False)

    def apply(self, lattice: sim.LatticeBoltzmann):
        """
        Implements the :method:`BaseBoundary.apply` method for a rigid wall.
        """
        super().apply(lattice)

        for b in self.boundaries:
            channels_opposite, channels, y, x = self.boundary_indices[b]
            # reflect the (pre-streaming) pdf values of the corresponding channels at the boundary grid cells
            # e.g. (BOTTOM): pdf[[2,5,6], -1, :] = pdf_pre[[4,7,8], -1, :]
            lattice.pdf[channels_opposite, y, x] = lattice.pdf_pre[channels, y, x]
