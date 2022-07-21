from __future__ import annotations
import src.simulation as sim
import numpy as np
from ._base import BaseBoundaryCondition, Boundary


class MovingWallBoundaryCondition(BaseBoundaryCondition):
    """
    Models a moving wall boundary condition, where the bouncing back populations will gain or lose momentum during the
    interaction with the wall.
    """

    def __init__(self,
                 boundaries: str | Boundary,
                 wall_velocity: list | np.ndarray,
                 extrapolate_density: bool = False):
        """
        Parameters
        ----------
        boundaries : str | Boundary
            See :class:`BaseBoundaryCondition`.

        wall_velocity : list | np.ndarray
            Velocity of the wall. The value should be an array of shape ``(2,)`` or list of length 2 where the 1st and
            2nd elements denote the velocity in y- and x-direction respectively.

        extrapolate_density : bool
            If ``True``, the density at the boundary is extrapolated.
            Otherwise it is assume to be equal to the average density.
        """
        super().__init__(boundaries, False)

        self._wall_velocity = np.array(wall_velocity)
        self._extrapolate_density = extrapolate_density

    def initialize(self, lattice: sim.LatticeBoltzmann, boundaries: str | Boundary = None):
        """
        Implements the :method:`BaseBoundaryCondition.initialize` method.
        """
        super().initialize(lattice, boundaries)

        # precompute constant part of the coefficient: 2 w_i (c_i · U_w) / c_s^2
        # note: c_s=1/√3 is the speed of sound in lattice units
        #       -> 2*(1/(√3)^2) = 2/(1/3) = 6
        #       -> 6 w_i (c_i · U_w)
        self._weighted_wall_velocity = np.zeros(lattice.pdf.shape)
        for b in self.boundaries:
            _, channels, y, x = self.boundary_indices[b]
            value = 6 * lattice.weight[channels] * (lattice.channel[channels] @ self._wall_velocity).T
            self._weighted_wall_velocity[channels, y, x] = value[:, np.newaxis]  # (3, :) = (3, 1)

        # wall density computation
        if self._extrapolate_density:
            self._wall_density = np.zeros(lattice.pdf.shape)
            # define pdf masks for extrapolation
            self._extrpltn_coefs_pre = {
                # pre-streaming: all channels on the side of the wall
                Boundary.TOP:    np.array([0, 0, 1, 0, 0, 1, 1, 0, 0]),
                Boundary.BOTTOM: np.array([0, 0, 0, 0, 1, 0, 0, 1, 1]),
                Boundary.LEFT:   np.array([0, 0, 0, 1, 0, 0, 1, 1, 0]),
                Boundary.RIGHT:  np.array([0, 1, 0, 0, 0, 1, 0, 0, 1]),
            }
            self._extrpltn_coefs_post = {
                # post-streaming: all channels on the side of the wall + their neighboring channels
                Boundary.TOP:    np.array([1, 1, 1, 1, 0, 1, 1, 0, 0]),
                Boundary.BOTTOM: np.array([1, 1, 0, 1, 1, 0, 0, 1, 1]),
                Boundary.LEFT:   np.array([1, 0, 1, 1, 1, 0, 1, 1, 0]),
                Boundary.RIGHT:  np.array([1, 1, 1, 0, 1, 1, 0, 0, 1]),
            }
        else:
            # if not extrapolated:
            # - use the density as wall density
            # - the whole coefficient is constant and can be precomputed
            self._wall_density = np.full(lattice.pdf.shape, lattice.density.mean())
            self._interaction_term = self._wall_density * self._weighted_wall_velocity

    def _determine_interaction_term(self, lattice: sim.LatticeBoltzmann):
        # in case we use the average density instead of interpolating, the interaction term is constant
        if not self._extrapolate_density:
            return

        # extrapolate density at the wall
        # references:
        # - Sorush Khajepor et al.
        #   A study of wall boundary conditions in pseudopotential lattice Boltzmann models
        #   Equation 14
        #   https://pure.hw.ac.uk/ws/portalfiles/portal/23111857/1_s2.0_S0045793018302548_main.pdf
        # - Qisu Zou and Xiaoyi He
        #   On pressure and velocity boundary conditions for the lattice Boltzmann BGK model
        #   Equation 19
        #   https://arxiv.org/pdf/comp-gas/9508001.pdf
        for b in self.boundaries:
            _, _, y, x = self.boundary_indices[b]
            self._wall_density[:, y, x] =\
                lattice.pdf_pre[:, y, x].T @ self._extrpltn_coefs_pre[b] +\
                lattice.pdf[:, y, x].T @ self._extrpltn_coefs_post[b]

        # compute interaction term based on wall density
        self._interaction_term = self._wall_density * self._weighted_wall_velocity

    def apply(self, lattice: sim.LatticeBoltzmann):
        """
        Implements the :method:`BaseBoundaryCondition.apply` method.
        """
        super().apply(lattice)

        self._determine_interaction_term(lattice)
        for b in self.boundaries:
            channels_opposite, channels, y, x = self.boundary_indices[b]
            # 1. like rigid wall: reflect the pdf values of the corresponding channels at the boundary cells
            # 2. additionally substract a term to which is proportional to the wall velocity
            lattice.pdf[channels_opposite, y, x] = lattice.pdf_pre[channels, y, x] - \
                self._interaction_term[channels, y, x]
