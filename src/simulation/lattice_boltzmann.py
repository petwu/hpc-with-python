from __future__ import annotations
import numpy as np
import src.boundary as boundary
import src.visualization as visualization
import sys
from tqdm import trange


class LatticeBoltzmann:
    """
    Implementation of the `Lattice Boltzman <https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods>`_ method for
    2D fluid simulations with a D2Q9 lattice.
    """

    def __init__(self,
                 X: int,
                 Y: int,
                 omega: float = 1.,
                 viscosity: float = None,
                 init_pdf: np.ndarray = None,
                 init_density: np.ndarray = None,
                 init_velocity: np.ndarray = None,
                 boundaries: list[boundary.BaseBoundaryCondition] = None,
                 decomposition: tuple[int, int] = None,
                 plot: bool | str = None,
                 plot_size: int = 200,
                 plot_stages: list[tuple[int, int]] = None,
                 animate: bool = None):
        """
        Construct a object that can run a Lattice Boltzman based fluid simulation.

        Parameters
        ----------
        X, Y : int
            Dimensions of the D2Q9 lattice.

        omega : float
            The rate at which the system is pushed towards the equilibrium: ω=1/τ

        viscosity : float
            The viscosity of the fluid. If specified, overrules the ``omega`` value.

        init_pdf, init_density, init_velocity : numpy.array
            Initial values for the probability density function (pdf), density field or velocity field.
            At least one of them is required. The pdf and density/velocity values are mutually exclusive,
            i.e. if the pdf is specified, the density and velocity must not be specified, and if the pdf
            is not specified, the density and/or velocity must be specified.

            The matrices/tensors must match the lattice dimensions, therefore they need to have the following shape:
            - ``init_pdf``: ``(9, Y, X)``
            - ``init_density``: ``(Y, X)``
            - ``init_velocity``: ``(2, Y, X)``

        boundaries : list[boundary.BaseBoundaryCondition]
            A list of boundary conditions to be applied. All passed objects must inherit and extend the
            :class:``BaseBoundaryCondition`` class. If no explicit boundary conditions are applied, the streaming
            operator implicitly implements periodic boundary conditions.

        decomposition : tuple[int, int]
            Decompose the simulation domain into a process grid for parallelization using MPI.
            The value must be a 2-tuple ``(x, y)`` specifying the number of processes in the respective dimension.

            Pass ``(-1, -1)`` for automatic decomposition. For more details see :class:`MpiDomain2D`.

            Note: In order to use MPI, the script has to be run with ``mpiexec``, e.g.::

                mpiexec -n 16 python3 my_simulation.py ...

        plot, plot_size, plot_stages, animate
            See :class:`DensityPlot`.
        """

        # parameter checks
        if not isinstance(omega, int | float) or not 0 < omega < 2:
            raise ValueError("for a stable simulation, reasonable values for omega should be in the interval (0, 2)")
        if init_pdf is None and init_density is None and init_velocity is None:
            raise ValueError("at least one of init_pdf/init_density/init_velocity must be provided")
        elif init_pdf is not None and (init_density is not None or init_velocity is not None):
            raise ValueError("init_pdf and init_density/init_velocity arguments are mutually exclusive")
        else:
            # else, because otherwise for some unknown reason Pylance marks the code after this class as unreachable
            pass

        # omega, viscosity
        if viscosity:
            self._omega = 1 / (3*viscosity + 0.5)
            self._viscosity = viscosity
        else:
            self._omega = omega
            self._viscosity = 1/3 * (1/self._omega - 0.5)

        # channel weights and velocities/directions
        self._weight_c = np.array([4. / 9.] + 4*[1. / 9.] + 4*[1. / 36.])
        self._channel_ca = np.array([
            # note: grid origin is in the top-left corner
            # [Y, X] shift
            [0, 0],    # channel 0: center
            [0, 1],    # channel 1: right / east
            [-1, 0],   # channel 2: top / north
            [0, -1],   # channel 3: left / west
            [1, 0],    # channel 4: bottom / south
            [-1, 1],   # channel 5: top-right / north-east
            [-1, -1],  # channel 6: top-left / north-west
            [1, -1],   # channel 7: bottom-left / south-west
            [1, 1]     # channel 8: bottom-right / south-east
        ])

        # initialize parallelization
        if decomposition:
            import src.parallelization as parallelization
            self._mpi = parallelization.MpiDomain2D((X, Y), decomposition)
        else:
            self._mpi = None

        # shapes
        self._X = X
        self._Y = Y
        self._shape = (Y, X)
        pdf_shape_g = (9, Y, X)
        velocity_shape_g = (2, Y, X)
        density_shape_g = (Y, X)
        if self.is_parallel:
            pdf_shape_l = (9, *self._mpi.buffered_subdomain_size)
            velocity_shape_l = (2, *self._mpi.buffered_subdomain_size)
            density_shape_l = self._mpi.buffered_subdomain_size
        else:
            pdf_shape_l = pdf_shape_g
            velocity_shape_l = velocity_shape_g
            density_shape_l = density_shape_g

        # initialize probability density function (pdf) [1/2]
        if init_pdf is not None:
            assert init_pdf.shape == pdf_shape_g, \
                f"init_pdf has the wrong shape: is {init_pdf.shape}, should be {pdf_shape_g}"
            if self.is_parallel:
                self._pdf_cij = np.pad(init_pdf[:][self._mpi.subdomain_selection], self._mpi.padding, "edge")
            else:
                self._pdf_cij = init_pdf
        else:
            self._pdf_cij = np.zeros(pdf_shape_l)
        self._pdf_eq_cij = np.zeros(pdf_shape_l)  # equilibrium
        self._pdf_pre_cij = np.zeros(pdf_shape_l)  # boundary handling

        # initialize density and mass
        if init_density is not None:
            assert init_density.shape == density_shape_g, \
                f"init_density has the wrong shape: is {init_density.shape}, should be {density_shape_g}"
            if self.is_parallel:
                self._density_ij = np.pad(init_density[self._mpi.subdomain_selection], self._mpi.padding, "edge")
            else:
                self._density_ij = init_density
            self._initial_mass = self.mass
        else:
            self._density_ij = np.zeros(density_shape_l)
            self._initial_mass = None
            self.update_density()
            self._initial_mass = self.mass

        # initialize velocity
        if init_velocity is not None:
            assert init_velocity.shape == velocity_shape_g, \
                f"init_velocity has the wrong shape: is {init_velocity.shape}, should be {velocity_shape_g}"
            if self.is_parallel:
                self._velocity_aij = np.pad(init_density[:][self._mpi.subdomain_selection], self._mpi.padding, "edge")
            else:
                self._velocity_aij = init_velocity
        else:
            self._velocity_aij = np.zeros(velocity_shape_l)
            self.update_velocity()

        # initialize pdf for given density/velocity [2/2]
        if init_pdf is None and (init_density is not None or init_velocity is not None):
            self.update_equilibrium()
            self._pdf_cij = self._pdf_eq_cij.copy()

        # initialize boundary handling
        boundaries = boundaries or []
        for b in boundaries:
            b.initialize(self, self._mpi.filter_boundaries(b.boundaries) if self.is_parallel else None)
        self._boundaries = {
            True: [b for b in boundaries if b.before_streaming and len(b.boundaries) > 0],
            False: [b for b in boundaries if not b.before_streaming and len(b.boundaries) > 0]
        }

        # initialize plotting
        self._step_i = 0
        min_density = np.min(np.sum(self._pdf_cij, axis=0))
        max_density = np.max(np.sum(self._pdf_cij, axis=0))
        self._plot = visualization.DensityPlot(X, Y,
                                               plot_size=plot_size,
                                               plot_stages=visualization.PlotStages(stages=plot_stages, omega=omega),
                                               vmin=min_density,
                                               vmax=max_density,
                                               plot=plot,
                                               animate=animate)
        self._plot.init(0, self._density_ij)

    @property
    def is_parallel(self) -> bool:
        """
        Boolean indicating whether the computations are parallelized using MPI.
        """
        return self._mpi is not None

    @property
    def X(self) -> int:
        """
        Returns the X dimension of the lattice.
        """
        return self._X

    @property
    def Y(self) -> int:
        """
        Returns the Y dimension of the lattice.
        """
        return self._Y

    @property
    def shape(self) -> tuple[int, int]:
        """
        Returns the (Y, X) shape/dimensions of the lattice.
        """
        return self._shape

    @property
    def weight(self) -> np.ndarray:
        """
        Returns the vector if channels weights.
        """
        return self._weight_c

    @property
    def channel(self) -> np.ndarray:
        """
        Returns the matrix of velocity direction channels.
        """
        return self._channel_ca

    @property
    def density(self) -> np.ndarray:
        """
        Returns the (Y, X) shaped density field.
        """
        return self._density_ij

    @property
    def mass(self) -> float:
        """
        Returns the total mass.

        Note that in a parallel implementation using MPI this only gives the mass of the subdomain handled by the
        current process, not the overall mass of the whole lattice.
        """
        if self.is_parallel:
            return np.sum(self._density_ij[self._mpi.physical_domain])
        return np.sum(self._density_ij)

    @property
    def omega(self) -> float:
        """
        Returns the omega value.
        """
        return self._omega

    @property
    def viscosity(self) -> float:
        """
        Returns the viscosity value.
        """
        return self._viscosity

    @property
    def pdf(self) -> np.ndarray:
        """
        Reference to the probability density function (pdf).
        """
        return self._pdf_cij

    @property
    def pdf_eq(self) -> np.ndarray:
        """
        Reference to the equilibrium probability density function (pdf).
        """
        return self._pdf_eq_cij

    @property
    def pdf_pre(self) -> np.ndarray:
        """
        Reference to the probability density function (pdf) before the streaming step.
        This property is intended to be used by boundary handling.
        """
        return self._pdf_pre_cij

    @property
    def velocity(self) -> float:
        """
        Returns the (2, Y, X) shaped velocity field.
        """
        return self._velocity_aij

    @property
    def mpi(self) -> any:
        """
        Reference to the MPI domain decomposition helper object of type :class:`parallelization.MpiDomain2D`.
        """
        return self._mpi

    @property
    def plot(self) -> visualization.DensityPlot:
        """
        Reference to the plotting object.
        """
        return self._plot

    def update_density(self):
        """
        Computes the density at each lattice point.
        """
        # at each lattice point, sum up the pdf over all velocity channels
        self._density_ij = np.sum(self._pdf_cij, axis=0)

        # sanity check: only positive densities
        assert np.min(self._density_ij) >= 0, \
            f"negative densities make no sense: {np.min(self._density_ij)}"
        # sanity check: at most 1% deviation from the initial mass
        assert self._initial_mass is None or abs((1-self.mass/self._initial_mass)*100) < 1, \
            f"the simulation generates/destroys mass: {(1-self.mass/self._initial_mass)*100:.2f}%"

    def update_velocity(self):
        """
        Computes the velocity at each lattice point.
        """
        # einsum:
        # - c
        #   -> repeated in both input -> values along this axis will be multiplied
        #   -> omitted in output -> values along this axis will be summed
        # - i, j, a
        #   -> both in input and output -> returned unsummed axis in any order
        # => for both x and y direction (a) ...
        # - multiply along axis c the velocity vector (_channel_ca[:, a]) with each value of the pdf (_pdf_cij[:, i, j])
        # - and sum up these products over all channels (c)
        # - the resulting sum of products is one of two velocity component (_velocity_aij[:, i, j])
        self._velocity_aij = np.einsum("cij,ca->aij", self._pdf_cij, self._channel_ca) / self._density_ij

    def update_equilibrium(self):
        assert self._density_ij.shape == self._velocity_aij.shape[1:], \
            f"some calculation messed up the density or velocity shape"

        # helper variables
        # (9, Y, X) <- (Y, X, 2) @ (2, 9)
        vel_dot_c = (self._velocity_aij.T @ self._channel_ca.T).T
        # (Y, X) <- (2, Y, X)
        v_norm2 = np.linalg.norm(self._velocity_aij, axis=0) ** 2

        # local equilibrium distribution function approximation
        # (9, Y, X) <- (9, 1, 1) * (1, Y, X) * ((9, Y, X) + (9, Y, X) - (1, Y, X))
        self._pdf_eq_cij = self._weight_c[..., np.newaxis, np.newaxis] * self._density_ij[np.newaxis, ...] * (
            1 + 3*vel_dot_c + 4.5*vel_dot_c**2 - 1.5*v_norm2[np.newaxis, ...])

    def streaming_step(self):
        """
        Implements the streaming operator (l.h.s.) of the BTE.
        """
        self.update_density()
        for i in range(9):
            # for each channel, shift/roll by the corresponding direction from _channel_ca
            self._pdf_cij[i] = np.roll(self._pdf_cij[i], shift=self._channel_ca[i], axis=(0, 1))

    def collision_step(self):
        """
        Implements the collision operator (r.h.s.) of the BTE.
        """
        self.update_density()
        self.update_velocity()
        self.update_equilibrium()
        self._pdf_cij += (self._pdf_eq_cij - self._pdf_cij) * self._omega

    def boundary_handling(self, before_streaming: bool):
        """
        Apply all boundary handling methods the lattice was initialized with.

        Parameters
        ----------
        before_streaming : bool
            Specify the kind of boundary handlings to be applied: Those that happen before (``True``) or those that
            happen after (``False``) the streaming operation.
        """
        if before_streaming:
            self._pdf_pre_cij = self._pdf_cij.copy()
        for b in self._boundaries[before_streaming]:
            b.apply(self)

    def communicate(self):
        """
        In case the computation is parallelized, perform the necessary communication between the subdomains.
        """
        if not self.is_parallel:
            return

        # send left
        sendbuf = np.ascontiguousarray(self._pdf_cij[:, :, 1])
        recvbuf = self._pdf_cij[:, :, -1].copy()
        self._mpi.comm.Sendrecv(sendbuf=sendbuf, dest=self._mpi._dest_left,
                                recvbuf=recvbuf, source=self._mpi._src_left)
        self._pdf_cij[:, :, -1] = recvbuf

        # send right
        sendbuf = np.ascontiguousarray(self._pdf_cij[:, :, -2])
        recvbuf = self._pdf_cij[:, :, 0].copy()
        self._mpi.comm.Sendrecv(sendbuf=sendbuf, dest=self._mpi._dest_right,
                                recvbuf=recvbuf, source=self._mpi._src_right)
        self._pdf_cij[:, :, 0] = recvbuf

        # send down
        sendbuf = np.ascontiguousarray(self._pdf_cij[:, -2, :])
        recvbuf = self._pdf_cij[:, 0, :].copy()
        self._mpi.comm.Sendrecv(sendbuf=sendbuf, dest=self._mpi._dest_down,
                                recvbuf=recvbuf, source=self._mpi._src_down)
        self._pdf_cij[:, 0, :] = recvbuf

        # send up
        sendbuf = np.ascontiguousarray(self._pdf_cij[:, 1, :])
        recvbuf = self._pdf_cij[:, -1, :].copy()
        self._mpi.comm.Sendrecv(sendbuf=sendbuf, dest=self._mpi._dest_up,
                                recvbuf=recvbuf, source=self._mpi._src_up)
        self._pdf_cij[:, -1, :] = recvbuf

    def step(self, n: int = 1, progress: bool = False, *args, **kwargs):
        """
        Run one or multiple simulation steps.

        Parameters
        ----------
        n : int
            The number of simulation steps.

        progress : bool
            Whether to show a progress bar. Only applicable if n > 1.

        *args, **kwargs
            Further arguments for the process bar. See :method`tqdm.trange`
        """
        if progress and n > 1 and (not self.is_parallel or self._mpi.rank == 0):
            if not "file" in kwargs:
                kwargs["file"] = sys.stdout
            steps = trange(n, *args, **kwargs)
        else:
            steps = range(n)

        for _ in steps:
            self.communicate()
            self.boundary_handling(True)
            self.streaming_step()
            self.boundary_handling(False)
            self.collision_step()
            self.update_plot()
            self._step_i += 1

    def update_plot(self):
        """
        Update the plot of the density field.
        """
        self.plot.update(self._step_i, self._density_ij)
