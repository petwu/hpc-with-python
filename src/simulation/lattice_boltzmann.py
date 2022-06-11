from IPython import display, get_ipython
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import re


class LatticeBoltzmann:
    """
    Implementation of the `Lattice Boltzman <https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods>`_ method for
    2D fluid simulations with a D2Q9 lattice.
    """

    def __init__(self,
                 X: int,
                 Y: int,
                 omega: float = 1.,
                 init_pdf: np.array = None,
                 init_density: np.array = None,
                 init_velocity: np.array = None,
                 plot: bool | str = None,
                 plot_width: int = 200,
                 plot_stages: list[tuple[int, int]] = None,
                 animate: bool = None):
        """
        Construct a object that can run a Lattice Boltzman based fluid simulation.

        Parameters
        ----------
        X, Y : int
            Dimensions of the D2Q9 lattice.

        omega : float
            the rate at which the system is pushed towards the equilibrium: ω=1/τ

        init_pdf, init_density, init_velocity : numpy.array
            Initial values for the probability density function (pdf), density field or velocity field.
            At least one of them is required. The pdf and density/velocity values are mutually exclusive,
            i.e. if the pdf is specified, the density and velocity must not be specified, and if the pdf
            is not specified, the density and/or velocity must be specified.

            The matrices/tensors must match the lattice dimensions, therefore they need to have the following shape:
            - ``init_pdf``: ``(9, Y, X)``
            - ``init_density``: ``(Y, X)``
            - ``init_velocity``: ``(2, Y, X)``

        plot : bool|"once"
            Whether to call ``matplotlib.pyplot.show()`` in order to show a plot.
            The plot gets updated with each call to ``step()`` automatically.
            Plotting gets done in a non-blocking way.

            If the value is ``once`` instead of a bool, then only the first update is shown.
            This might be useful in Jupyter to show the the initial state and then the final animation.

            Defaults to ``once`` inside Jupyter/IPython and ``True`` otherwise.

        plot_width : int
            Width of the plot in mm.

        plot_stages : list[tuple[int, int]]
            If not every plotted image should be part of the animation, but e.g. less for later iteration steps, then
            this can be realized by defining stages. A stage is a 2-tuple ``(i, s)`` where ``i`` denotes the index
            where the stage begins and ``s`` is step size to use in that stage.

        animate : bool
            Whether to store each image in order to create an animation.
            Defaults to ``True`` inside Jupyter/IPython and ``True`` otherwise.
        """

        # parameter checks
        if not isinstance(omega, int | float) or not 0 < omega < 2:
            raise ValueError("for a stable simulation, reasonable values for omega should be in the interval (0, 2)")
        if init_pdf is None and init_density is None and init_velocity is None:
            raise ValueError("at least one of init_pdf/init_density/init_velocity must be provided")
        elif init_pdf is not None and (init_density is not None or init_velocity is not None):
            raise ValueError("init_pdf and init_density/init_velocity arguments are mutually exclusive")

        # omega
        self._omega = omega

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

        # shapes
        self._X = X
        self._Y = Y
        pdf_shape = (9, Y, X)
        velocity_shape = (2, Y, X)
        density_shape = (Y, X)
        self._shape = (Y, X)

        # initialize probability density function (pdf) [1/2]
        if init_pdf is not None:
            assert init_pdf.shape == pdf_shape, \
                f"init_pdf has the wrong shape: is {init_pdf.shape}, should be {pdf_shape}"
            self._pdf_cij = init_pdf
        else:
            self._pdf_cij = np.zeros(pdf_shape)
        self._pdf_eq_cij = np.zeros(pdf_shape)  # equilibrium

        # initialize density and mass
        if init_density is not None:
            assert init_density.shape == density_shape, \
                f"init_density has the wrong shape: is {init_density.shape}, should be {density_shape}"
            self._density_ij = init_density
            self._initial_mass = self.mass
        else:
            self._density_ij = np.zeros(density_shape)
            self._initial_mass = None
            self.update_density()
            self._initial_mass = self.mass

        # initialize velocity
        if init_velocity is not None:
            assert init_velocity.shape == velocity_shape, \
                f"init_velocity has the wrong shape: is {init_velocity.shape}, should be {velocity_shape}"
            self._velocity_aij = init_velocity
        else:
            self._velocity_aij = np.zeros(velocity_shape)
            self.update_velocity()

        # initialize pdf for given density/velocity [2/2]
        if init_pdf is None and (init_density is not None or init_velocity is not None):
            self.update_equilibrium()
            self._pdf_cij = self._pdf_eq_cij

        # initialize density boundaries
        self._max_density = np.max(np.sum(self._pdf_cij, axis=0))
        self._min_density = np.min(np.sum(self._pdf_cij, axis=0))

        # initialize plotting
        if plot is None:
            plot = "once" if get_ipython() is not None else False
        if animate is None:
            animate = "once" if get_ipython() is not None else False
        self._plot = plot
        self._plot_width = plot_width
        self._plot_stages = plot_stages
        self._animate = animate
        self._init_plot()

    @property
    def shape(self) -> tuple[int, int]:
        """
        Returns the (Y, X) shape/dimensions of the lattice.
        """
        return self._shape

    @property
    def density(self) -> np.array:
        """
        Returns the (Y, X) shaped density field.
        """
        return self._density_ij

    @property
    def mass(self) -> float:
        """
        Returns the total mass.
        """
        return np.sum(self._density_ij)

    def update_density(self):
        """
        Computes the density at each lattice point.
        """
        # at each lattice point, sum up the pdf over all velocity channels
        self._density_ij = np.sum(self._pdf_cij, axis=0)
        assert np.min(self._density_ij) >= 0, \
            f"negative densities make no sense: {np.min(self._density_ij)}"
        assert self._initial_mass is None or self._initial_mass - self.mass < 1e-6, \
            f"the simulation generates/destroys mass: {self._initial_mass - self.mass}"

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
        self._velocity_aij = np.einsum("cij,ca->aij", self._pdf_cij, self._channel_ca) / \
            np.maximum(self._density_ij, 1e-12)

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
            1. + 3. * vel_dot_c + 4.5 * vel_dot_c ** 2 - 1.5 * v_norm2[np.newaxis, ...])

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

    def step(self, n: int = 1):
        """
        Run one or multiple simulation steps.

        Parameters
        ----------
        n : int
            The number of simulation steps.
        """
        for _ in range(n):
            self.streaming_step()
            self.collision_step()
            self.update_plot()

    def _init_plot(self):
        """
        Initialize the density color plot with ``matplotlib``.
        """
        if not self._plot and not self._animate:
            return
        self._step_i = 0
        self._images = []
        # init stages handling
        if self._plot_stages is None:
            self._plot_stages = [(0, 1), (500, 2), (1000, 5), (2000, 10), (10000, 50), (100000, 100), (1000000, 1000)]
        self._plot_stages_di = self._plot_stages[0][1]
        self._plot_stages_step = self._plot_stages[0][0]
        self._plot_stages.pop(0)
        # plot size
        w = self._plot_width / 25.4  # mm to inch
        h = w * (self._Y / self._X) + 1  # width * ratio + 1 for the title
        # create plot figure
        self._figure, self._ax = plt.subplots(figsize=(w, h))
        # square cells
        self._ax.set_aspect("equal", adjustable="box")
        # render major ticks at center of each cell
        self._ax.invert_yaxis()
        self._ax.xaxis.tick_top()
        self._ax.xaxis.set_label_position("top")
        # self._ax.set_xticks(np.arange(self._X) + 0.5)
        # self._ax.set_yticks(np.arange(self._Y) + 0.5)
        # self._ax.set_xticklabels(np.arange(self._X))
        # self._ax.set_yticklabels(np.arange(self._Y))
        # set axis labels
        self._ax.set_xlabel("X (axis 1)")
        self._ax.set_ylabel("Y (axis 0)")
        # title
        self._title = self._ax.text(0.5, -0.05, "",
                                    size=plt.rcParams["axes.titlesize"],
                                    ha="center", va="top",
                                    transform=self._ax.transAxes)
        # fill plot with initial state
        self.update_plot()
        if self._plot:
            plt.show(block=False)

    def _is_step_plotted(self, i: int) -> bool:
        """
        Implementation of the stages mechanism.
        Checks whether a specific simulation step should be plotted or not.

        Parameters
        ----------
        i : int
            The considered step.

        Returns
        -------
        bool
            True if the step should be plotted, false otherwise.
        """
        # no stages -> plot all steps
        if self._plot_stages is None:
            return True
        # check if we need to proceed to the next stage
        if len(self._plot_stages) > 0 and i >= self._plot_stages[0][0]:
            self._plot_stages_di = self._plot_stages[0][1]
            self._plot_stages.pop(0)
        # is step plotted according to stage?
        if i == self._plot_stages_step:
            # next plotted step in stage
            self._plot_stages_step += self._plot_stages_di
            return True
        # all other steps are not plotted
        return False

    def update_plot(self):
        """
        Update the density color plot. This function should be called exactly once per simulation step.
        """
        if not self._plot and not self._animate:
            return
        if not self._is_step_plotted(self._step_i):
            self._step_i += 1
            return
        # update the max density
        self._max_density = max(self._max_density, np.max(self._density_ij))
        self._min_density = min(self._min_density, np.min(self._density_ij))
        # update heatmap
        img = self._ax.pcolormesh(self.density,
                                  vmin=max(self._min_density, 0),
                                  vmax=max(self._max_density, 0.01),
                                  cmap=plt.cm.Blues)
        # update title
        # (use _ax.text instead of _ax.set_title, because the title would not get animated)
        title = f"step {self._step_i} (x{self._plot_stages_di})" if self._step_i > 0 else "initial state"
        self._title.set_text(title)
        title = self._ax.text(0.5, -0.05, title,
                              size=plt.rcParams["axes.titlesize"],
                              ha="center", va="top",
                              transform=self._ax.transAxes,
                              animated=True)
        self._step_i += 1
        self._step_modulo = 1
        # collect artists for animation
        if self._animate:
            self._images.append([img, title])
        # update plot
        if self._plot:
            self._figure.canvas.draw()
            self._figure.canvas.flush_events()
            if self._plot == "once":
                self._plot = False

    def get_animation(self, interval: float = 0.05) -> animation.Animation:
        """
        Get a ``matplotlib.animation.Animation`` that comprises all performed simulation steps.

        Parameters
        ----------
        interval : float
            The delay between frames in seconds.
        """
        # create animation
        self._title.set_text("")
        return animation.ArtistAnimation(self._figure,
                                         self._images,
                                         interval=interval*1000,
                                         blit=True,
                                         repeat=False)

    def display_animation(self, autoplay: bool = False, **kwargs):
        """
        Display an animation of all performed simulation steps.
        This function is only useful inside a Jupyter/IPython environment.

        Parameters
        ----------
        autoplay : bool
            Whether to inject some JavaScript that automatically starts playing the video upon displaying.

        **kwargs
            Arguments for ``get_animation()``.
        """
        html = self.get_animation(**kwargs).to_jshtml()
        if autoplay:
            # html contains: animation_object_with_hash = new Animation(...);
            # append:        animation_object_with_hash.play_animation();
            html = re.sub(r"(\n( +\w+) = new Animation\([^)]+\);?)", "\\1\n\\2.play_animation();", html)
        display.display(display.HTML(html))

    def save_plot(self, path: str):
        """
        Save the current plot.

        Parameters
        ----------
        path : str
            File path where to save the plot.
        """
        dir = os.path.dirname(path)
        if dir != "":
            os.makedirs(dir, exist_ok=True)
        plt.savefig(path)
