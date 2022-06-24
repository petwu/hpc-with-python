from matplotlib import animation, pyplot as plt
from IPython import display, get_ipython
from .stages import PlotStages
import numpy as np
import os
import re


class DensityPlot:
    """
    Helper class to plot a 2D heatmap of a density field/distribution.
    """

    def __init__(self,
                 X: int,
                 Y: int,
                 plot_size: int = 200,
                 plot_stages: PlotStages = None,
                 vmin: float = None,
                 vmax: float = None,
                 plot: bool = None,
                 animate: bool = None):
        """
        X, Y : int
            Dimensions of the density field.

        plot_size : int
            Size of the plot in mm. This refers to the larger grid size, i.e. the width if X>Y or the height otherwise.

        plot_stages : list[tuple[int, int]]
            See :class:`PlotStages`.

        vmin, vmax : float
            The colorbar range, which should e.g. correspond to the (expected) min/max density values.
            If None, suitable min/max values are automatically chosen.
            The values are updated if required when the plot gets updated.

        plot : bool|"once"
            Whether to call ``matplotlib.pyplot.show()`` in order to show a plot.
            The plot gets updated with each call to ``step()`` automatically.
            Plotting gets done in a non-blocking way.

            If the value is ``once`` instead of a bool, then only the first update is shown.
            This might be useful in Jupyter to show the the initial state and then the final animation.

            Defaults to ``once`` inside Jupyter/IPython and ``True`` otherwise.

        animate : bool
            Whether to store each image in order to create an animation.
            Defaults to ``True`` inside Jupyter/IPython and ``True`` otherwise.
        """
        self._X = X
        self._Y = Y
        self._plot_size = plot_size
        self._plot_stages = plot_stages or PlotStages()
        self._vmin = vmin
        self._vmax = vmax

        if plot is None:
            plot = "once" if get_ipython() is not None else False
        if animate is None:
            animate = get_ipython() is not None
        self._plot = plot
        self._animate = animate

    def init(self, step: int, density: np.ndarray):
        """
        Initialize the density color plot with ``matplotlib``.
        """
        if not self._plot and not self._animate:
            return
        self._images = []
        # plot size
        if self._X >= self._Y:
            w = self._plot_size / 25.4  # mm to inch
            h = w * (self._Y / self._X) + 1  # width * ratio + 1 for the title
        else:
            h = self._plot_size / 25.4 + 1  # mm to inch
            w = (h-1) * (self._X / self._Y)  # width * ratio + 1 for the title
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
        # if only plotting, store a reference to the color mesh and only update the data -> faster
        # (if we are also animating, then we need to create a separate artist each step anyway)
        if self._plot and not self._animate:
            self._pcolormesh = self._ax.pcolormesh(density,
                                                   vmin=max(self._vmin, 0),
                                                   vmax=max(self._vmax, 0.01),
                                                   cmap=plt.cm.Blues)
        # fill plot with initial state
        self.update(step, density)
        if self._plot:
            plt.show(block=False)

    def update(self, step: int, density: np.ndarray):
        """
        Update the density color plot.
        This function should be called exactly once per simulation step.
        """
        if not self._plot and not self._animate:
            return
        if not self._plot_stages.is_step_plotted(step):
            step += 1
            return

        # update the min/max density values
        if self._vmax is None:
            self._vmax = density.max()
        else:
            self._vmax = max(self._vmax, density.max())
        if self._vmin is None:
            self._vmin = density.min()
        else:
            self._vmin = min(self._vmin, density.min())

        # update heatmap
        if self._animate and self._plot:
            img = self._ax.pcolormesh(density,
                                      vmin=max(self._vmin, 0),
                                      vmax=max(self._vmax, 0.01),
                                      cmap=plt.cm.Blues)
        elif self._plot:
            self._pcolormesh.set_array(density)
            self._pcolormesh.set_clim(vmin=max(self._vmin, 0),
                                      vmax=max(self._vmax, 0.01))
        # update title
        # (use _ax.text instead of _ax.set_title, because the title would not get animated)
        title = f"step {step} (x{self._plot_stages.current_step_size})" if step > 0 else "initial state"
        self._title.set_text(title)
        title = self._ax.text(0.5, -0.05, title,
                              size=plt.rcParams["axes.titlesize"],
                              ha="center", va="top",
                              transform=self._ax.transAxes,
                              animated=True)

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
