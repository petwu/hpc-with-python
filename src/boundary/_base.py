from __future__ import annotations
import abc
import enum
import numpy as np
import src.simulation as sim


class BoundaryMeta(enum.EnumMeta):
    """
    Meta class for :class:`Boundary`.
    """
    def __call__(cls, value, *args, **kwargs):
        if isinstance(value, str):
            bl = Boundary(0)
            for char in value.lower():
                bl |= {
                    "t": Boundary.TOP,
                    "b": Boundary.BOTTOM,
                    "l": Boundary.LEFT,
                    "r": Boundary.RIGHT,
                }.get(char, Boundary(0))
            return bl
        return super().__call__(value, *args, **kwargs)


class Boundary(enum.Flag, metaclass=BoundaryMeta):
    """
    Enum class for representing lattice boundaries.

    Multiple values can be combined in a flag-like manner, e.g.: ``Boundary.TOP | Boundary.BOTTOM``
    to represent both the top and bottom boundary or ``if flag & Boundary.TOP: ...`` to check whether
    the variable ``flag`` includes the top boundary.

    You can also pass a string representation of the desired boundaries to the ``__call__()`` function, e.g.
    ``Boundary("tb")``. The string should be a combination of the first letters of the possible enum values,
    i.e. ``t`` for ``Boundary.TOP``, ``b`` for ``Boundary.BOTTOM`` etc. Unknown letters are ignored.
    """
    TOP = 1
    BOTTOM = 2
    LEFT = 4
    RIGHT = 8

    def __iter__(self):
        # built-in support with Python >=3.11 (expected to be released in 2022-10)
        for b in Boundary:
            if b & self:
                yield b

    def __len__(self):
        # built-in support with Python >=3.11 (expected to be released in 2022-10)
        return self.value.bit_count()


class BaseBoundaryCondition(abc.ABC):
    """
    Abstract base class for all boundary conditions.
    """

    def __init__(self, boundaries: str | Boundary, before_streaming: bool):
        """
        Parameters
        ----------
        boundaries : str|Boundary
            A flag indicating the boundaries of the boundaties.

            Examples:
            ``Boundary.TOP|Boundary.BOTTOM`` for two boundaries at the top and bottom of the lattice.
            Or the equivalent shortcut string ``tb`` would be possible to.
        """
        self._before_streaming = before_streaming
        self._boundaries = Boundary(boundaries)
        self.__initialized = False

        # map each boundary to the indices required for the reflection/back bouncing:
        self._boundary_indices = {
            # boundary: (opposite/reflected channels, source channels, y, x)
            # 6   2   5
            #   \ | /
            # 3 – 0 – 1
            #   / | \
            # 7   4   8
            Boundary.TOP:    (np.array([4, 7, 8]), np.array([2, 5, 6]),  0, slice(None)),  # first row
            Boundary.BOTTOM: (np.array([2, 5, 6]), np.array([4, 7, 8]), -1, slice(None)),  # last row
            Boundary.LEFT:   (np.array([1, 8, 5]), np.array([3, 6, 7]), slice(None),  0),  # first column
            Boundary.RIGHT:  (np.array([3, 7, 6]), np.array([1, 5, 8]), slice(None), -1),  # last column
            # note: slice(None) is equivalent to :
        }

    def __str__(self) -> str:
        return f"{type(self).__name__.replace('BoundaryCondition', '')}({self.boundaries})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def before_streaming(self) -> bool:
        """
        A boolean property indicating whether the boundary condition should be applied
        before (``True``) or after (``False``) the streaming operator.
        """
        return self._before_streaming

    @property
    def boundaries(self) -> Boundary:
        """
        The ``boundaries`` value passed to the constructor.
        """
        return self._boundaries

    @property
    def boundary_indices(self) -> dict[Boundary, tuple]:
        """
        A dictionary mapping the boundary to a tuple of indices/slices:
        (opposite/reflected channels, source channels, y, x)
        """
        return self._boundary_indices

    def initialize(self, lattice: sim.LatticeBoltzmann, boundaries: str | Boundary = None):
        """
        Initilization method. Can be used e.g. to precompute specific constant values or similar.
        This may only be executed once.

        Parameters
        ----------
        lattice : LatticeBoltzmann
            The concerned lattice instance.

        boundaries : str | Boundary
            Override the ``boundaries`` parameter passed to the constructor.
            This may be useful for parallel implementations using MPI.
        """
        if self.__initialized:
            raise RuntimeError(f"{type(self).__name__}.{self.initialize.__name__}() may only be called once")

        if boundaries is not None:
            self._boundaries = Boundary(boundaries)

        self.__initialized = True

    @abc.abstractmethod
    def apply(self, lattice: sim.LatticeBoltzmann):
        """
        Applies the boundary conditions behavior to the lattice by modifying its pdf.

        Parameters
        ----------
        lattice : LatticeBoltzmann
            The concerned lattice instance.
        """
        if not self.__initialized:
            raise RuntimeError(f"{type(self).__name__}.{self.initialize.__name__}() needs to be called once before " +
                               f"calling {type(self).__name__}.{self.apply.__name__}()")
