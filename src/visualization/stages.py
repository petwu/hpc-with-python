class PlotStages:
    """
    This class provides constructs to define different stages of plotting.
    A stage consist of two parameters ``(i, s)`` where
    - ``i`` is the index at which the stage starts and
    - ``s`` is the step size inside this stage, that defines which steps are to be plotted

    Example
    -------
    The stages ``[(0, 1), (10, 2), (20, 5)]`` mean:

    - from index 0 to 9, every step is plotted
    - from index 10 to 19, every 2nd step is plotted
    - from index 20 onwards, every 5th step is plotted

    This means, the following steps are plotted: 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, ...
    """
    def __init__(self, stages: list[tuple[int, int]] = None, omega: float = 1.0) -> None:
        """
        Contructor.

        Parameters
        ----------
        stages : list[tuple[int, int]]
            The stages definitions, i.e. a list of ``(start_index, step_size)`` tuples.

        omega : float
            The rate at which the Lattice Boltzmann system is pushed towards the equilibrium: ω=1/τ

            This parameter is only used, if ``stages`` is not defined in order to set an appropriate default value.
        """
        if stages is None:
            # TODO: use omega for the default value
            stages = [(0, 1), (500, 2), (1000, 5), (2000, 10), (10000, 50), (100000, 100), (1000000, 1000)]
        self._stages = stages
        self._step_size = self._stages[0][1]
        self._next_plotted_step = self._stages[0][0]
        self._stages.pop(0)

    @property
    def current_step_size(self) -> int:
        """
        Gives the step size of the current stage.
        """
        return self._step_size

    def is_step_plotted(self, i: int) -> bool:
        """
        Determines, whether a given step should be plotted according to the stages definition.

        Parameters
        ----------
        i : int
            The considered step.

        Returns
        -------
        bool
            ``True``, in case the step should be plotted, ``False`` otherwise.
        """
        # no stages -> plot all steps
        if self._stages is None:
            return True

        # check if we need to proceed to the next stage
        if len(self._stages) > 0 and i >= self._stages[0][0]:
            self._step_size = self._stages[0][1]
            self._stages.pop(0)

        # is step plotted according to stage definition?
        if i == self._next_plotted_step:
            # next plotted step in stage
            self._next_plotted_step += self._step_size
            return True

        # all other steps are not plotted
        return False
