import argparse
import typing
import os

import numpy as np
from scipy.special import lambertw


def init_args(arg_parser: argparse.ArgumentParser,
              handler: typing.Callable,
              size_x: int = None,
              size_y: int = None,
              n_steps: int = None,
              omega: float = None,
              viscosity: float = None):
    """
    Initialize a :class:`argparse.ArgumentParser` object with arguments common to all milestones.

    Parameters
    ----------
    arg_parser: argparse.ArgumentParser
        The argument parser to initialize.

    handler: function
        The function that gets set as ``handler`` property on the argument parser and can then be called.

    size_x, size_y, n_steps : int
    omega, viscosity : float
        Default values for the respective arguments.
    """

    # HACK: increase the width of the first column of the --help output using a non-public API
    arg_parser.formatter_class = lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=48)

    # set handler and other defaults
    arg_parser.set_defaults(handler=handler,
                            tqdm_kwargs={"bar_format": "{l_bar}{bar:10}{r_bar}",
                                         "leave": False})

    # define common arguments
    arg_parser.add_argument("-x", "--size_x", metavar="<X>",
                            type=int, required=size_x is None, default=size_x,
                            help=_help_default("size of the grid in x direction", size_x))
    arg_parser.add_argument("-y", "--size_y", metavar="<Y>",
                            type=int, required=size_y is None, default=size_y,
                            help=_help_default("size of the grid in y direction", size_y))
    arg_parser.add_argument("-n", "--n_steps", metavar="<N>",
                            type=int, required=n_steps is None, default=n_steps,
                            help=_help_default("number of simulation steps", n_steps))
    arg_parser.add_argument("-w", "--omega", metavar="<W>",
                            type=float, default=omega,
                            help=_help_default("1/τ: relaxation time constant, in the range [0.1; 1.9]", omega))
    arg_parser.add_argument("-v", "--viscosity", metavar="<V>",
                            type=float, default=viscosity,
                            help=_help_default("viscosity constant of the fluid, in the range [0.01, 3.17]\n"
                                               "if specified, this overrides -w|--omega", viscosity))
    arg_parser.add_argument("-p", "--dpi", metavar="<dpi>",
                            type=int, default=300,
                            help="DPI (dots per inch) value for saved plots; higher values increase the resolution "
                            "(default: 300)")
    arg_parser.add_argument("-o", "--output_dir", metavar="<dir>",
                            type=str, default=f"out",
                            help="path to the output directory, where e.g. plots or data gets emitted")


def _help_default(help: str, default: any) -> str:
    """
    Optionally append a non-``None`` default value to the help test of an argument.

    Parameters
    ----------
    help : str
        Help text for the argument.

    default : any
        Default value for the argument.

    Returns
    -------
    str
        ``help`` with a note about the default value appended in case the default value is not ``None``.
    """
    return help if default is None else help + " (default: %(default)s)"


def validate_args(args: argparse.Namespace):
    """
    Validate the values for those arguments defined by :method:`init_args`.
    For additional arguments, one must do the validation manually.
    """

    # size_x, size_y n_steps
    if not isinstance(args.size_x, int) or args.size_x <= 0:
        raise ValueError("-x|--size_x must be an integer > 0")
    if not isinstance(args.size_y, int) or args.size_y <= 0:
        raise ValueError("-y|--size_y must be an integer > 0")
    if not isinstance(args.n_steps, int) or args.n_steps <= 0:
        raise ValueError("-n|--n_steps must be an integer > 0")

    # omega, viscosity
    if not args.omega and not args.viscosity:
        raise ValueError("at least one of -w|--omega or -v|--viscosity must be specified")
    if args.omega:
        omega = args.omega
        if not isinstance(omega, list):
            omega = [omega]
        for o in omega:
            if not isinstance(o, float) or not 0.1 <= o <= 1.9:
                raise ValueError("-w|--omega must be a float in the range [0.1; 1.9]")
    if args.viscosity:
        viscosity = args.viscosity
        if not isinstance(viscosity, list):
            viscosity = [viscosity]
        for v in viscosity:
            if not isinstance(v, float) or not 0.01 <= v <= 3.17:
                raise ValueError("-v|--viscosity must be a float in the range [0.01; 3.17]")

    # output_dir
    os.makedirs(f"{args.output_dir}/data", exist_ok=True)


def exponential_steps(end: int, n: int) -> np.ndarray:
    """
    Generate a sequence of exponetionally increasing steps by solving the following equation::

        end^(1/b) / b = n-1

    Parameters
    ----------
    end : int
        The max. value.

    n : int
        The number of steps.

    Returns
    -------
    tuple[numpy.ndarray, float]
        A 2-tuple of:
            - an 1D array of size ``n`` containing increasing values in the range ``[0; end]``
            - ``b``
    """
    # steps with exponentially increasing step size
    # solve n_steps^(1/b)/b=20 for b to plot exactly 20 steps --> 20 + last = 21 lines
    b = np.log(end) / lambertw((n-1) * np.log(end)).real + 1e-12
    steps = np.append((np.arange(np.power(end, 1/b), step=b)**b).astype(int), end)
    return steps, b


def rnd_tex(var: str, val: float, decimals: int) -> str:
    """
    Check if a number needs to be rounded and return an appropriate LaTeX expression with either an equal or an
    approximation sign.

    Parameters
    ----------
    var : str
        Name of the variable.

    val : float
        Number to be rounded.

    decimals : int
        Number of digits after the decimal place.

    Return
    ------
    str
        A LaTex expression, e.g. ``$var = val$`` or ``$var \\approx round(val)$``.
    """
    val_r = round(val, decimals)
    if val - val_r == 0:
        return f"${var} = {val}$"
    return f"${var} \\approx {val:.{decimals}f}$"


def omega_to_viscosity(omega: float | np.ndarray) -> float | np.ndarray:
    """
    Convert omega (relaxation rate) to viscosity values.
    """
    return 1/3 * (1/np.array(omega) - 0.5)
