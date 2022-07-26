import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from src.simulation import LatticeBoltzmann
from tqdm import trange

from ._common import init_args, validate_args


def args(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    arg_parser = subparsers.add_parser("m3", help="milestone 3: shear wave decay",
                                       description="Perform two shear wave decay experiments with:\n"
                                       "1. sinusoidal density\n"
                                       "2. sinusoidal velocity\n\n"
                                       "Note:\n"
                                       "For the 2nd experiment the --size_x and --size_y arguments get flipped, since "
                                       "compared to the 1st experiment the sine gets rotated by 90°.")
    init_args(arg_parser, handler=main, size_x=80, size_y=20, n_steps=10000, omega=1.3)
    arg_parser.add_argument("-e", "--epsilon", metavar=("p", "u"), type=float, nargs=2, default=[0.01, 0.1],
                            help="amplitudes for the sinusoidal density (p) and velocity (u) perturbations")
    arg_parser.add_argument("-r", "--rho0", metavar="p", type=float, default=0.5,
                            help="base density (only for the sinusoidal density perturbation)")
    return arg_parser


def main(args: argparse.Namespace):
    # validate arguments
    validate_args(args)

    # simulate for different omega 0.1...1.9
    omega_i = np.arange(0.1, 2.0, 0.1).round(1)
    if not args.omega in omega_i:
        omega_i = np.sort(np.append(omega_i, args.omega))

    # experiment 1: sinusodial density (sd_*)
    sd_experiment(args, omega_i)

    # flip (x,y) because the sine for experiment 2 is rotated by 90°
    args.size_x, args.size_y = args.size_y, args.size_x

    # experiment 2: sinusoidal velocity (sv_*)
    sv_experiment(args, omega_i)


def perturbation_formula(a0: float, e: float, v: float | np.ndarray, l: int, t: int | np.ndarray) -> np.ndarray:
    """
    Calculate the analytical solution for the decreasing density or velocity perturbation over time.

    Parameters
    ----------
    a0 : float
        Constant offset, e.g. base density.

    e : float
        Initial perturbation amplitude.

    v : float | numpy.ndarray
        Kinematic viscosity.

    l : int
        Domain size.

    t : int | numpy.ndarray
        Time point to evaluate.
    """
    return a0 + e * np.exp(-v * (2*np.pi/l)**2 * t)


def sd_experiment(args: argparse.Namespace, omega_i: np.ndarray):
    """
    Run the shear wave decay experiment for a sinusoidal density and c constant velocity of 0.
    """
    if args.rho0 - args.epsilon[0] < 0:
        raise ValueError("rho0 - elpsion must be >=0, otherwise this results in negative densities")

    # initialize density
    density_x = args.rho0 + args.epsilon[0] * np.sin(2*np.pi*np.arange(args.size_x)/args.size_x)
    density = np.tile(density_x, (args.size_y, 1))
    # use the peak position of the initial sine as measurement point
    measurement_point = np.argmax(density_x)

    # simulate for different omega
    o_idx = np.argwhere(omega_i == args.omega)[0, 0]
    decay_ij = np.empty((omega_i.shape[0], args.n_steps+1))
    viscosity_i = np.empty_like(omega_i)
    sine_ij = np.empty((args.n_steps+1, args.size_x))
    print("-- run simulation: sinusoidal density")
    for i, omega in enumerate(omega_i):
        # create lattice
        lattice = LatticeBoltzmann(args.size_x, args.size_y, omega=omega, viscosity=args.viscosity,
                                   init_density=density)
        # run simulation and take measurements
        viscosity_i[i] = lattice.viscosity
        decay_ij[i, 0] = lattice.density[0, measurement_point]
        if omega == args.omega:
            sine_ij[0] = lattice.density[0]
        for j in trange(args.n_steps, desc=f"lattice {i+1}/{omega_i.shape[0]}, omega={omega}", **args.tqdm_kwargs):
            lattice.step()
            decay_ij[i, j+1] = lattice.density[0, measurement_point]
            if omega == args.omega:
                sine_ij[j+1] = lattice.density[0]
    decay0_ij = decay_ij - args.rho0

    # plot results
    filename = f"{args.output_dir}/m3_density_decay_sine_evolution.png"
    print(f"-- save plot: {filename}")
    sd_plot_decay_evolution(filename, args, decay_ij[o_idx], sine_ij)

    filename = f"{args.output_dir}/m3_density_decay_single_omega.png"
    print(f"-- save plot: {filename}")
    sd_plot_decay_single(filename, args, decay_ij[o_idx], viscosity_i[o_idx], measurement_point)

    filename = f"{args.output_dir}/m3_density_decay_different_omegas_normalized.png"
    print(f"-- save plot: {filename}")
    sd_plot_decay_multiple(filename, args, omega_i, decay0_ij, viscosity_i, measurement_point)

    filename = f"{args.output_dir}/m3_density_viscosity.png"
    print(f"-- save plot: {filename}")
    sd_plot_viscosity(filename, args, omega_i, decay0_ij, viscosity_i)


def sd_plot_decay_evolution(filename: str, args: argparse.Namespace, decay_i: np.ndarray, sine_ij: np.ndarray):
    plt.figure(figsize=(6.5, 5))
    plt.xlabel("$x$ dimension")
    plt.ylabel("density $\\rho_y(x,t)$")
    peaks = np.append([0], argrelextrema(decay_i, np.greater)[0])[:15]
    for p in peaks:
        plt.plot(sine_ij[p], color=plt.cm.rainbow(p/peaks[-1]), label=f"$t={p}$")
    plt.plot(np.ones_like(sine_ij[0])*args.rho0, "k-.", label="$\\rho_0$")
    plt.legend(ncol=2)
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


def sd_plot_decay_single(filename: str,
                         args: argparse.Namespace,
                         decay_i: np.ndarray,
                         viscosity: float,
                         measurement_point: int):
    plt.figure(figsize=(15, 6))
    plt.xlabel("time $t$")
    plt.ylabel("density $\\rho$")
    plt.plot(np.ones_like(decay_i)*args.rho0, label="$\\rho_0$")
    plt.plot(decay_i, label=f"$\\rho_y(x={measurement_point}, t)$")
    plt.plot(perturbation_formula(args.rho0, args.epsilon[0], viscosity, args.size_x, np.arange(len(decay_i))),
             label="analytical solution")
    plt.legend()
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


def sd_plot_decay_multiple(filename: str,
                           args: argparse.Namespace,
                           omega_i: np.ndarray,
                           decay_ij: np.ndarray,
                           viscosity_i: np.ndarray,
                           measurement_point: int):
    plt.figure(figsize=(15, 6))
    plt.xlabel("time $t$")
    plt.ylabel("normalized density perturbation "
               f"$\\rho_y(x={measurement_point}, t)\,/\,\\rho_y(x={measurement_point}, t=0)$")
    legend = np.empty((len(omega_i)+2, 2), dtype=tuple)
    for i, omega in enumerate(omega_i):
        # find measurements peaks, since those need to align with the exponential
        peaks = np.append([0], argrelextrema(decay_ij[i], np.greater)[0])
        norm = decay_ij[i, 0]
        p1, = plt.plot(peaks, decay_ij[i][peaks]/norm, "x")
        # analytical solution
        p2, = plt.plot(perturbation_formula(0, args.epsilon[0], viscosity_i[i], args.size_x, np.arange(len(decay_ij[i]))) / norm,
                       p1.get_color())
        # legend
        legend[i] = [(p1, p2), f"$\omega = {omega}$"]
    legend[-2] = [plt.plot([], [], "k-")[0], "analytical solutions"]
    legend[-1] = [plt.plot([], [], "kx")[0], "measurements (peaks only)"]
    l1 = plt.legend(legend[-2:, 0], legend[-2:, 1], ncol=2, loc="upper right")
    plt.legend(legend[:-2, 0], legend[:-2, 1], ncol=3, loc="upper right", bbox_to_anchor=(0, 0, 1, 0.93))
    plt.gca().add_artist(l1)
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


def sd_plot_viscosity(filename: str,
                      args: argparse.Namespace,
                      omega_i: np.ndarray,
                      decay_ij: np.ndarray,
                      viscosity_i: np.ndarray):
    viscosity_measured_i = np.empty_like(viscosity_i)
    for i, _ in enumerate(omega_i):
        peaks = np.append([0], argrelextrema(decay_ij[i], np.greater)[0])
        viscosity_measured_i[i] = curve_fit(lambda t, v: perturbation_formula(0, args.epsilon[0], v, args.size_x, t),
                                            xdata=peaks,
                                            ydata=decay_ij[i, peaks])[0][0]

    plt.figure()
    plt.xlabel("relaxation rate $\omega$")
    plt.ylabel("kinematic viscosity $\\nu$")
    plt.xticks(omega_i)
    plt.plot(omega_i, viscosity_i, label="analytical")
    plt.plot(omega_i, viscosity_measured_i, label="empirical")
    plt.legend()
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


def sv_experiment(args: argparse.Namespace, omega_i: np.ndarray):
    """
    Run the shear wave decay experiment for a sinusoidal velocity and a constant density of 1.
    """

    # initialize velocity and density
    velocity_y = args.epsilon[1] * np.sin(2*np.pi*np.arange(args.size_y)/args.size_y)
    velocity = np.array([np.zeros((args.size_y, args.size_x)),
                        np.tile(velocity_y[:, np.newaxis], (1, args.size_x))])
    density = np.ones((args.size_y, args.size_x))
    measurement_point = np.argmax(velocity_y)

    # run simulation for different omega
    decay_ij = np.empty((len(omega_i), args.n_steps+1))
    viscosity_i = np.empty((len(omega_i)))
    sine_ij = np.empty((args.n_steps+1, args.size_y))
    print("-- run simulation: sinusoidal velocity")
    for i, omega in enumerate(omega_i):
        lattice = LatticeBoltzmann(args.size_x, args.size_y, omega=omega, viscosity=args.viscosity,
                                   init_density=density, init_velocity=velocity)
        viscosity_i[i] = lattice.viscosity
        if omega == args.omega:
            sine_ij[0] = lattice.velocity[1, :, 0]
        # run simulation and take measurement
        decay_ij[i, 0] = lattice.velocity[1, measurement_point, 0]
        for j in trange(args.n_steps, desc=f"lattice {i+1}/{omega_i.shape[0]}, omega={omega}", **args.tqdm_kwargs):
            lattice.step()
            decay_ij[i, j+1] = lattice.velocity[1, measurement_point, 0]
            if omega == args.omega:
                sine_ij[j+1] = lattice.velocity[1, :, 0]

    filename = f"{args.output_dir}/m3_velocity_sine_evolution.png"
    print(f"-- save plot: {filename}")
    sv_plot_decay_evolution(filename, args, sine_ij)

    filename = f"{args.output_dir}/m3_velocity_decay_different_omegas_normalized.png"
    print(f"-- save plot: {filename}")
    sv_plot_decay_multiple(filename, args, omega_i, decay_ij, viscosity_i, measurement_point)

    filename = f"{args.output_dir}/m3_velocity_viscosity.png"
    print(f"-- save plot: {filename}")
    sv_plot_viscosity(filename, args, omega_i, decay_ij, viscosity_i)


def sv_plot_decay_evolution(filename: str, args: argparse.Namespace, sine_ij: np.ndarray):
    plt.figure(figsize=(6.5, 5))
    plt.gca().invert_yaxis()
    plt.xlabel("velocity $u_x(y,t)$")
    plt.ylabel("$y$ dimension")
    y_axis = np.arange(args.size_y)
    frames = np.arange(0, args.n_steps+1, args.n_steps/15).round().astype(int)
    for t in frames:
        plt.plot(sine_ij[t], y_axis, color=plt.cm.rainbow(t/frames[-1]), label=f"$t={t}$")
    plt.legend(ncol=2)
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


def sv_plot_decay_multiple(filename: str,
                           args: argparse.Namespace,
                           omega_i: np.ndarray,
                           decay_ij: np.ndarray,
                           viscosity_i: np.ndarray,
                           measurement_point: int):
    plt.figure(figsize=(15, 6))
    plt.xlabel("time $t$")
    plt.ylabel(f"normalized velocity perturbation $u_x(y={measurement_point}, t)\,/\,\\varepsilon$")
    legend = np.empty((len(omega_i)+2, 2), dtype=tuple)
    for i, omega in enumerate(omega_i):
        # measurements
        norm = decay_ij[i, 0]
        p1, = plt.plot(decay_ij[i]/norm, alpha=0.4)
        # analytical solution
        sol = perturbation_formula(0, args.epsilon[1], viscosity_i[i], args.size_y, np.arange(len(decay_ij[i]))) / norm
        p2, = plt.plot(sol, linestyle="-.", dashes=(10, 10, 2, 10, 2, 10), color=p1.get_color())
        # legend
        legend[i] = [(p1, p2), f"$\omega = {omega}$"]
    legend[-2] = [plt.plot([], [], "k-.")[0], "analytical solutions (dashed)"]
    legend[-1] = [plt.plot([], [], "k", alpha=0.4)[0], "measurements (semi-transparent)"]
    l1 = plt.legend(legend[-2:, 0], legend[-2:, 1], ncol=2, loc="upper right")
    plt.legend(legend[:-2, 0], legend[:-2, 1], ncol=3, loc="upper right", bbox_to_anchor=(0, 0, 1, 0.93))
    plt.gca().add_artist(l1)
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


def sv_plot_viscosity(filename: str,
                      args: argparse.Namespace,
                      omega_i: np.ndarray,
                      decay_ij: np.ndarray,
                      viscosity_i: np.ndarray):
    viscosity_measured = np.empty_like(viscosity_i)
    for i, _ in enumerate(omega_i):
        viscosity_measured[i] = curve_fit(lambda t, v: perturbation_formula(0, args.epsilon[1], v, args.size_y, t),
                                          xdata=np.arange(decay_ij[i].shape[0]),
                                          ydata=decay_ij[i])[0][0]

    plt.figure(figsize=(8, 6))
    plt.xlabel("relaxation rate $\omega$")
    plt.ylabel("kinematic viscosity $\\nu$")
    plt.xticks(omega_i)
    plt.plot(omega_i, viscosity_i, label="analytical")
    plt.plot(omega_i, viscosity_measured, label="empirical")
    plt.legend()
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()
