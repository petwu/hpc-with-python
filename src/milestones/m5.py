import argparse

import matplotlib.pyplot as plt
import numpy as np
import src.boundary as bdry
import src.simulation as sim
from tqdm import trange

from .common import init_args, validate_args, exponential_steps


def args(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    arg_parser = subparsers.add_parser("m5", help="milestone 5: Poiseuille flow",
                                       description="Implementation of the Poiseuille flow with the following set of "
                                       "boundaries:\n"
                                       "- left, right: periodic boundary conditions with pressure gradient\n"
                                       "- top, bottom: rigid wall\n"
                                       "as well as a pressure gradient between the inlet (left) and outlet (right).")
    init_args(arg_parser, handler=main, size_x=120, size_y=50, n_steps=10000, omega=1.0)
    arg_parser.add_argument("-d", "--density_gradient", nargs=2, metavar=("i", "o"),
                            type=float, default=[1.005, 0.995],
                            help="linear density gradient defined by an inlet (i) and outlet (o) density\n"
                            "note: density ρ and pressure p are linearly related by p=ρ·c² with c=1/3")
    return arg_parser


def main(args: argparse.Namespace):
    # validate arguments
    validate_args(args)

    # create lattice
    lattice = sim.LatticeBoltzmann(args.size_x, args.size_y, omega=args.omega, viscosity=args.viscosity,
                                   init_density=np.ones((args.size_y, args.size_x)),
                                   boundaries=[bdry.RigidWallBoundaryCondition("tb"),
                                               bdry.PeriodicPressureGradientBoundaryCondition("h", args.density_gradient)])

    # run simulation and take measurements
    print("-- run simulation: Poiseuille flow")
    velocity_field = np.empty((args.n_steps+1, 2, args.size_y, args.size_x))
    velocity_field[0] = lattice.velocity
    for i in trange(args.n_steps, **args.tqdm_kwargs):
        lattice.step()
        velocity_field[i+1] = lattice.velocity

    # plot results
    filename = f"{args.output_dir}/m5_velocity_profile_evolution.png"
    print(f"-- save plot: {filename}")
    plot_velocity_profile_evolution(filename, args, velocity_field, lattice.viscosity, lattice.density)

    filename = f"{args.output_dir}/m5_velocity_field.png"
    print(f"-- save plot: {filename}")
    plot_velocity_field(filename, args, velocity_field)

    filename = f"{args.output_dir}/m5_density_along_centerline.png"
    print(f"-- save plot: {filename}")
    plot_density_gradient(filename, args, lattice.density)


def plot_velocity_profile_evolution(filename: str,
                                    args: argparse.Namespace,
                                    velocity_field: np.ndarray,
                                    viscosity: float,
                                    density: np.ndarray):
    measurement_point = args.size_x//2
    fig, ax = plt.subplots(figsize=(10, 5.1))
    ax.invert_yaxis()
    ax.set_xlabel(f"velocity $u_x(x={measurement_point}, y, t)$")
    ax.set_ylabel("$y$ dimension")
    # measured velocity profile at different time steps
    steps, b = exponential_steps(args.n_steps, 16)
    y_data = np.arange(args.size_y)
    for t in steps:
        x_data = velocity_field[t, 1, :, measurement_point]
        ax.plot(x_data, y_data, color=plt.cm.rainbow(np.power(t/args.n_steps, 1/b)), label=f"$t=${t:,}")
    # analytical solution
    y_data = np.arange(args.size_y+1)
    pressure_derivative_x = (args.density_gradient[1] - args.density_gradient[0]) / 3.0 / args.size_x  # p=ρ·c² w/ c=1/3
    dynamic_viscosity = viscosity * density[:, measurement_point].mean()
    analytical_solution = - 0.5 / dynamic_viscosity * pressure_derivative_x * y_data * (args.size_y - y_data)
    ax.plot(analytical_solution, y_data-0.5, "k-.", linewidth=2, label="analytical\nsolution")
    ax.set_xlim(xmax=np.max(analytical_solution)*1.25)
    ax.legend()
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


def plot_velocity_field(filename: str, args: argparse.Namespace, velocity_field: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(f"$t={args.n_steps}$")
    ax.set_ylabel("$y$ dimension")
    ax.set_xlabel("$x$ dimension")
    ax.invert_yaxis()
    x, y, v, u = np.arange(args.size_x), np.arange(args.size_y), velocity_field[-1, 0], velocity_field[-1, 1]
    ax.streamplot(x, y, u, v, color=u, density=0.9)
    ax.plot([-0.5, args.size_x-0.5], [-0.5, -0.5], color="red")
    ax.plot([-0.5, args.size_x-0.5], [args.size_y+0.5, args.size_y+0.5], color="red", label="rigid wall")
    ax.legend(bbox_to_anchor=(1.225, 1.025), loc="upper left")
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(u.min(), u.max()), plt.cm.viridis),
                 ax=ax, label="velocity magnitude $|u|$")
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


def plot_density_gradient(filename: str, args: argparse.Namespace, density: np.ndarray):
    plt.figure()
    plt.xlabel("$x$ dimension")
    plt.ylabel("density $\\rho$")
    plt.plot(density[args.size_y//2, 1:-1], label=f"$\\rho(y={args.size_y//2}, x, t={args.n_steps})$")
    plt.legend()
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()
