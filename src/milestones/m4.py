import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import src.boundary as bdry
import src.simulation as sim
from tqdm import trange

from ._common import init_args, validate_args, exponential_steps


def args(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    arg_parser = subparsers.add_parser("m4", help="milestone 4: Couette flow",
                                       description="Implementation of the Couette flow with the following set of "
                                       "boundaries:\n"
                                       "- left, right: periodic boundary conditions\n"
                                       "- top: moving wall\n"
                                       "- bottom: rigid wall\n\n"
                                       "Simulation results are cached under <output_dir>/data.")
    init_args(arg_parser, handler=main, size_x=100, size_y=100, n_steps=40000, omega=1.0)
    arg_parser.add_argument("-u", "--wall_velocity", metavar="u", type=float, default=0.1,
                            help="velocity of the moving wall (top boundary)\n"
                            "values must be >0, which corresponds to a wall moving to the right")
    return arg_parser


def main(args: argparse.Namespace):
    # validate arguments
    validate_args(args)
    if args.wall_velocity <= 0:
        raise ValueError("-u|--wall_velocity must be >0")

    # initialize lattice
    lattice = sim.LatticeBoltzmann(args.size_x, args.size_y, omega=args.omega, viscosity=args.viscosity,
                                   init_density=np.ones((args.size_y, args.size_x)),
                                   boundaries=[bdry.MovingWallBoundaryCondition("t", [0, args.wall_velocity]),
                                               bdry.RigidWallBoundaryCondition("b")])

    # run simulation and take measurements
    print("-- run simulation: couette flow")
    cache_path = f"{args.output_dir}/data/m4_x{args.size_x}_y{args.size_y}_w{args.omega}_Uw{args.wall_velocity}_n{args.n_steps}.npy"
    if os.path.isfile(cache_path):
        print(f"-- use cache: {cache_path}")
        velocity_field = np.load(cache_path)
    else:
        velocity_field = np.empty((args.n_steps+1, args.size_y, args.size_x))
        velocity_field[0] = lattice.velocity[1]
        for i in trange(args.n_steps, **args.tqdm_kwargs):
            lattice.step()
            velocity_field[i+1] = lattice.velocity[1]
        print(f"-- save cache: {cache_path}")
        np.save(cache_path, velocity_field)

    # plot results
    filename = f"{args.output_dir}/m4_velocity_profile_evolution.png"
    print(f"-- save plot: {filename}")
    plot_velocity_profile_evolution(filename, args, velocity_field)

    filename = f"{args.output_dir}/m4_velocity_flow_field_evolution.png"
    print(f"-- save plot: {filename}")
    plot_velocity_flow_field_evolution(filename, args, velocity_field)


def plot_velocity_profile_evolution(filename: str, args: argparse.Namespace, velocity_field: np.ndarray):
    measurement_point = args.size_x//2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.invert_yaxis()
    ax.set_xlabel(f"velocity $u_x(x={measurement_point}, y, t)$")
    ax.set_ylabel("$y$ dimension")
    # measured profile at different time steps
    steps, b = exponential_steps(args.n_steps, 20)
    y_data = np.arange(args.size_y)
    for t in steps:
        x_data = velocity_field[t, :, measurement_point]
        ax.plot(x_data, y_data, color=plt.cm.rainbow(np.power(t/args.n_steps, 1/b)), label=f"$t=${t:,}")
    # analytical solution
    y_data = np.arange(args.size_y+1)
    x_data = np.flip(y_data/args.size_y*args.wall_velocity)
    ax.plot(x_data, y_data, "k-.", label="analytical\nsolution")
    ax.legend(ncol=2)
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


def plot_velocity_flow_field_evolution(filename: str, args: argparse.Namespace, velocity_field: np.ndarray):
    steps, _ = exponential_steps(args.n_steps, 5)
    n_cols = 5
    n_rows = len(steps) // n_cols
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(9, 2))
    axf = ax.flatten()
    axf[0].invert_yaxis()
    # streamplots for different time steps
    x = np.arange(args.size_x)
    y = np.arange(args.size_y)
    v = np.zeros((args.size_y, args.size_y))
    for i, t in enumerate(steps):
        axf[i].set_title(f"$t={t}$")
        if i % n_cols == 0:
            axf[i].set_ylabel("$y$ dimension")
        if i >= n_cols*(n_rows-1):
            axf[i].set_xlabel("$x$ dimension")
        u = velocity_field[t]
        axf[i].streamplot(x, y, u, v, density=0.375,
                          color=u if u.max() != 0 or u.min() != 0 else None, cmap=plt.cm.Greys)
        axf[i].plot([-0.5, args.size_x-0.5], [-0.5, -0.5], color="red", label=f"moving wall")
        axf[i].plot([-0.5, args.size_x-0.5], [args.size_y-0.5, args.size_y-0.5], color="blue", label="rigid wall")
    # legend right of the first row
    axf[n_cols-1].legend(bbox_to_anchor=(1.8, 0.5), loc="center left")
    fig.tight_layout()
    # colorbar right of the last row
    p = axf[-1].get_position()
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0.0, args.wall_velocity), plt.cm.Greys),
                 ax=fig.add_axes([p.x0+0.2*p.width, p.y0, p.width, p.height], visible=False),
                 label="velocity $u$")
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()
