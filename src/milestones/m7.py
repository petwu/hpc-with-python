import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import src.boundary as bdry
import src.simulation as sim

from ._common import init_args, rnd_tex, validate_args


def args(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    arg_parser = subparsers.add_parser("m7", help="milestone 7: sliding lid (parallel)",
                                       description="Implementation of the sliding lid experiment with the following "
                                       "set of boundaries:\n"
                                       "- left, right, bottom: rigid wall\n"
                                       "- top: moving wall",
                                       conflict_handler="resolve")
    init_args(arg_parser, handler=main, size_x=300, size_y=300, n_steps=100000, viscosity=0.03)
    arg_parser.add_argument("-u", "--wall_velocity", metavar="U",
                            type=float, default=0.1,
                            help="velocity of the moving wall")
    arg_parser.add_argument("-d", "--decomposition", metavar=("X", "Y"),
                            type=int, nargs=2, default=[-1, -1],
                            help="domain decomposition")
    return arg_parser


def main(args: argparse.Namespace):
    # validate arguments
    validate_args(args)
    import mpi4py.MPI as mpi
    if mpi.COMM_WORLD.Get_size() == 1:
        args.decomposition = None

    # initialize lattice
    lattice = sim.LatticeBoltzmann(args.size_x, args.size_y, omega=args.omega, viscosity=args.viscosity,
                                   decomposition=args.decomposition,
                                   init_density=np.ones((args.size_y, args.size_x)),
                                   boundaries=[bdry.RigidWall("lrb"),
                                               bdry.MovingWall("t", [0, args.wall_velocity])])

    stem = "_".join(["m7",
                     f"x{args.size_x}",
                     f"y{args.size_y}",
                     f"d{f'{lattice.mpi.decomposition[0]}x{lattice.mpi.decomposition[1]}' if lattice.is_parallel else 'serial'}",
                     f"n{args.n_steps}",
                     f"v{args.viscosity}" if args.viscosity else f"w{args.omega}",
                     f"Uw{args.wall_velocity}"])

    # run simulation
    if not lattice.is_parallel or lattice.mpi.rank == 0:
        print("-- run simulation")
    if (lattice.is_parallel
        and os.path.isfile(f"{args.output_dir}/data/{stem}_u.npy")
            and os.path.isfile(f"{args.output_dir}/data/{stem}_v.npy")):
        if lattice.mpi.rank == 0:
            print(f"-- use cache: {args.output_dir}/data/{stem}_u.npy")
            print(f"-- use cache: {args.output_dir}/data/{stem}_v.npy")
    elif not lattice.is_parallel and os.path.isfile(f"{args.output_dir}/data/{stem}.npy"):
        print(f"-- use cache: {args.output_dir}/data/{stem}.npy")
    else:
        lattice.step(args.n_steps, progress=True, **args.tqdm_kwargs)
        if lattice.is_parallel:
            if lattice.mpi.rank == 0:
                print(f"-- save cache: {args.output_dir}/data/{stem}_u.npy + {args.output_dir}/data/{stem}_v.npy")
            lattice.mpi.save_mpiio_2d(f"{args.output_dir}/data/{stem}_u.npy",
                                      lattice.velocity[1][lattice.mpi.physical_domain])
            lattice.mpi.save_mpiio_2d(f"{args.output_dir}/data/{stem}_v.npy",
                                      lattice.velocity[0][lattice.mpi.physical_domain])
        else:
            print(f"-- save cache: {args.output_dir}/data/{stem}.npy")
            np.save(f"{args.output_dir}/data/{stem}.npy", lattice.velocity)

    # save results
    if lattice.is_parallel:
        grid = lattice.mpi.get_grid(rank=0)
        if lattice.mpi.rank == 0:
            filename = f"{args.output_dir}/m7_domain_decomposition.png"
            print(f"-- save plot: {filename}")
            plot_domain_decomposition(filename, args, grid)

    if not lattice.is_parallel or lattice.mpi.rank == 0:
        if lattice.is_parallel:
            velocity = np.array([np.load(f"{args.output_dir}/data/{stem}_v.npy"),
                                 np.load(f"{args.output_dir}/data/{stem}_u.npy")])
        else:
            velocity = np.load(f"{args.output_dir}/data/{stem}.npy")
        filename = f"{args.output_dir}/{stem}.png"
        print(f"-- save plot: {filename}")
        plot_velocity_field(filename, args, lattice, velocity)


def plot_velocity_field(filename: str, args: argparse.Namespace, lattice: sim.LatticeBoltzmann, velocity: np.ndarray):
    L = max(args.size_x, args.size_y)
    Re = L * args.wall_velocity / lattice.viscosity

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ax.set_xlim(0, args.size_x-1)
    ax.set_ylim(0, args.size_y-1)
    ax.invert_yaxis()
    ax.set_ylabel("$y$ dimension")
    ax.set_xlabel("$x$ dimension")
    x, y = np.arange(args.size_x), np.arange(args.size_y)
    v, u = velocity
    norm = plt.Normalize(0, args.wall_velocity)
    ax.streamplot(x, y, u, v,
                  color=np.sqrt(u**2 + v**2) if u.sum() and v.sum() else None,
                  norm=norm,
                  density=2.0)
    fig.colorbar(plt.cm.ScalarMappable(norm, plt.cm.viridis), ax=ax,
                 fraction=0.05, label="velocity magnitude $|u|$")
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


def plot_domain_decomposition(filename: str, args: argparse.Namespace, grid: np.ndarray):
    nx, ny = grid.shape
    fig, ax = plt.subplots(figsize=(6.4, 4.8*ny/nx))
    ax.set_xlabel("axis 1 $(x)$")
    ax.set_ylabel("axis 0 $(y)$")
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.set_xticks(np.arange(nx))
    ax.set_yticks(np.arange(ny))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.invert_yaxis()
    ax.grid(visible=True, which="minor")
    for coords, rank in np.ndenumerate(grid):
        ax.annotate(f"$\\bf rank~{rank}$\n{coords}", coords, ha="center", va="center")
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()
