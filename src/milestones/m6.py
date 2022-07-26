import argparse
import os

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import src.boundary as bdry
import src.simulation as sim
import src.visualization as visz
from tqdm import tqdm, trange

from .common import init_args, omega_to_viscosity, rnd_tex, validate_args


def args(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    arg_parser = subparsers.add_parser("m6", help="milestone 6: sliding lid (serial)",
                                       description="Implementation of the sliding lid experiment with the following "
                                       "set of boundaries:\n"
                                       "- left, right, bottom: rigid wall\n"
                                       "- top: moving wall\n\n"
                                       "The experiment is mainly driven by specifying exactly 2 of --reynolds_nr, "
                                       "--viscosity|--omega and --wall_velocity.\n"
                                       "If none of the is provided, a default set of parameters will be chosen.\n\n"
                                       "Simulation results are cached under <output_dir>/data.",
                                       conflict_handler="resolve")
    init_args(arg_parser, handler=main, size_x=300, size_y=300, n_steps=100000)
    arg_parser.add_argument("-w", "--omega", nargs="+", metavar="W",
                            type=float, default=None,
                            help="1/τ: relaxation time constant(s), in the range [0.1; 1.9]")
    arg_parser.add_argument("-v", "--viscosity", nargs="+", metavar="V",
                            type=float, default=None,
                            help="viscosity constant(s) of the fluid, in the range [0.01, 3.17]\n"
                            "if specified, this overrides -w|--omega")
    arg_parser.add_argument("-r", "--reynolds_nr", nargs="+", metavar="Re",
                            type=float, default=None,
                            help="Reynolds number")
    arg_parser.add_argument("-u", "--wall_velocity", nargs="+", metavar="U",
                            type=float, default=None,
                            help="velocities of the moving wall")
    arg_parser.add_argument("-a", "--animate", nargs="+", metavar="i",
                            type=float, default=[],
                            help="list of parameter indices to be animated: "
                            "i = (--reynolds_nr index + 1) * (--viscosity|--omega index + 1) - 1\n"
                            "example:\n"
                            "- with --reynolds_nr 250 500 750 1000 and --viscosity 0.03 0.09 0.15\n"
                            "- you wish to (Re=1000, ν=0.09)\n"
                            "- then pass --animate 7, since (3+1)*(1+1)-1 = 7\n"
                            "pass -1 in order to skip animation")
    return arg_parser


def main(args: argparse.Namespace):
    # default params
    if not args.reynolds_nr and not args.viscosity and not args.wall_velocity:
        args.viscosity = [0.03, 0.09, 0.15]
        args.reynolds_nr = [250, 500, 750, 1000]
        if not args.animate:
            args.animate = [7]

    # validate arguments
    validate_args(args)
    # for simplicity, convert omega to viscosity
    if args.omega:
        args.viscosity = omega_to_viscosity(np.array(args.omega))
        args.oemga = None
    # check reynolds_nr, viscosity, wall_velocity
    # TODO: what is L in case size_y != size_x?
    L = args.size_y
    if args.reynolds_nr and args.viscosity and args.wall_velocity:
        raise ValueError("--reynolds_nr, --viscosity|--omega and --wall_velocity can't be present at the same time")
    elif args.reynolds_nr and args.viscosity:
        pass
    elif args.reynolds_nr and args.wall_velocity:
        args.viscosity = L * np.array(args.wall_velocity) / np.array(args.reynolds_nr)
    elif args.viscosity and args.wall_velocity:
        args.reynolds_nr = L * np.array(args.wall_velocity) / np.array(args.viscosity)
    else:
        raise ValueError("exactly 2 of --reynolds_nr, --viscosity|--omega or --wall_velocity must be specified")

    # parameter sets
    # note: round to account for conversion errors from above
    params = [(Re, nu, Re*nu/L, f"{args.output_dir}/data/m6_x{args.size_x}_y{args.size_y}_n{args.n_steps}_v{nu}_Re{Re}.npy")
              for nu in np.array(args.viscosity).round(12)
              for Re in np.array(args.reynolds_nr).round(12)]

    # animation: plot only every nth step and increase step size over time
    anim_stages = visz.PlotStages([(0, 10), (1000, 50), (5000, 100), (20000, 500), (50000, 1000)])
    anim_steps = anim_stages.list(args.n_steps)

    # initialize lattices
    lattice = [
        sim.LatticeBoltzmann(args.size_x, args.size_y, viscosity=viscosity,
                             init_density=np.ones((args.size_y, args.size_x)),
                             boundaries=[bdry.MovingWallBoundaryCondition("t", [0, wall_velocity]),
                                         bdry.RigidWallBoundaryCondition("blr")])
        for _, viscosity, wall_velocity, _ in params
    ]

    # run simulation
    print("-- run simulation: sliding lid (serial)")
    for l in range(len(lattice)):
        # cached?
        cache_path = params[l][3]
        if os.path.isfile(cache_path):
            print(f"-- use cache: {cache_path}")
            continue
        # if not: run simulation
        velocity_field = np.empty((len(anim_steps), 2, args.size_y, args.size_x))
        j = 0
        for i in trange(args.n_steps, desc=f"lattice {l+1:2}/{len(lattice)}", **args.tqdm_kwargs):
            if i in anim_steps:
                velocity_field[j] = lattice[l].velocity
                j += 1
            lattice[l].step()
        velocity_field[-1] = lattice[l].velocity
        # save result
        print(f"-- save cache: {cache_path}")
        np.save(cache_path, velocity_field)

    # plot results
    filename = f"{args.output_dir}/m6_streamplots.png"
    print(f"-- save plot: {filename}")
    plot_streamplots(filename, args, lattice, params)

    for l in args.animate:
        if l < 0:
            continue
        Re, nu, _, _ = params[l]
        filename = f"{args.output_dir}/m6_x{args.size_x}_y{args.size_y}_v{nu}_Re{Re}.webm"
        print(f"-- save animation: {filename}")
        plot_animation_webm(filename, args, params[l], lattice[l], anim_stages, anim_steps)


def plot_streamplots(filename, args: argparse.Namespace, lattice: list[sim.LatticeBoltzmann], params: np.ndarray):
    n_cols = len(args.reynolds_nr)
    n_rows = len(lattice) // n_cols
    fig, ax = plt.subplots(n_rows, n_cols,
                           sharex=True, sharey=True,
                           figsize=(16, 14*(n_rows/n_cols)),
                           gridspec_kw={"width_ratios": [1, 1, 1, 1.11]})
    axf = ax.flatten()
    axf[0].set_xlim(0, args.size_x)
    axf[0].set_ylim(0, args.size_y)
    axf[0].invert_yaxis()
    x = np.arange(args.size_x)
    y = np.arange(args.size_y)
    for l in trange(len(lattice), **args.tqdm_kwargs):
        Re, viscosity, wall_velocity, cache_path = params[l]
        axf[l].set_title("{}, {}, {}, {}".format(rnd_tex('Re', Re, 1),
                                                 rnd_tex('\\nu', viscosity, 2),
                                                 rnd_tex('\\omega', lattice[l].omega, 2),
                                                 rnd_tex('U_w', wall_velocity, 3)))
        if l % n_cols == 0:
            axf[l].set_ylabel("$y$ dimension", fontsize="large")
        if l >= n_cols*(n_rows-1):
            axf[l].set_xlabel("$x$ dimension", fontsize="large")
        velocity_field = np.load(cache_path)
        u = velocity_field[-1, 1]
        v = velocity_field[-1, 0]
        norm = plt.Normalize(0, wall_velocity)
        axf[l].streamplot(x, y, u, v, color=np.sqrt(u**2 + v**2), norm=norm, density=1.2)
        if l % n_cols == 3:
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm, plt.cm.viridis), ax=axf[l], fraction=0.05)
            cbar.set_label(label="velocity magnitude $|u|$", fontsize="large")
            cbar.ax.set_yticks(ticks=[0, wall_velocity], labels=["0", "$U_w$"])
    fig.tight_layout()
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


def plot_animation_webm(filename: str,
                        args: argparse.Namespace,
                        params: tuple,
                        lattice: sim.LatticeBoltzmann,
                        anim_stages: visz.PlotStages,
                        anim_steps: list[int]):

    Re, viscosity, wall_velocity, cache_path = params
    velocity_field = np.load(cache_path)
    velocity_magnitude = np.sqrt(velocity_field[:, 1]**2 + velocity_field[:, 0]**2)

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    fig.suptitle(f"Sliding Lid Experiment", fontweight="bold")
    cmap = plt.cm.viridis
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, wall_velocity), cmap), ax=ax, label="velocity magnitude $|u|$")
    x = np.arange(args.size_x)
    y = np.arange(args.size_y)

    def update_streamplot(frame):
        i, t = frame
        ax.clear()
        ax.set_title("{}, {}, {}, {} (x{})".format(rnd_tex('Re', Re, 1),
                                                   rnd_tex('\\nu', viscosity, 2),
                                                   rnd_tex('\\omega', lattice.omega, 2),
                                                   rnd_tex('U_w', wall_velocity, 3),
                                                   anim_stages.get_step_size(t)),
                     loc="left")
        ax.set_xlabel(f"$x$ dimension")
        ax.set_ylabel(f"$y$ dimension")
        ax.set_xlim(0, args.size_x)
        ax.set_ylim(0, args.size_y)
        ax.invert_yaxis()
        fig.tight_layout()
        u, v = velocity_field[i, 1], velocity_field[i, 0]
        c = velocity_magnitude[i] if velocity_magnitude[i].sum() > 0 else "black"
        return ax.streamplot(x, y, u, v, density=2.0, color=c, cmap=cmap)

    animation = anim.FuncAnimation(fig, update_streamplot, frames=list(enumerate(anim_steps)))
    with tqdm(anim_steps, **args.tqdm_kwargs) as pbar:
        animation.save(filename, fps=25, dpi=180,
                       writer="ffmpeg", codec="libvpx-vp9", extra_args=["-an", "-crf", "42"],
                       progress_callback=lambda i, n: pbar.update())
    plt.close()
