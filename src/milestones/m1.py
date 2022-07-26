import argparse

import numpy as np
import src.simulation as sim
from tqdm import tqdm, trange

from .common import init_args


def args(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    arg_parser = subparsers.add_parser("m1", help="milestone 1: streaming",
                                       description="Introduction to LBE and implementation of the streaming operator.")
    init_args(arg_parser, handler=main, size_x=20, size_y=12, n_steps=100)
    return arg_parser


def main(args: argparse.Namespace):
    # create lattice with initial PDF
    pdf = np.zeros((9, args.size_y, args.size_x))
    for c in [1, 4, 8]:
        pdf[c, :3, :3] = 1
    lattice = sim.LatticeBoltzmann(args.size_x, args.size_y, init_pdf=pdf,
                                   animate=True, plot_stages=[(0, 1)])

    # save initial state image
    png = f"{args.output_dir}/m1_streaming.png"
    print(f"-- save initial state: {png}")
    lattice.plot.save(png, dpi=args.dpi)
    lattice.plot.close()

    # run simulation
    print("-- run simulation")
    for _ in trange(args.n_steps, **args.tqdm_kwargs):
        lattice.update_density()
        lattice.update_velocity()
        lattice.streaming_step()
        lattice.update_plot()

    # save animation
    webm = f"{args.output_dir}/m1_streaming.webm"
    print(f"-- save animation: {webm}")
    with tqdm(total=args.n_steps, **args.tqdm_kwargs) as pbar:
        lattice.plot.get_animation().save(webm, fps=25, dpi=180, writer="ffmpeg", codec="libvpx-vp9",
                                        progress_callback=lambda i, n: pbar.update())
