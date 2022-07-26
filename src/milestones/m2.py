import argparse

import numpy as np
import src.simulation as sim
from tqdm import tqdm

from ._common import init_args


def args(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    arg_parser = subparsers.add_parser("m2", help="milestone 2: collision",
                                       description="Implementation of the collision operator.",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    init_args(arg_parser, handler=main, size_x=40, size_y=25, n_steps=100, omega=1.0)
    return arg_parser


def main(args: argparse.Namespace):
    for test, desc in [(test1, "uniform density with higher value at center"),
                       (test2, "random density")]:

        # create lattice
        d, v = test(args.size_x, args.size_y)
        lattice = sim.LatticeBoltzmann(args.size_x, args.size_y, omega=args.omega, viscosity=args.viscosity,
                                       init_density=d, init_velocity=v,
                                       animate=True, plot_stages=[(0, 1), (100, 2)])

        # run simulation
        print(f"-- run simulation: {desc}")
        lattice.step(args.n_steps, progress=True, **args.tqdm_kwargs)

        # save animation
        webm = f"{args.output_dir}/m2_{desc.split(' ')[0]}.webm"
        print(f"-- save animation: {webm}")
        with tqdm(total=args.n_steps, **args.tqdm_kwargs) as pbar:
            lattice.plot.get_animation().save(webm, fps=25, dpi=180, writer="ffmpeg", codec="libvpx-vp9",
                                              progress_callback=lambda i, n: pbar.update())


def test1(x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    # uniform density with a slightly higher value at the center
    c_w = max(min(x, y)//3, 1)
    c_x = x//2 - c_w//2
    c_y = y//2 - c_w//2
    assert c_x > 0 and c_y > 0
    density = np.ones((y, x)) * 0.1
    density[c_y:c_y+c_w, c_x:c_x+c_w] = 0.5
    # zero velocity
    velocity = np.zeros((2, y, x))
    return density, velocity


def test2(x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    # random density
    density = np.random.rand(y, x)
    # random velocity
    velocity = np.random.rand(2, y, x) * 0.2 - 0.1
    return density, velocity
