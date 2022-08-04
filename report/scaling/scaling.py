import argparse
import json
import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np


def args() -> argparse.Namespace:
    args = argparse.ArgumentParser(formatter_class=lambda prog:
                                   argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=32),
                                   description="Plot the #processes vs MLUPS figure.\n\n"
                                   "The data is expected to be a json file containing an object with a `mlups` property.\n"
                                   "The filenames need to be structed like this: <L>x<L>_<1..2^c>_<1..e>.json")
    args.add_argument("-i", "--input", metavar="<dir>", type=str, default=f"report/scaling/scaling.tar.gz",
                      help="path to the input directory or tarball containing the measurement data")
    args.add_argument("-l", "--lattice_sizes", metavar="<L>", type=int, nargs="+", default=[500, 300, 100],
                      help="list of lattice dimensions; used to determine the filename")
    args.add_argument("-c", "--n_cpu", metavar="<c>", type=int, default=11,
                      help="number of processor in range 2^0 to 2^c; used to determine the filename")
    args.add_argument("-e", "--n_experiments", metavar="<e>", type=int, default=3,
                      help="number of times the experiment was run with the same setting; used to determine the filename")
    args.add_argument("-o", "--output_dir", metavar="<dir>", type=str, default=f"out",
                      help="path to the output directory, where e.g. plots or data gets emitted")
    args.add_argument("-p", "--dpi", metavar="<dpi>", type=int, default=300,
                      help="DPI (dots per inch) value for saved plots; higher values increase the resolution")
    return args.parse_args()


def main(args: argparse.Namespace):
    filename = f"{args.output_dir}/m7_mlups.png"
    print(f"-- save plot: {filename}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of MPI processes")
    ax.set_ylabel("MLUPS")

    n_cpu = 2**np.arange(args.n_cpu+1)
    legend_handles = []
    legend_labels = []
    tar = tarfile.open(args.input) if os.path.isfile(args.input) else None
    tar_prefix = os.path.commonprefix(tar.getnames())
    if tar_prefix != "":
        tar_prefix += "/"
    for L in args.lattice_sizes:
        # read in data
        mlups = np.zeros((n_cpu.shape[0], args.n_experiments), dtype=float)
        for i, n in np.ndenumerate(n_cpu):
            for j in range(args.n_experiments):
                with (tar.extractfile(f"{tar_prefix}{L}x{L}_{n}_{j+1}.json") if tar else
                      open(f"{args.input}/{L}x{L}_{n}_{j+1}.json")) as json_file:
                    info = json.load(json_file)
                mlups[i, j] = info["mlups"]

        # take average
        mlups_avg = np.mean(mlups, axis=1)

        # plot curve with points
        l, = ax.plot(n_cpu, mlups_avg)
        s = ax.scatter(n_cpu, mlups_avg, marker="x", color=l.get_color())

        # plot min-max area
        f = ax.fill_between(n_cpu, mlups.min(axis=1), mlups.max(axis=1), color=l.get_color(), alpha=0.1)

        # combined legend
        legend_handles.append((l, s, f))
        legend_labels.append(f"{L}x{L}")

    ax.legend(legend_handles, legend_labels)
    plt.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main(args())
