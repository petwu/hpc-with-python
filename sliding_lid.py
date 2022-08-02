import argparse
import json
import os
import time

import numpy as np
import mpi4py.MPI as mpi

import src.boundary as bdry
import src.simulation as sim


# parse arguments
args = argparse.ArgumentParser(formatter_class=lambda prog:
                               argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=32),
                               description="Run the sliding lid experiment in serial or parallel. "
                               "In order to run in parallel, you need to run it using `mpiexec`.")
args.add_argument("-x", "--size_x", metavar="<X>", type=int, default=300,
                  help=f"size of the grid in x direction")
args.add_argument("-y", "--size_y", metavar="<Y>", type=int, default=300,
                  help=f"size of the grid in y direction")
args.add_argument("-n", "--n_steps", metavar="<N>", type=int, default=100000,
                  help=f"number of simulation steps")
args.add_argument("-w", "--omega", metavar="<W>", type=float, default=1.7,
                  help="1/τ: relaxation time constant, in the range [0.1; 1.9]")
args.add_argument("-u", "--wall_velocity", metavar="U", type=float, default=0.1,
                  help="velocity of the moving wall")
args.add_argument("-d", "--decomposition", metavar=("X", "Y"), type=int, nargs=2, default=[-1, -1],
                  help="domain decomposition")
args.add_argument("-e", "--extrapolate", default=False, action=argparse.BooleanOptionalAction,
                  help="extrapolate the density at the moving wall; otherwise it is approximated by the avg. density")
args.add_argument("-o", "--output", metavar="<file>", type=str,
                  help="file to store the parameters and time measurement in JSON format\n"
                  "if not set, it will be emitted to stdout")
args.add_argument("-p", "--progress", default=False, action=argparse.BooleanOptionalAction,
                  help="show a progress bar (only rank 0)")
args = args.parse_args()
if mpi.COMM_WORLD.Get_size() == 1:
    args.decomposition = None


# initialize lattice
lattice = sim.LatticeBoltzmann(args.size_x, args.size_y, decomposition=args.decomposition, omega=args.omega,
                               init_density=np.ones((args.size_y, args.size_x)),
                               boundaries=[bdry.RigidWall("lrb"),
                                           bdry.MovingWall("t", [0, args.wall_velocity], args.extrapolate)])


# get start time
if lattice.is_parallel:
    lattice.mpi.comm.Barrier()
start_time_ms = time.time_ns()


# run simulation
lattice.step(args.n_steps,
             progress=args.progress, bar_format="{l_bar}{bar:10}{r_bar}", leave=False)


# get end time
if lattice.is_parallel:
    lattice.mpi.comm.Barrier()
end_time_ms = time.time_ns()


# save/print results
if not lattice.is_parallel or lattice.mpi.rank == 0:
    elapsed_ns = (end_time_ms - start_time_ms)
    elapsed_s = elapsed_ns / 1e9
    mlups = args.size_x * args.size_y * args.n_steps / elapsed_s / 1e6
    info = {
        "elapsed_time_ns": elapsed_ns,
        "elapsed_time_s": elapsed_s,
        "mlups": mlups,
        "size_x": args.size_x,
        "size_y": args.size_y,
        "n_steps": args.n_steps,
        "omega": args.omega,
        "wall_velocity": args.wall_velocity,
        "decomposition": lattice.mpi.decomposition if lattice.is_parallel else None,
    }
    info_json = json.dumps(info, indent=2) + os.linesep
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as file:
            file.write(info_json)
    else:
        print(info_json)
