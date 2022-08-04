#!/bin/sh

# load required modules
module load devel/python/3.10.0_gnu_11.1
module load compiler/gnu/12.1
module load mpi/openmpi/4.1

# install python dependencies
pip install --user --upgrade numpy scipy mpi4py tqdm matplotlib

# run experiment
exec mpiexec --bind-to core --map-by core python3 lid_driven_cavity.py $@
