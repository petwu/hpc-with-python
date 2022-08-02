<!-- omit in toc -->
# HPC with Python

This repository is for the course _High-Performance Computing: Fluid Mechanics
with Python_ at the University of Freiburg in the summer semester 2022.

**Contents**

- [Setup](#setup)
- [Scaling Tests with Lid-Driven Cavity](#scaling-tests-with-lid-driven-cavity)
- [Milestones](#milestones)
- [Tests](#tests)

## Setup

Install dependencies from PyPi using pip:
```sh
pip3 install -r requirements.txt
```

In order to run the parallelized code, you need some MPI implementation like [OpenMPI](https://www.open-mpi.org).

## Scaling Tests with Lid-Driven Cavity

A parameterizable implementation of the lid-driven cavity or sliding-lid experiment can be run with the
[sliding_lid.py](sliding_lid.py) script. Run
```sh
python3 sliding_lid.py --help
```
for a list of possible arguments.

Use `mpiexec` in order to run it in parallel, e.g.:
```sh
mpiexec -n 16 python3 sliding_lid.py -x 300 -y 300 -w 1.7 -n 100000
```

## Milestones

Implementation of the milestones is located in [src/milestones](src/milestones).
Use [milestone.py](milestone.py) in order to run a specific milestone. See
```sh
python3 milestone.py --help
```
for usage information.

## Tests

Run all unit tests in the `test/` directory using the following command:
```sh
python3 -m unittest
```
The following naming conventions need to be applied in order for the
[test discovery](https://docs.python.org/3/library/unittest.html#unittest-test-discovery)
to work:
- the directory must be named `test*`
- every (sub)directory need to be a package, i.e. they need a `__init__.py` file
- all filenames need to match `test*.py` and must be valid identifiers, e.g.
  `test_foo.py` but not `test-foo.py`
- all classes must extend `unittest.TestCase`
- all test case method names must match `test*`
