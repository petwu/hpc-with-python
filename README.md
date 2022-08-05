<!-- omit in toc -->
# HPC with Python

This repository is for the course _High-Performance Computing: Fluid Mechanics
with Python_ at the University of Freiburg in the summer semester 2022.

**Contents**

- [Animation: Lid-Driven Cavity](#animation-lid-driven-cavity)
- [Setup](#setup)
- [Scaling Tests with Lid-Driven Cavity](#scaling-tests-with-lid-driven-cavity)
- [Milestones](#milestones)
- [Report](#report)
- [Reproducibility](#reproducibility)
- [Tests](#tests)

## Animation: Lid-Driven Cavity

[m6_x300_y300_v0.03_Re1000.webm](https://user-images.githubusercontent.com/39537032/182847788-903f7f5b-e381-4c12-977a-41c67552431a.mp4)

<details>
The versioned video is located under [report/media/m6_x300_y300_v0.03_Re1000.webm](report/media/m6_x300_y300_v0.03_Re1000.webm).
</details>

## Setup

Install dependencies from PyPi using pip:
```sh
pip3 install -r requirements.txt
```

In order to run the parallelized code, you need some MPI implementation like [OpenMPI](https://www.open-mpi.org).

## Scaling Tests with Lid-Driven Cavity

A parametrizable implementation of the lid-driven cavity experiment can be run with the
[lid_driven_cavity.py](lid_driven_cavity.py) script. Run
```sh
python3 lid_driven_cavity.py --help
```
for a list of possible arguments.

Use `mpiexec` in order to run it in parallel, e.g.:
```sh
mpiexec -n 16 python3 lid_driven_cavity.py -x 300 -y 300 -w 1.7 -n 100000
```

## Milestones

Implementation of the milestones is located in [src/milestones](src/milestones).
Use [milestone.py](milestone.py) in order to run a specific milestone. See
```sh
python3 milestone.py --help
```
for usage information.

## Report

The final report is located at [report/main.pdf](main.report/main.pdf).

Building the report:
```sh
$ cd report
$ make report
```

## Reproducibility

All plots with results from experiments from the report can be reproduced:
```
$ cd report
$ make plots
```

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
