<!-- omit in toc -->
# HPC with Python

This repository is for the course _High-Performance Computing: Fluid Mechanics
with Python_ at the University of Freiburg in the summer semester 2022.

**Contents**

- [Setup](#setup)
- [Milestones](#milestones)
- [Tests](#tests)

## Setup

Install dependencies from PyPi using pip:
```sh
pip3 install -r requirements.txt
```

## Milestones

Implementation of the milestones is located in [src/milestones](src/milestones).
They are implemented as [Jupyter](https://jupyter.org) notebooks and exported as
HTML.

## Tests

Run all unit tests in the `test/` directory using the following command:
```sh
python -m unittest
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
