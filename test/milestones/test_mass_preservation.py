import numpy as np
import unittest
from src.simulation import LatticeBoltzmann


class MassPreservationTest(unittest.TestCase):
    """
    Test that the simulation process is mass preserving.
    Everything else would be unrealistic.
    """
    def test(self):
        size_x = size_y = 100

        # initialize probability distribution function
        pdf = np.ones((9, size_y, size_x))
        channels = [1, 4, 8]
        s_x = size_x//5
        s_y = size_y//5
        d = 5
        for c in channels:
            pdf[c, :s_x, :s_y] = d

        # create lattice
        lattice = LatticeBoltzmann(size_x, size_y, init_pdf=pdf)

        # check initial mass
        expected_mass = size_x*size_y*9 + s_x*s_y*len(channels)*(d-1)
        self.assertEqual(lattice.mass, expected_mass)

        # run simulation a few time and check mass
        for _ in range(1000):
            lattice.step()
            self.assertAlmostEqual(lattice.mass, expected_mass, places=6)
