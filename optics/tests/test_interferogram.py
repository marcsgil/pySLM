import unittest
import numpy.testing as npt

from optics.calc.interferogram import Interferogram
from optics.utils.array import add_dims_on_right
from optics.utils import ft
from optics.utils.ft import Grid

import numpy as np


class TestInterferogram(unittest.TestCase):
    def setUp(self):
        grid = Grid([512, 512], extent=4)
        rho = np.sqrt(sum(_**2 for _ in grid))
        phi = np.arctan2(grid[1], grid[0])

        self.complex_images = [np.ones(grid.shape, dtype=np.complex64),
                               (1.0+0.0j) * (rho == 0),
                               (rho < 1) * np.exp(1j * phi)]
        self.raw_interference_images = [np.abs(np.exp(2j * np.pi * sum(_ * g for _, g in zip([0.25, 0.25], grid))) + c_field) ** 2
                                        for c_field in self.complex_images]

    def test_constructor(self):
        for raw, c_field in zip(self.raw_interference_images, self.complex_images):
            i = Interferogram(raw)


if __name__ == '__main__':
    unittest.main()

