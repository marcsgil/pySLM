import unittest
import numpy.testing as npt

from optics.instruments import objective

import numpy as np


class TestObjective(unittest.TestCase):
    def setUp(self):
        self.objectives = [
            objective.Lens(100e-3, 10e-3),
            objective.Olympus(20, 0.50),
            objective.Nikon(40, 0.80),
            objective.Nikon(50, 1.45, refractive_index=1.51),
            objective.Mitutoyo(20, 0.42),
            objective.EdmundOptics(100, 0.55),
        ]
        self.focal_lengths = [100e-3, 9e-3, 5e-3, 4e-3, 10e-3, 2e-3]
        self.pupil_diameters = [10e-3, 9e-3, 8e-3, 8e-3 * 1.45/1.51, 8.4e-3, 2.2e-3]
        self.refractive_indices = [1, 1, 1, 1.51, 1, 1]
        self.numerical_apertures = [0.05, 0.50, 0.80, 1.45, 0.42, 0.55]
        self.tube_lens_focal_lengths = [None, 180e-3, 200e-3, 200e-3, 200e-3, 200e-3]

    def test_constructor(self):
        for lens, f, d, na, tll, ri in zip(self.objectives, self.focal_lengths, self.pupil_diameters,
                                      self.numerical_apertures, self.tube_lens_focal_lengths, self.refractive_indices):
            npt.assert_almost_equal(lens.focal_length, f, err_msg=f'Focal length incorrect for {lens}.')
            npt.assert_almost_equal(lens.pupil_diameter, d, err_msg=f'Pupil diameter incorrect for {lens}.')
            npt.assert_almost_equal(lens.refractive_index, ri, err_msg=f'Refractive index incorrect for {lens}.')
            npt.assert_almost_equal(lens.numerical_aperture, na, err_msg=f'Numerical aperture incorrect for {lens}.')
            if isinstance(lens, objective.Objective):
                npt.assert_almost_equal(lens.tube_lens_focal_length, tll, err_msg=f'Tube lens focal length incorrect for {lens}.')


if __name__ == '__main__':
    unittest.main()
