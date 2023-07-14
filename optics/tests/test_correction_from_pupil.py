import unittest
import numpy.testing as npt

from optics.calc import correction_from_pupil

import numpy as np


class TestCorrectionFromPupil(unittest.TestCase):
    def test_scalar_phase(self):
        npt.assert_almost_equal(correction_from_pupil(-1, attenuation_limit=1.0), -1)
        npt.assert_almost_equal(correction_from_pupil(-1, attenuation_limit=10.0), -1)
        npt.assert_almost_equal(correction_from_pupil(1j, attenuation_limit=1.0), -1j)
        npt.assert_almost_equal(correction_from_pupil(1j, attenuation_limit=10.0), -1j)
        npt.assert_almost_equal(correction_from_pupil(np.sqrt(1j), attenuation_limit=1.0), np.sqrt(-1j))
        npt.assert_almost_equal(correction_from_pupil(np.sqrt(1j), attenuation_limit=10.0), np.sqrt(-1j))

    def test_scalar_too_large(self):
        npt.assert_almost_equal(correction_from_pupil(20.0, attenuation_limit=1.0), 1.0)
        npt.assert_almost_equal(correction_from_pupil(20.0, attenuation_limit=10.0), 1.0)
        npt.assert_almost_equal(correction_from_pupil(2.0, attenuation_limit=1.0), 1.0)
        npt.assert_almost_equal(correction_from_pupil(2.0, attenuation_limit=10.0), 1.0)

    def test_scalar_unity(self):
        npt.assert_almost_equal(correction_from_pupil(1.0, attenuation_limit=1.0), 1.0)
        npt.assert_almost_equal(correction_from_pupil(1.0, attenuation_limit=10.0), 1.0)

    def test_scalar_small(self):
        npt.assert_almost_equal(correction_from_pupil(0.5, attenuation_limit=1.0), 1.0)
        npt.assert_almost_equal(correction_from_pupil(0.5, attenuation_limit=10.0), 1.0)
        npt.assert_almost_equal(correction_from_pupil(0.01, attenuation_limit=1.0), 1.0)
        npt.assert_almost_equal(correction_from_pupil(0.01, attenuation_limit=10.0), 1.0)

    def test_scalar_zero(self):
        npt.assert_almost_equal(correction_from_pupil(0.0, attenuation_limit=1.0), 1.0)
        npt.assert_almost_equal(correction_from_pupil(0.0, attenuation_limit=10.0), 1.0)

    def test_pair_too_large(self):
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 20.0], attenuation_limit=1.0), np.array([1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 20.0], attenuation_limit=10.0), np.array([1.0, 0.1]))
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 2.0], attenuation_limit=1.0), np.array([1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 2.0], attenuation_limit=10.0), np.array([1.0, 0.5]))

    def test_pair_unity(self):
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 1.0], attenuation_limit=1.0), np.array([1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 1.0], attenuation_limit=10.0), np.array([1.0, 1.0]))

    def test_pair_small(self):
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 0.5], attenuation_limit=1.0), np.array([1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 0.5], attenuation_limit=10.0), np.array([0.5, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 0.01], attenuation_limit=1.0), np.array([1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 0.01], attenuation_limit=10.0), np.array([0.1, 1.0]))

    def test_pair_zero(self):
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 0.0], attenuation_limit=1.0), np.array([1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([1.0, 0.0], attenuation_limit=10.0), np.array([0.1, 1.0]))

    def test_triplet_large(self):
        npt.assert_array_almost_equal(correction_from_pupil([20, 5, 1], attenuation_limit=1.0), np.array([1.0, 1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([20, 5, 1], attenuation_limit=10.0), np.array([0.1, 0.4, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([10, 2, 0.5], attenuation_limit=1.0), np.array([1.0, 1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([10, 2, 0.5], attenuation_limit=10.0), np.array([0.1, 0.5, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([5, 1, 0.2], attenuation_limit=1.0), np.array([1.0, 1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([5, 1, 0.2], attenuation_limit=10.0), np.array([0.1, 0.5, 1.0]))

    def test_triplet_unity(self):
        npt.assert_array_almost_equal(correction_from_pupil([1, 0.2, 0.05], attenuation_limit=1.0), np.array([1.0, 1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([1, 0.2, 0.05], attenuation_limit=10.0), np.array([0.1, 0.5, 1.0]))

    def test_triplet_small(self):
        npt.assert_array_almost_equal(correction_from_pupil([0.5, 0.1, 0.02], attenuation_limit=1.0), np.array([1.0, 1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([0.5, 0.1, 0.02], attenuation_limit=10.0), np.array([0.1, 0.5, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([0.1, 0.02, 0.0], attenuation_limit=1.0), np.array([1.0, 1.0, 1.0]))
        npt.assert_array_almost_equal(correction_from_pupil([0.1, 0.02, 0.0], attenuation_limit=10.0), np.array([0.1, 0.5, 1.0]))

    def test_defocus_1d_small(self):
        rng = np.linspace(-1, 1, 4)

        sigma = 0.5
        lim = 10.0
        defocus = np.exp(2j * np.pi * 0 * rng**2) * np.exp(-0.5 * ((rng/sigma)**2))
        amplitude = np.exp(0.5 * ((rng/sigma)**2))
        amplitude = amplitude / np.min(amplitude.ravel()) / lim
        amplitude = np.minimum(amplitude, 1)
        amplitude /= np.max(np.abs(amplitude.ravel()))
        corr = np.exp(2j * np.pi * -0 * rng**2) * amplitude

        npt.assert_array_almost_equal(correction_from_pupil(defocus, attenuation_limit=lim), corr)

    def test_defocus_1d(self):
        rng = np.linspace(-1, 1, 50)

        sigma = 0.5
        lim = 10.0
        defocus = np.exp(2j * np.pi * 0 * rng**2) * np.exp(-0.5 * ((rng/sigma)**2))
        amplitude = np.exp(0.5 * ((rng/sigma)**2))
        amplitude = amplitude / np.min(amplitude.ravel()) / lim
        amplitude = np.minimum(amplitude, 1)
        amplitude /= np.max(np.abs(amplitude.ravel()))
        corr = np.exp(2j * np.pi * -0 * rng**2) * amplitude

        npt.assert_array_almost_equal(correction_from_pupil(defocus, attenuation_limit=lim), corr)

    def test_defocus_small(self):
        rng = np.linspace(-1, 1, 4)
        x = rng[:, np.newaxis]
        y = rng[np.newaxis, :]
        r = np.sqrt(x**2 + y**2)
        
        sigma = 0.1
        lim = 10.0
        defocus = np.exp(2j * np.pi * 3 * r**2) * np.exp(-0.5 * ((r/sigma)**2))
        amplitude = np.exp(0.5 * ((r/sigma)**2))
        amplitude = amplitude / np.min(amplitude.ravel()) / lim
        amplitude = np.minimum(amplitude, 1)
        amplitude /= np.max(np.abs(amplitude.ravel()))
        corr = np.exp(2j * np.pi * -3 * r**2) * amplitude

        npt.assert_array_almost_equal(correction_from_pupil(defocus, attenuation_limit=lim), corr)

    def test_defocus(self):
        rng = np.linspace(-1, 1, 50)
        x = rng[:, np.newaxis]
        y = rng[np.newaxis, :]
        r = np.sqrt(x**2 + y**2)

        sigma = 0.1
        lim = 10.0
        defocus = np.exp(2j * np.pi * 3 * r**2) * np.exp(-0.5 * ((r/sigma)**2))
        amplitude = np.exp(0.5 * ((r/sigma)**2))
        amplitude = amplitude / np.min(amplitude.ravel()) / lim
        amplitude = np.minimum(amplitude, 1)
        amplitude /= np.max(np.abs(amplitude.ravel()))
        corr = np.exp(2j * np.pi * -3 * r**2) * amplitude

        npt.assert_array_almost_equal(correction_from_pupil(defocus, attenuation_limit=lim), corr)


if __name__ == '__main__':
    unittest.main()
