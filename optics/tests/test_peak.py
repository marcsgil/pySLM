import unittest
import numpy as np
import numpy.testing as npt

from optics.utils.peak import Peak, fwhm, fwtm


class TestPeak(unittest.TestCase):
    def test_peak(self):
        a = np.zeros(11)
        a[1:4] = 100
        npt.assert_equal(fwhm(a), 3.0)
        npt.assert_almost_equal(Peak(a).width(), 3.0)
        npt.assert_almost_equal(Peak(a).position, 2.0)
        a[5:] = 90
        npt.assert_equal(fwhm(a), 3.0)

        a = np.zeros(11)
        a[0:3] = 1.0
        npt.assert_equal(fwhm(a), np.inf)

        a = np.zeros(11)
        a[-4:-1] = 100.0
        npt.assert_equal(fwhm(a), 3.0)
        npt.assert_almost_equal(fwtm(a), 3.8)
        npt.assert_almost_equal(Peak(a).width(), 3.0)
        npt.assert_almost_equal(Peak(a).position, 8.0)
        a[1] = 101.0
        npt.assert_equal(fwhm(a), 1.0)
        npt.assert_almost_equal(fwtm(a), 1.8)

        a = np.zeros(11)
        a[1] = 80.0
        a[2] = 90.0
        a[3] = 100.0
        a[4] = 80.0
        a[5:8] = 20.0
        npt.assert_equal(fwhm(a), 3 + (1 - 0.5/0.8) + 0.5)
        npt.assert_almost_equal(Peak(a).width(), 3 + (1 - 0.5/0.8) + 0.5)
        npt.assert_almost_equal(Peak(a).position, 3.0)
        a[4] = 100.0
        npt.assert_almost_equal(Peak(a).position, 3.5)
        a[5] = 100.0
        npt.assert_almost_equal(Peak(a).position, 4.0)


if __name__ == '__main__':
    unittest.main()
