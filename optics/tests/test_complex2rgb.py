import unittest
import numpy.testing as npt

from optics.utils.display import complex2rgb


class TestComplex2RGB(unittest.TestCase):
    def test_scalar_0d(self):
        npt.assert_array_equal(complex2rgb(0), [0, 0, 0])
        npt.assert_array_equal(complex2rgb(0.0), [0, 0, 0])
        npt.assert_array_equal(complex2rgb(0.0 + 0.0j), [0, 0, 0])
        npt.assert_array_equal(complex2rgb(1.0), [0, 1, 1])
        npt.assert_array_equal(complex2rgb(1.0j), [0.5, 0, 1])
        npt.assert_array_equal(complex2rgb(-1.0), [1, 0, 0])
        npt.assert_array_equal(complex2rgb(-1.0j), [0.5, 1, 0])

    def test_scalar_1d(self):
        npt.assert_array_equal(complex2rgb([0]), [[0, 0, 0]])
        npt.assert_array_equal(complex2rgb([0.0]), [[0, 0, 0]])
        npt.assert_array_equal(complex2rgb([0.0 + 0.0j]), [[0, 0, 0]])
        npt.assert_array_equal(complex2rgb([1.0]), [[0, 1, 1]])
        npt.assert_array_equal(complex2rgb([1.0j]), [[0.5, 0, 1]])
        npt.assert_array_equal(complex2rgb([-1.0]), [[1, 0, 0]])
        npt.assert_array_equal(complex2rgb([-1.0j]), [[0.5, 1, 0]])

    def test_scalar_2d(self):
        npt.assert_array_equal(complex2rgb([[0]]), [[[0, 0, 0]]])
        npt.assert_array_equal(complex2rgb([[0.0]]), [[[0, 0, 0]]])
        npt.assert_array_equal(complex2rgb([[0.0 + 0.0j]]), [[[0, 0, 0]]])
        npt.assert_array_equal(complex2rgb([[1.0]]), [[[0, 1, 1]]])
        npt.assert_array_equal(complex2rgb([[1.0j]]), [[[0.5, 0, 1]]])
        npt.assert_array_equal(complex2rgb([[-1.0]]), [[[1, 0, 0]]])
        npt.assert_array_equal(complex2rgb([[-1.0j]]), [[[0.5, 1, 0]]])

    def test_scalar_3d(self):
        npt.assert_array_equal(complex2rgb([[[0]]]), [[[[0, 0, 0]]]])
        npt.assert_array_equal(complex2rgb([[[0.0]]]), [[[[0, 0, 0]]]])
        npt.assert_array_equal(complex2rgb([[[0.0 + 0.0j]]]), [[[[0, 0, 0]]]])
        npt.assert_array_equal(complex2rgb([[[1.0]]]), [[[[0, 1, 1]]]])
        npt.assert_array_equal(complex2rgb([[[1.0j]]]), [[[[0.5, 0, 1]]]])
        npt.assert_array_equal(complex2rgb([[[-1.0]]]), [[[[1, 0, 0]]]])
        npt.assert_array_equal(complex2rgb([[[-1.0j]]]), [[[[0.5, 1, 0]]]])

    def test_vector(self):
        npt.assert_array_equal(complex2rgb([[0, 1]]), [[[0, 0, 0], [0, 1, 1]]])
        npt.assert_array_equal(complex2rgb([[0], [1]]), [[[0, 0, 0]], [[0, 1, 1]]])

    def test_matrix(self):
        npt.assert_array_equal(complex2rgb([[0, 1], [1j, -1j]]),
                               [[[0, 0, 0], [0, 1, 1]], [[0.5, 0, 1], [0.5, 1, 0]]])

if __name__ == '__main__':
    unittest.main()
