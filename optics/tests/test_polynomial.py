import unittest
import numpy.testing as npt
import numpy as np

from tests import log

from optics.calc.polynomial import Polynomial, find_roots


class TestPolynomial(unittest.TestCase):
    def setUp(self):
        self.coefficients = [
            [0, 1],
            [-1, 1],
            [-4, 1],
            [2, 2],
            [1, 1],
            [2, 1],
            [[6, 3], [2, 1], [1, 1]],
            [0, 0, 1],
            [-1, 0, 1],
            [-4, 0, 1],
            [1, 0, 1],
            [-1, 0, -1],
            [2, 0, 2],
            [[-12, 0, 3], [-1, 0, 1]],
            [1, -2, 1],
            [2, -3, 1],
            [2, -4, 2],
            [0, 0, 0, 1],
            [-1, 0, 0, 1],
            [1, 0, 0, 1],
            [2, 0, 0, 2],
            [-1, 0, 0, -1],
            [-8, 0, 0, 1],
            [-1, 3, -3, 1],
            [0, -1, 0, 1],
            [0, 1, -2, 1],
            [0, 2, -3, 1],
            [-2, 5, -4, 1],
            [0, 0, 0, 0, 1],
            [-1, 0, 0, 0, 1],
            [0, 0, 1, -2, 1],
        ]
        self.scale = [np.asarray(_)[..., -1, np.newaxis] for _ in self.coefficients]
        self.roots = [
            [0],
            [1],
            [4],
            [-1],
            [-1],
            [-2],
            [[-2], [-2], [-1]],
            [0, 0],
            [-1, 1],
            [-2, 2],
            [1j, -1j],
            [1j, -1j],
            [1j, -1j],
            [[2, -2], [-1, 1]],
            [1, 1],
            [1, 2],
            [1, 1],
            [0, 0, 0],
            np.exp(2j * np.pi / 3 * np.arange(3)),
            -np.exp(2j * np.pi / 3 * np.arange(3)),
            -np.exp(2j * np.pi / 3 * np.arange(3)),
            -np.exp(2j * np.pi / 3 * np.arange(3)),
            2 * np.exp(2j * np.pi / 3 * np.arange(3)),
            [1, 1, 1],
            [0, 1, -1],
            [0, 1, 1],
            [0, 1, 2],
            [1, 1, 2],
            [0, 0, 0, 0],
            np.exp(2j * np.pi / 4 * np.arange(4)),
            [0, 0, 1, 1],
        ]

    @staticmethod
    def compare_set(set_a, set_b, err_msg=''):
        """
        Compare if arrays are almost equal up to a permutation per column
        :param set_a: A sequence of numbers of which the order does not matter.
        :param set_b: The sequence of numbers that is to be compared.
        :param err_msg: The optional error message.
        """
        set_a = np.moveaxis(set_a, -1, 0)
        set_b = np.moveaxis(set_b, -1, 0)
        distance2 = []
        for _, a in enumerate(set_a):
            distance2.append(np.abs(a - set_b) ** 2)
        view0 = np.amax(np.amin(distance2, axis=0), axis=0)
        view1 = np.amax(np.amin(distance2, axis=1), axis=0)
        npt.assert_almost_equal(view0, 0, err_msg=err_msg)
        npt.assert_almost_equal(view1, 0, err_msg=err_msg)

    def test_simple(self):
        for coefficients, scale, roots in zip(self.coefficients, self.scale, self.roots):
            poly = Polynomial(coefficients, axis=-1)
            log.info(f'Testing {poly} with roots {roots} and scale {scale}...')
            npt.assert_equal(poly.order, np.asarray(roots).shape[-1], err_msg='order not as expected')
            npt.assert_equal(poly.shape, np.asarray(scale).shape[:-1], err_msg='shape not as expected')
            npt.assert_equal(poly.ndim, np.asarray(scale).ndim - 1, err_msg='dimensions not as expected')
            npt.assert_equal(poly.scale, scale, err_msg='scale not as expected')
            self.compare_set(poly.roots, roots, err_msg='roots not as expected')
            npt.assert_almost_equal(poly.coefficients, coefficients, err_msg='coefficients not recovered from roots')

    def test_random_coefficients(self):
        rng = np.random.RandomState(seed=1)

        nb_tests = 1000
        for degree in range(1, 5):
            log.info(f'Testing {nb_tests} polynomials of degree {degree}.')
            coefficients = rng.randn(nb_tests, degree + 1) + 1j * rng.randn(nb_tests, degree + 1)
            poly = Polynomial(coefficients, axis=-1)
            npt.assert_almost_equal(poly.coefficients, coefficients, err_msg='coefficients not recovered from roots')

    def test_find_roots(self):
        for coefficients, scale, roots in zip(self.coefficients, self.scale, self.roots):
            self.compare_set(find_roots(coefficients, axis=-1), roots, err_msg='find_roots did not reply as expected')


if __name__ == '__main__':
    unittest.main()

