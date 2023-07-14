import unittest
import numpy.testing as npt

from optics.calc.zernike import index2orders, orders2index, noll2orders, orders2noll, BasisPolynomial, Polynomial, fit

import numpy as np


class TestZernike(unittest.TestCase):
    def test_index2orders(self):
        npt.assert_equal(index2orders(0), (0, 0),
                         'Piston not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(1), (1, -1),
                         'Tilt not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(2), (1, 1),
                         'Tip not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(3), (2, -2),
                         'Oblique-astigmatism is not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(4), (2, 0),
                         'Defocus not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(5), (2, 2),
                         'Astigmatism-cartesian not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(6), (3, -3),
                         'Trefoil not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(7), (3, -1),
                         'Coma not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(8), (3, 1),
                         'Coma not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(9), (3, 3),
                         'Trefoil not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(index2orders(12), (4, 0),
                         'Spherical aberration not converted to the correct radial degree and azimuthal frequency.')

    def test_noll2orders(self):
        npt.assert_equal(noll2orders(1), (0, 0),
                         'Piston not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(2), (1, 1),
                         'Tip not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(3), (1, -1),
                         'Tilt not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(4), (2, 0),
                         'Defocus not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(5), (2, -2),
                         'Astigmatism-diag not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(6), (2, 2),
                         'Astigmatism-cartesian not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(7), (3, -1),
                         'Coma not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(8), (3, 1),
                         'Coma not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(9), (3, -3),
                         'Trefoil not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(10), (3, 3),
                         'Trefoil not converted to the correct radial degree and azimuthal frequency.')
        npt.assert_equal(noll2orders(11), (4, 0),
                         'Spherical aberration not converted to the correct radial degree and azimuthal frequency.')

    def test_orders2index(self):
        npt.assert_equal(orders2index(0, 0), 0,
                         'Piston not converted to the correct index.')
        npt.assert_equal(orders2index(1, -1), 1,
                         'Tilt not converted to the correct index.')
        npt.assert_equal(orders2index(1, 1), 2,
                         'Tip not converted to the correct index.')
        npt.assert_equal(orders2index(2, -2), 3,
                         'Astigmatism-diag not converted to the correct index.')
        npt.assert_equal(orders2index(2, 0), 4,
                         'Defocus not converted to the correct index.')
        npt.assert_equal(orders2index(2, 2), 5,
                         'Astigmatism-cartesian not converted to the correct index.')
        npt.assert_equal(orders2index(3, -3), 6,
                         'Trefoil- not converted to the correct index.')
        npt.assert_equal(orders2index(3, -1), 7,
                         'Coma not converted to the correct index.')
        npt.assert_equal(orders2index(3, 1), 8,
                         'Coma not converted to the correct index.')
        npt.assert_equal(orders2index(3, 3), 9,
                         'Trefoil+ not converted to the correct index.')
        npt.assert_equal(orders2index(4, 0), 12,
                         'Spherical aberration not converted to the correct index.')

    def test_orders2noll(self):
        npt.assert_equal(orders2noll(0, 0), 1,
                         'Piston not converted to the correct index.')
        npt.assert_equal(orders2noll(1, 1), 2,
                         'Tip not converted to the correct index.')
        npt.assert_equal(orders2noll(1, -1), 3,
                         'Tilt not converted to the correct index.')
        npt.assert_equal(orders2noll(2, 0), 4,
                         'Defocus not converted to the correct index.')
        npt.assert_equal(orders2noll(2, -2), 5,
                         'Astigmatism-diag not converted to the correct index.')
        npt.assert_equal(orders2noll(2, 2), 6,
                         'Astigmatism-cartesian not converted to the correct index.')
        npt.assert_equal(orders2noll(3, -1), 7,
                         'Coma not converted to the correct index.')
        npt.assert_equal(orders2noll(3, 1), 8,
                         'Coma not converted to the correct index.')
        npt.assert_equal(orders2noll(3, -3), 9,
                         'Trefoil- not converted to the correct index.')
        npt.assert_equal(orders2noll(3, 3), 10,
                         'Trefoil+ not converted to the correct index.')
        npt.assert_equal(orders2noll(4, 0), 11,
                         'Spherical aberration not converted to the correct index.')

    def test_orders2index2orders_tuple(self):
        npt.assert_equal(orders2index((0, 1, 2), (0, 1, 0)), np.array((0, 2, 4)),
                         'Tuple not converted to the correct index.')
        npt.assert_equal(index2orders((0, 2, 4)), np.array([[0, 1, 2], [0, 1, 0]]),
                         'Tuple not converted to the correct index.')

    def test_orders2noll2orders_tuple(self):
        npt.assert_equal(orders2noll((0, 1, 2), (0, 1, 0)), np.array((1, 2, 4)),
                         'Tuple not converted to the correct index.')
        npt.assert_equal(noll2orders((1, 2, 4)), np.array([[0, 1, 2], [0, 1, 0]]),
                         'Tuple not converted to the correct index.')

    def test_orders2index2orders_array(self):
        npt.assert_equal(orders2index(((0, 1), (2, 3)), ((0, 1), (0, -1))), np.array(((0, 2), (4, 7))),
                         'Tuple not converted to the correct index.')
        npt.assert_equal(index2orders([[0, 2], [4, 7]]), (np.array([[0, 1], [2, 3]]), np.array([[0, 1], [0, -1]])),
                         'Nested list of indices not converted to the correct tuple of orders.')

    def test_orders2noll2orders_array(self):
        npt.assert_equal(orders2noll(((0, 1), (2, 3)), ((0, 1), (0, -1))), np.array(((1, 2), (4, 7))),
                         'Tuple not converted to the correct index.')
        npt.assert_equal(noll2orders([[1, 2], [4, 7]]), (np.array([[0, 1], [2, 3]]), np.array([[0, 1], [0, -1]])),
                         'Tuple not converted to the correct index.')

    # def test_polynomial_r(self):
    #     values = __polynomial_r(n=0, m=0, rho=np.array(0))
    #     npt.assert_array_equal(values, np.array(1), '__polynomial_r(0) not correct')
    #     values = __polynomial_r(n=0, m=0, rho=np.array(1))
    #     npt.assert_array_equal(values, np.array(1), '__polynomial_r(1) not correct')
    #     values = __polynomial_r(n=0, m=0, rho=np.array(0.5))
    #     npt.assert_array_equal(values, np.array(1), '__polynomial_r(0.5) not correct')
    #     values = __polynomial_r(n=1, m=1, rho=np.array(0))
    #     npt.assert_array_equal(values, np.array(0), '__polynomial_r(1,1,0) not correct')
    #     values = __polynomial_r(n=1, m=1, rho=np.array(1))
    #     npt.assert_array_equal(values, np.array(1), '__polynomial_r(1,1,1) not correct')
    #     values = __polynomial_r(n=1, m=1, rho=np.array(0.5))
    #     npt.assert_array_equal(values, np.array(0.5), '__polynomial_r(1,1,0.5) not correct')
    #     values = __polynomial_r(n=2, m=0, rho=np.array(0))
    #     npt.assert_array_equal(values, np.array(-1), '__polynomial_r(2,0,0) not correct')
    #     values = __polynomial_r(n=2, m=0, rho=np.array(1))
    #     npt.assert_array_equal(values, np.array(1), '__polynomial_r(2,0,1) not correct')
    #     values = __polynomial_r(n=2, m=0, rho=np.array(1 / 2))
    #     npt.assert_array_equal(values, np.array(-0.5), '__polynomial_r(2,0,1/2) not correct')

    # def test_polynomial_r_odd(self):
    #     values = __polynomial_r(n=2, m=1, rho=np.array(0))
    #     npt.assert_array_equal(values, np.array(0.0), 'Odd difference __polynomial_r(2,1,0) should have been 0')
    #     values = __polynomial_r(n=2, m=1, rho=np.array(1 / 2))
    #     npt.assert_array_equal(values, np.array(0.0), 'Odd difference __polynomial_r(2,1,1/2) should have been 0')
    #     values = __polynomial_r(n=2, m=1, rho=np.array(1))
    #     npt.assert_array_equal(values, np.array(0.0), 'Odd difference __polynomial_r(2,1,1) should have been 0')
    #
    #     values = __polynomial_r(n=2, m=-1, rho=np.array(1))
    #     npt.assert_array_equal(values, np.array(0.0), 'Odd difference __polynomial_r(2,-1,1) should have been 0')
    #     values = __polynomial_r(n=0, m=-1, rho=np.array(1))
    #     npt.assert_array_equal(values, np.array(0.0), 'Odd difference __polynomial_r(2,0,1) should have been 0')

    # def test_polynomial_r_array(self):
    #     values = __polynomial_r(n=0, m=0, rho=[0, 0.5, 1])
    #     npt.assert_array_equal(values, np.array([1, 1, 1]), '__polynomial_r(0,0) with list failed')
    #     values = __polynomial_r(n=0, m=0, rho=np.array([0, 0.5, 1]))
    #     npt.assert_array_equal(values, np.array([1, 1, 1]), '__polynomial_r(0,0) with array failed')
    #     values = __polynomial_r(n=1, m=1, rho=np.array([0, 0.5, 1]))
    #     npt.assert_array_equal(values, np.array([0, 0.5, 1]), '__polynomial_r(1,1) with array failed')
    #     values = __polynomial_r(n=2, m=0, rho=np.array([0, 1 / 2, 1]))
    #     npt.assert_array_equal(values, np.array([-1, -0.5, 1]), '__polynomial_r(2,0) with array failed')
    #
    #     values = __polynomial_r(n=2, m=0, rho=np.array([[0], [1 / 2], [1]]))
    #     npt.assert_array_equal(values, np.array([[-1], [-0.5], [1]]), '__polynomial_r(2,0) with 2D array failed')
    #     values = __polynomial_r(n=2, m=0, rho=np.array([[0, 1 / 2, 1]]))
    #     npt.assert_array_equal(values, np.array([[-1, -0.5, 1]]), '__polynomial_r(2,0) with 2D array failed')

    # def test_polynomial_r_array_n(self):
    #     values = __polynomial_r(n=np.array([0, 1, 1, 2]), m=np.array([0, 1, -1, 0]), rho=0)
    #     npt.assert_array_equal(values, np.array([1, 0, 0, -1]),
    #                            '__polynomial_r([0, 1, 1, 2], [0, 1, -1, 0], 0) with array failed')
    #
    #     values = __polynomial_r(n=np.array([[0, 1, 1, 2]]), m=np.array([[0, 1, -1, 0]]), rho=0)
    #     npt.assert_array_equal(values, np.array([[1, 0, 0, -1]]),
    #                            '__polynomial_r([[0, 1, 1, 2]], [[0, 1, -1, 0]], 0) with array failed')
    #
    #     values = __polynomial_r(n=np.array([0, 1, 1, 2]), m=np.array([0, 1, -1, 0]),
    #                             rho=np.array([[0], [0.5], [1.0]]))
    #     npt.assert_array_equal(values, np.array([[1, 0, 0, -1], [1, 0.5, 0.5, -0.5], [1, 1, 1, 1]]),
    #                            '__polynomial_r with vectors for n, m, and rho failed')
    #
    #     values = __polynomial_r(n=[[0, 1], [1, 2]], m=[[0, 1], [-1, 0]],
    #                             rho=np.array([[[0]], [[0.5]], [[1.0]]]))
    #     npt.assert_array_equal(values, np.array([[[1, 0], [0, -1]], [[1, 0.5], [0.5, -0.5]], [[1, 1], [1, 1]]]),
    #                            '__polynomial_r with 2D arrays for n, m, and vector rho failed')
    #
    #     values = __polynomial_r(n=[[0, 1], [1, 2]], m=[[0, 1], [-1, 0]],
    #                             rho=np.array([[0, 0.5], [0.5, 1.0]])[..., np.newaxis, np.newaxis])
    #     npt.assert_array_equal(values, np.array([[[[1, 0], [0, -1]], [[1, 0.5], [0.5, -0.5]]],
    #                                              [[[1, 0.5], [0.5, -0.5]], [[1, 1], [1, 1]]]]),
    #                            '__polynomial_r with 2D arrays for n, m, and rho failed')
    #
    #     values = __polynomial_r(n=[0, 1, 1, 2], m=[0, 1, -1, 0], rho=[[[0]], [[1]]])
    #     npt.assert_array_almost_equal(values, np.array([[[1, 0, 0, -1]], [[1, 1, 1, 1]]]),
    #                                   12, "__polynomial_r with 2D arrays for n, m, and rho failed")

    def test_zernike_index(self):
        piston = BasisPolynomial(0)
        tilt = BasisPolynomial(1)
        tip = BasisPolynomial(2)
        defocus = BasisPolynomial(4)
        npt.assert_equal(piston(0), 1, "Piston fit failed")
        npt.assert_equal(piston(1/2), 1, "Piston fit failed")
        npt.assert_equal(piston(1), 1, "Piston fit failed")
        npt.assert_equal(tip(0.0), 0.0, "Tip fit failed")
        npt.assert_equal(tip(0.5), 1.0, "Tip fit failed")
        npt.assert_equal(tip(1.0), 2.0, "Tip fit failed")
        npt.assert_equal(tip(np.array([0.0, 0.5, 1.0])), np.array([0.0, 1.0, 2.0]), "Tip fit failed")
        npt.assert_almost_equal(tilt(np.array([0.0, 0.5, 1.0])), np.array([0.0, 0.0, 0.0]), 12, "Tilt fit failed")
        npt.assert_equal(tilt(np.array([0.0, 0.5, 1.0]), np.pi/2), np.array([0.0, 1.0, 2.0]), "Rotated tilt fit failed")
        npt.assert_almost_equal(tip(np.array([0.0, 0.5, 1.0]), np.pi/2), np.array([0.0, 0.0, 0.0]), 12, "Rotated tip fit failed")
        npt.assert_almost_equal(
            tilt(np.array([[0.0, 0.5, 1.0]]), [[0], [np.pi/2]]), np.array([[0, 0, 0], [0, 1, 2]]),
            12, "Non|rotated tilt fit failed")
        npt.assert_almost_equal(defocus([[0.0, 0.5, 1.0]], [[0.0], [np.pi/2]]),
                                np.sqrt(3) * np.array([[-1.0, -0.5, 1.0], [-1.0, -0.5, 1.0]]),
                                12, "Defocus fit failed")

    def test_zernike_index_array(self):
        ab4 = BasisPolynomial([0, 2, 1, 4])
        ab22 = BasisPolynomial([[0, 2], [1, 4]])

        npt.assert_equal(ab4(0), np.array([1, 0, 0, -np.sqrt(3)]), "Array of aberrations failed at rho=0")
        npt.assert_array_almost_equal(ab4(1), np.array([1, 2, 0, np.sqrt(3)]), 12, "Array of aberrations failed at rho=1")
        npt.assert_array_almost_equal(ab4([[0], [1]]), np.array([[1, 0, 0, -np.sqrt(3)], [1, 2, 0, np.sqrt(3)]]),
                                      12, "Array of aberrations failed at rho=[0, 1]")

        npt.assert_array_equal(ab22(0), np.array([[1, 0], [0, -np.sqrt(3)]]), "Array of aberrations failed at rho=0")
        npt.assert_array_almost_equal(ab22(1), np.array([[1, 2], [0, np.sqrt(3)]]), 12, "Array of aberrations failed at rho=1")
        npt.assert_array_almost_equal(ab22([[[0]], [[1]]]), np.array([[[1, 0], [0, -np.sqrt(3)]], [[1, 2], [0, np.sqrt(3)]]]),
                                      12, "Array of aberrations failed at rho=[0, 1]")

        npt.assert_array_equal(ab4(0, 0), np.array([1, 0, 0, -np.sqrt(3)]),
                               "Array of aberrations failed at (rho, theta)=(0, 0)")
        npt.assert_array_almost_equal(ab4(1, 0), np.array([1, 2, 0, np.sqrt(3)]), 12,
                                      "Array of aberrations failed at (rho, theta)=(1, 0)")
        npt.assert_array_almost_equal(ab4([[0], [1]], [[0], [0]]), np.array([[1, 0, 0, -np.sqrt(3)], [1, 2, 0, np.sqrt(3)]]),
                                      12, "Array of aberrations failed at (rho,theta)=([0, 1],[0, 0])")

        npt.assert_array_equal(ab4(0, np.pi/2), np.array([1, 0, 0, -np.sqrt(3)]),
                               "Array of aberrations failed at (rho, theta)=(0, pi/2)")
        npt.assert_array_almost_equal(ab4(1, np.pi/2), np.array([1, 0, 2, np.sqrt(3)]), 12,
                                      "Array of aberrations failed at (rho, theta)=(1, pi/2)")
        npt.assert_array_almost_equal(ab4([[0], [1]], [[np.pi/2], [np.pi/2]]),
                                      np.array([[1, 0, 0, -np.sqrt(3)], [1, 0, 2, np.sqrt(3)]]),
                                      12, "Array of aberrations failed at (rho,theta)=([0, 1],[pi/2, pi/2])")
        npt.assert_array_almost_equal(ab4([[[0]], [[1]]], [[[0], [np.pi/2]]]),
                                      np.array([[[1, 0, 0, -np.sqrt(3)], [1, 0, 0, -np.sqrt(3)]],
                                                [[1, 2, 0, np.sqrt(3)], [1, 0, 2, np.sqrt(3)]]]),
                                      12, "Array of aberrations failed for 2D (rho,theta)")

        npt.assert_array_equal(ab22(0), np.array([[1, 0], [0, -np.sqrt(3)]]), "Array of aberrations failed at rho=0")
        npt.assert_array_almost_equal(ab22(1), np.array([[1, 2], [0, np.sqrt(3)]]), 12, "Array of aberrations failed at rho=1")
        npt.assert_array_almost_equal(ab22([[[0]], [[1]]]), np.array([[[1, 0], [0, -np.sqrt(3)]], [[1, 2], [0, np.sqrt(3)]]]),
                                      12, "Array of aberrations failed at rho=[0, 1]")

        npt.assert_array_equal(ab22(0, 0), np.array([[1, 0], [0, -np.sqrt(3)]]),
                               "Array of aberrations failed at (rho, theta)=(0, 0)")
        npt.assert_array_almost_equal(ab22(1, 0), np.array([[1, 2], [0, np.sqrt(3)]]), 12,
                                      "Array of aberrations failed at (rho, theta)=(1, 0)")
        npt.assert_array_almost_equal(ab22([[[0]], [[1]]], 0), np.array([[[1, 0], [0, -np.sqrt(3)]], [[1, 2], [0, np.sqrt(3)]]]),
                                      12, "Array of aberrations failed at (rho, theta)=([0, 1], 0)")
        npt.assert_array_almost_equal(ab22(np.array([0, 1])[:, np.newaxis, np.newaxis, np.newaxis],
                                           np.array([0, np.pi/2])[np.newaxis, :, np.newaxis, np.newaxis]
                                           ),
                                      np.array([[[[1, 0], [0, -np.sqrt(3)]], [[1, 0], [0, -np.sqrt(3)]]],
                                                [[[1, 2], [0, np.sqrt(3)]], [[1, 0], [2, np.sqrt(3)]]]]),
                                      12, "Array of aberrations failed at (rho, theta)=([[0], [1]], [[0, pi/2]])")

        npt.assert_array_equal(ab22(0, np.pi/2), np.array([[1, 0], [0, -np.sqrt(3)]]),
                               "Array of aberrations failed at (rho, theta)=(0, pi/2)")
        npt.assert_array_almost_equal(ab22(1, np.pi/2), np.array([[1, 0], [2, np.sqrt(3)]]), 12,
                                      "Array of aberrations failed at (rho, theta)=(1, pi/2)")

    def test_error(self):
        with npt.assert_raises(ValueError):
            BasisPolynomial(2, 0)

    def test_zernike_superposition(self):
        s = Polynomial([1, 2, 3, 4])
        npt.assert_array_equal(s.coefficients, np.array([1, 2, 3, 4]))
        npt.assert_array_equal(s(0, 0), 1)

    def test_zernike_fit_cartesian(self):
        rng = np.linspace(-1, 1, 32)
        y, x = rng[:, np.newaxis], rng[np.newaxis, :]

        piston = BasisPolynomial(0)
        f = fit(z=piston.cartesian(y=y, x=x), y=y, x=x, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([1, 0, 0, 0, 0]), decimal=8)

        defocus = BasisPolynomial(4)
        f = fit(z=defocus.cartesian(y=y, x=x), y=y, x=x, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([0, 0, 0, 0, 1]), decimal=8)

        s = Polynomial([4, 3, 2, 0, 1])
        f = fit(z=s.cartesian(y=y, x=x), y=y, x=x, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([4, 3, 2, 0, 1]), decimal=8)

    def test_zernike_fit_polar(self):
        nb_subdivisions = 32
        rho = np.linspace(0, 1, nb_subdivisions)[:, np.newaxis]
        phi = np.linspace(-np.pi, np.pi, nb_subdivisions + 1, endpoint=False)[np.newaxis, :]

        piston = BasisPolynomial(0)
        f = fit(z=piston(rho=rho, phi=phi), rho=rho, phi=phi, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([1, 0, 0, 0, 0]), decimal=8)

        defocus = BasisPolynomial(4)
        f = fit(z=defocus(rho=rho, phi=phi), rho=rho, phi=phi, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([0, 0, 0, 0, 1]), decimal=8)

        s = Polynomial([4, 3, 2, 0, 1])
        f = fit(z=s(rho=rho, phi=phi), rho=rho, phi=phi, order=5)
        npt.assert_array_almost_equal(f.coefficients, np.array([4, 3, 2, 0, 1]), decimal=8)


if __name__ == '__main__':
    unittest.main()
