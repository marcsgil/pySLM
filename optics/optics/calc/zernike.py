"""
Zernike polynomial definition and fitting.

:py:class:``BasisPolynomial``s can be specified using standard indices or radial and azimuthal orders. Conversion functions exist for
Noll order. These objects can be used as polar-coordinate functions or as Cartesian functions using the
:py:obj:``BasisPolynomial.cartesian`` property. General :py:class:``Polynomial``s are superpositions of ``BasisPolynomial``s.
"""
from __future__ import annotations
import numpy as np
from typing import Union, Sequence, Callable

from optics.utils.polar import cart2pol
from optics.utils.factorial_fraction import factorial_product_fraction
from optics.utils.cache import disk
from optics.utils.display.subsup import subscript, superscript


__all__ = ['index2orders', 'orders2index', 'noll2orders', 'orders2noll', 'index2noll', 'noll2index',
           'BasisPolynomial', 'Polynomial', 'Fit', 'fit',
           'piston',
           'tip', 'tilt',
           'oblique_astigmatism', 'defocus', 'vertical_astigmatism',
           'vertical_trefoil', 'vertical_coma', 'horizontal_coma', 'oblique_trefoil',
           'primary_spherical'
           ]


def index2orders(j: Union[int, Sequence, np.ndarray]):
    """
    Converts a Zernike index or indices, js > 0, to a tuple (radial degree m, azimuthal frequency n), for which 0 <= m <= n.
    When multiple values are specified, m and n will have the same shape as the input js.

    The standard OSA/ANSI ordering starts at 0. https://en.wikipedia.org/wiki/Zernike_polynomials

    See also the inverse operation: js = orders2index(n, m)

    :param j: The standard Zernike index, or an ndarray thereof.
    :return: a tuple (m, n) of order subscripts or ndarrays thereof.
    """
    j = np.array(j, int)

    n = np.array(np.ceil((np.sqrt(9 + 8 * j) - 1) / 2) - 1, dtype=int)
    m = 2 * j - n * (n + 2)

    n[j < 0] = -1  # Mark all indexes less than 1 as invalid

    return n, m


def noll2orders(j: Union[int, Sequence, np.ndarray]):
    """
    Converts a Noll index or indices, js > 0, to a tuple (radial degree m, azimuthal frequency n), for which 0 <= m <= n.
    When multiple values are specified, m and n will have the same shape as the input js.

    Note that the Noll ordering starts counting at 1, not 0! The ordering is described here:
    Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence" (PDF). J. Opt. Soc. Am. 66 (3): 207. Bibcode:1976JOSA...66..207N. doi:10.1364/JOSA.66.000207.

    See also the inverse operation: js = orders2noll(n, m)

    :param j: The standard Zernike index, or an ndarray thereof.
    :return: a tuple (m, n) of order subscripts or ndarrays thereof.
    """
    j = np.array(j, int)

    n = np.array(np.ceil((np.sqrt(1 + 8 * j) - 1) / 2) - 1, dtype=int)
    m_seq = j - n * (n + 1) / 2 - 1  # the zero-based sequence number for the real m = 0, 2, -2, 4, -4, 6, -6,... or 1, -1, 3, -3, 6, -6, ... (or the inverse depending on mod(j,2) )
    m = 2 * np.array((m_seq + (1 - np.mod(n, 2))) / 2, dtype=int) + np.mod(n, 2)  # absolute value of real m
    m *= (1 - 2 * np.mod(j, 2))  # If j odd, make m negative.

    n[j < 1] = -1  # Mark all indexes less than 1 as invalid

    return n, m


def orders2index(n: Union[int, Sequence, np.ndarray], m: Union[int, Sequence, np.ndarray]=0):
    """
    Converts a Zernike coordinate (radial degree n, azimuthal frequency m), j_index, to standard OSA/ANSI indexes
    When multiple values are specified, j_index will have the same shape as the inputs n and m.

    See also the inverse operation: n, m = index2orders(j)

    :param n: The radial degree or ndarrays thereof.
    :param m: The azimuthal frequency or ndarrays thereof.
    :return: The standard Zernike index, or an ndarray thereof.
    """
    n = np.array(n, dtype=int)
    m = np.array(m, dtype=int)

    j_index = np.array((n * (n + 2) + m) / 2, dtype=int)

    j_index[np.logical_or(np.logical_or(n < 0, np.abs(m) > n), np.mod(m + n, 2) != 0)] = -1  # Mark invalid indices

    return j_index


def orders2noll(n: Union[int, Sequence, np.ndarray], m: Union[int, Sequence, np.ndarray]=0):
    """
    Converts a Zernike coordinate (radial degree n, azimuthal frequency m), j_index, to Noll indexes
    When multiple values are specified, j_index will have the same shape as the inputs n and m.

    The ordering is described here:
    Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence" (PDF). J. Opt. Soc. Am. 66 (3): 207. Bibcode:1976JOSA...66..207N. doi:10.1364/JOSA.66.000207.
    See also the inverse operation: n, m = noll2orders(j_index)

    :param n: The radial degree or ndarrays thereof.
    :param m: The azimuthal frequency or ndarrays thereof.
    :return: The Noll Zernike index, or an ndarray thereof.
    """
    n = np.array(n, dtype=int)
    m = np.array(m, dtype=int)

    j_index = np.array(n * (n + 1) / 2, dtype=int)  # number up to n-1
    j_index += np.abs(m) + (m == 0)  # correct number or one too low
    j_index += np.logical_and((m != 0), np.logical_xor(m < 0, np.mod(j_index, 2)))  # make j_index odd if m negative

    j_index[np.logical_or(np.logical_or(n < 0, np.abs(m) > n), np.mod(m + n, 2) != 0)] = -1  # Mark invalid indices

    return j_index


def index2noll(j: Union[int, Sequence, np.ndarray]):
    """
    Converts a Zernike index or indices, js > 0, to Noll indexes.
    When multiple values are specified, m and n will have the same shape as the input js.

    The standard OSA/ANSI ordering starts at 0. https://en.wikipedia.org/wiki/Zernike_polynomials

    See also the inverse operation: js = orders2index(n, m)

    :param j: The standard Zernike index, or an ndarray thereof.
    :return: The Noll Zernike index, or an ndarray thereof.
    """
    return orders2noll(*index2orders(j))


def noll2index(j: Union[int, Sequence, np.ndarray]):
    """
    Converts a Noll index or indices, js > 0, to a Zernike index or indices, for which 0 <= m <= n.
    When multiple values are specified, m and n will have the same shape as the input js.

    Note that the Noll ordering starts counting at 1, not 0! The ordering is described here:
    Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence" (PDF). J. Opt. Soc. Am. 66 (3): 207. Bibcode:1976JOSA...66..207N. doi:10.1364/JOSA.66.000207.

    See also the inverse operation: js = orders2noll(n, m)

    :param j: The Noll Zernike index, or an ndarray thereof.
    :return: The standard Zernike index, or an ndarray thereof.
    """
    return orders2index(*noll2orders(j))


class BasisPolynomial:  # todo: refactor so that this inherits from Polynomial
    """
    A class representing one of the Zernike basis polynomials, or an array thereof.
    Superpositions of weighted basis polynomials are represented by zernike.Polynomial.
    """
    def __init__(self, j_index: Union[int, Sequence, np.ndarray] = None,
                 n: Union[int, Sequence, np.ndarray] = None, m: Union[int, Sequence, np.ndarray] = None,
                 odd_and_even: bool = False):
        """
        Constructs one of the Zernike basis polynomial or an array thereof.
        The Zernike basis polynomials form a sqrt(pi)*orthonormal basis on the unit disk, for 2x2-unit square,
        multiply with 4 / pi. The returned Zernike polynomials are themselves functions of polar coordinates (rho=0, phi=0)

        ::

            result = BasisPolynomial(n=n, m=m)

        Returns the Zernike polynomial of radial order n and azimuthal frequency m, where m is between -n and n.

        ::
            result = BasisPolynomial(j)

        Returns the standard OSA/ANSI Zernike polynomial with standard coefficient j_index
        The first of which are: piston,
                                 tilt, tip,
                                 oblique-astigmatism, defocus, vertical-astigmatism,
                                 vertical-trefoil, vertical-coma, horizontal-coma,  horizontal-trefoil,
                                 oblique-trefoil, oblique-quadrafoil, oblique-secondary-astigmatism,
                                 spherical aberration, vertical-secondary-astigmatism vertical-quadrafoil, ...
        where the postscripts indicate the position of the extreme value on the pupil edge.

        When many polynomials need to computed, it will be more efficient to compute multiple polynomials in parallel.
        This function can handle rho and phi matrices and n and m vectors, while the option odd_and_even returns the
        even and odd polynomials as a complex result.

        ::

            result = BasisPolynomial(n=n, m=m, odd_and_even = True)

        For m >= 0, returns the even Zernike polynomial(cos) value as the real part, and the odd polynomial(sin)
        value as the imaginary part. For m < 0, the odd Zernike value is returned as the real part, and the even is
        returned as the imaginary part.

        See also: fit, Fit and Polynomial, index2orders(j), noll2orders(j), orders2index(n, m=0), and orders2noll(n, m=0)

        :param j_index: (optional) The standard (OSA) index of the polynomial. This can a non-negative integer or an nd-array of such integers.
        :param n: (optional) The radial order of the polynomial. This can a non-negative integer or an nd-array of such integers.
        :param m: (optional) The azimuthal frequency of the polynomial. This can a integer <= n or an nd-array of such integers.
        :param odd_and_even: A boolean to indicate if the odd or even counterpart should also be returned. When set to true,
            the imaginary parts of the result contain the counterpart of the requested polynomial (default: False).
        """
        if j_index is not None:
            if n is None and m is None:
                n, m = index2orders(j_index)
            else:
                raise ValueError('When the j-index of the Zernike basis polynomial is specified, ' +
                                 'neither order n, nor order m, should be specified.')
        else:
            if n is None or m is None:
                raise ValueError('When the j-index of the Zernike basis polynomial is not specified, ' +
                                 'both order n, and order m, should be specified.')

        self.__n = None
        self.n = np.asarray(n)
        self.__m = None
        self.m = np.asarray(m)

        self.odd_and_even = odd_and_even

    def __call__(self, rho: Union[float, Sequence, np.ndarray] = 0, phi: Union[float, Sequence, np.ndarray] = 0):
        # return _polynomial(self.n, self.m, rho, phi, odd_and_even=self.odd_and_even)
        """
        Returns the Zernike polynomial of order (n, m) evaluated in polar coordinates at rho and phi.
        The result is represented as a numpy ndarray of dimensions equal, or broadcastable, to the shape of rho (and theta),
        or higher dimensions when n and m are also vectors or arrays.
        The arrays: n, m, j, rho, and phi must be broadcastable.

        :param rho: The radian coordinate. When negative, phi is changed by pi.
            This can be a single number or an nd-array with shape that is broadcastable with the orders n and m of the polynomial.
        :param phi: The azimuthal coordinate [-pi, pi). This can be a single number or an nd-array with shape that is
            broadcastable with the orders n and m of the polynomial.
        :return: A numpy ndarray of dimensions equal to the shape of rho (and phi),
            or higher dimensions when n and m are also vectors or arrays.
        """
        rho = np.array(rho)
        phi = np.array(phi)
        # Make orthogonal basis on unit disk (for 2x2 square, set everything outside unit disk to zero and multiply by 4/pi)
        # The norm of each basis vector is sqrt(pi), so that piston(rho, phi) = 1 everywhere.
        normalization = np.sqrt(2 * (self.n + 1) / (1 + (self.m == 0)))
        # Set the real part as requested, the imaginary part will be the odd-counterpart polynomial
        zernike_phase = self.m * (phi + np.pi * (rho < 0)) + (self.m < 0) * np.pi / 2
        if self.odd_and_even:
            zernike_phasor = np.exp(1j * zernike_phase)
        else:
            zernike_phasor = np.cos(zernike_phase)
        result = normalization * self.__polynomial_r(np.abs(rho)) * zernike_phasor

        return result

    @property
    def n(self) -> Union[int, np.ndarray]:
        """
        Get the radial order of the Zernike polynomial.
        """
        return self.__n

    @n.setter
    def n(self, new_radial_order: Union[int, Sequence, np.ndarray]):
        """
        Set the radial order of the Zernike polynomial.
        """
        self.__n = np.array(new_radial_order)

    @property
    def m(self) -> Union[int, np.ndarray]:
        """
        Get the azimuthal order of the Zernike polynomial.
        """
        return self.__m

    @m.setter
    def m(self, new_azimuthal_order: Union[int, Sequence, np.ndarray]):
        """
        Set the azimuthal order of the Zernike polynomial.
        """
        self.__m = np.array(new_azimuthal_order)

    @property
    def j(self) -> Union[int, np.ndarray]:
        """
        Get the standard OSA/ANSI index of the Zernike polynomial.
        """
        return orders2index(n=self.n, m=self.m)

    @j.setter
    def j(self, new_j_index: Union[int, Sequence, np.ndarray]):
        """
        Set the standard OSA/ANSI index of the Zernike polynomial.
        """
        self.n, self.m = index2orders(new_j_index)

    @property
    def symbol(self) -> str:
        return f'Z{subscript(self.n)}{superscript(self.m)}'

    @property
    def name(self) -> str:
        def radial_multiplicity(_: int) -> str:
            prefixes = ['0-', 'prim', 'second', 'terti', 'quatern', 'quint', 'sext', 'sept', 'oct']
            if _ < len(prefixes):
                result = prefixes[_]
            else:
                result = f'{_}-'
            return result + 'ary '

        def azimulthal_multiplicity(m: int) -> str:
            _ = abs(m)
            special_names = ['spherical', 'coma', 'astigmatism']
            if _ < len(special_names):
                return special_names[_]
            else:
                prefixes = ['0-', '1-', '2-', 'tre', 'quadra', 'penta', 'hexa', 'hepta', 'octa', 'nona', 'deca']
                if _ < len(prefixes):
                    prefix = prefixes[_]
                else:
                    prefix = f'{_}-'
                return prefix + 'foil'

        name = ''
        if self.m == 0:
            if self.n == 0:
                name = 'piston'
            elif self.n == 2:
                name = 'defocus'
            else:
                name = radial_multiplicity(self.n // 2 - 1) + azimulthal_multiplicity(self.m)  # Start counting from spherical
        elif abs(self.m) == 1:
            if self.n == 1:
                name = 'tilt' if self.m < 0 else 'tip'
            else:
                name = 'vertical ' if self.m < 0 else 'horizontal '
                if self.n > 3:
                    name += radial_multiplicity((self.n - abs(self.m)) // 2)
                name += azimulthal_multiplicity(self.m)
        else:
            if abs(self.m) % 2 == 0:
                name = 'oblique ' if self.m < 0 else 'vertical '
            else:
                name = 'vertical ' if self.m < 0 else ('horizontal ' if self.n > 3 else 'oblique ')
            if self.n > abs(self.m):
                name += radial_multiplicity(1 + (self.n - abs(self.m)) // 2)
            name += azimulthal_multiplicity(self.m)

        name2 = name

        # names = ['piston', 'tilt', 'tip', 'oblique astigmatism', 'defocus', 'vertical astigmatism',
        #          'vertical trefoil', 'vertical coma', 'horizontal coma', 'oblique trefoil',
        #          'oblique quadrafoil', 'oblique secondary astigmatism', 'primary spherical',
        #          'vertical secondary astigmatism ', 'vertical quadrafoil',
        #          None, 'vertical secondary trefoil', 'vertical secondary coma', 'horizontal secondary coma', 'horizontal secondary trefoil', None,
        #          None, 'oblique secondary quadrafoil', 'oblique tertiary astigmatism', 'secondary spherical', 'vertical tertiary astigmatism', 'vertical secondary', None,
        #          ]
        # name = names[self.j] if self.j < len(names) and names[self.j] else self.symbol

        return name

    @property
    def cartesian(self):
        return lambda y, x: self.__call__(rho=np.sqrt(y**2 + x**2), phi=np.arctan2(y, x))

    def __str__(self):
        return f"zernike.Polynomial(j={self.j}, n={self.n}, m={self.m}) = {self.name}"

    def __polynomial_r(self, rho: Union[float, Sequence, np.ndarray]=0):
        """
        Calculate the radial component, the polynomial polynomial, for all rho in a matrix
        prerequisites: m >= 0, rho >= 0, mod(n - m, 2) == 0
        Output: a matrix of the same shape as rho, or the multidimensional 0 indicating an all zero result in case the difference n - m is odd.

        :param rho: An nd-array with the radial distances. This array must have singleton. Non negativeness is enforced.
        :return: The polynomial values in an nd-array of the same shape as rho_i, but broad-casted over the dimensions
        of n and m.
        """
        rho = np.abs(np.asarray(rho))
        if rho.dtype == int:
            rho = rho.astype(float)

        # Complete the shapes of ns and ms
        n, m = np.broadcast_arrays(self.n, np.abs(self.m))  # Make also sure that m is non-negative
        return self.__polynomial_r_static(n, m, rho)

    @staticmethod
    @disk.cache
    def __polynomial_r_static(n: Union[int, Sequence, np.ndarray], m: Union[int, Sequence, np.ndarray],
                              rho: Union[float, Sequence, np.ndarray]=0):
        """
        Calculate the radial component, the polynomial polynomial, for all rho in a matrix
        prerequisites: m >= 0, rho >= 0, mod(n - m, 2) == 0
        Output: a matrix of the same shape as rho, or the multidimensional 0 indicating an all zero result in case the difference n - m is odd.

        :param n: A non-negative integer or array_like indicating the radial order.
        :param m: An integer or array_like indicating the azimuthal order.
        :param rho: An nd-array with the radial distances. This array must have singleton. Non negativeness is enforced.
        :return: The polynomial values in an nd-array of the same shape as rho_i, but broad-casted over the dimensions
        of n and m.
        """
        # log.info('Evaluating Zernike polynomial...')
        n_m_dim = n.ndim

        # Expand the output to the shape of that of rho_i broadcasted with n and m
        if rho.ndim < 1:
            rho = rho[..., np.newaxis]
        output_shape = (*rho.shape[:rho.ndim-n_m_dim], *np.maximum(np.array(n.shape), np.array(rho.shape[rho.ndim-n_m_dim:])))
        calculation_shape = (*output_shape[:len(output_shape)-n_m_dim], np.prod(output_shape[len(output_shape)-n_m_dim:], dtype=int))
        # log.info(f"rho.shape={rho.shape}, n.shape={n.shape}, output_shape={output_shape}, calculation_shape={calculation_shape}, n_m_dim={n_m_dim}")
        rho = np.broadcast_to(rho, shape=output_shape)

        # Start with the first n_m_dim dimensions flattened
        result = np.zeros(shape=calculation_shape)
        rho = np.reshape(rho, newshape=calculation_shape)
        for idx in range(n.size):
            n_i = n.ravel()[idx]
            m_i = m.ravel()[idx]
            rho_i = rho[..., idx]
            if np.mod(n_i - m_i, 2) == 0:  # Skip odd differences, for these the result is zero.
                rho_pow = rho_i**m_i
                rho_sqd = rho_i**2

                coefficients = np.arange((n_i - m_i) / 2, -1, -1, dtype=int)
                for c_idx, c in enumerate(coefficients):
                    # For speedup: rho_pow = rho_i**(n_i-2*coefficients)
                    if c_idx > 0:
                        rho_pow *= rho_sqd  # note the coefficients are in reversed order

                    sub_result_weight = ((-1.0)**c) * factorial_product_fraction(n_i - c, (c, (n_i + m_i) / 2 - c, (n_i - m_i) / 2 - c))
                    result[..., idx] += sub_result_weight * rho_pow

        return result.reshape(output_shape)


class Polynomial(Callable):
    def __init__(self, coefficients: Union[tuple, Sequence, np.ndarray],
                 indices: Union[tuple, Sequence, np.ndarray] = None):
        """
        Construct an object that represents superpositions of basis-Zernike polynomials.

        :param coefficients: The coefficients of the polynomials.
        :param indices: The standard indices of the polynomials (Default: all).
        """
        self._polynomials = None
        self.__coefficients = None
        if indices is not None:
            raise NotImplementedError('Setting indices is not yet implemented on a zernike Polynomial.')
        self.__indices = indices

        if coefficients is None:
            coefficients = []
        self.coefficients = coefficients

    @property
    def indices(self) -> np.ndarray:
        """ISO/ANSI indices of the basis polynomials."""
        return self.__indices

    @indices.setter
    def indices(self, new_indices: Union[int, Sequence, np.ndarray]):
        new_indices = np.asarray(new_indices).flatten()
        if not np.array_equal(self.__indices, new_indices):
            self._polynomials = BasisPolynomial(j_index=new_indices)
        self.__indices = new_indices

    @property
    def coefficients(self) -> np.ndarray:
        return self.__coefficients

    @coefficients.setter
    def coefficients(self, new_coefficients: Union[float, Sequence, np.ndarray]):
        new_coefficients = np.asarray(new_coefficients).flatten()
        if self.coefficients is None or new_coefficients.size != self.coefficients.size:
            self._polynomials = BasisPolynomial(j_index=range(new_coefficients.size))
        self.__coefficients = new_coefficients

    @property
    def order(self) -> int:
        return self.coefficients.size

    @property
    def cartesian(self) -> Callable:
        """A function to evaluate this polynomial using Cartesian coordinates instead of the default polar coordinates."""
        return lambda y, x: self.__call__(rho=np.sqrt(y**2 + x**2), phi=np.arctan2(y, x))

    def __call__(self, rho: Union[float, Sequence, np.ndarray] = 0, phi: Union[float, Sequence, np.ndarray] = 0
                 ) -> np.ndarray:
        """
        Evaluate this polynomial at polar coordinates.

        :param rho: The radial coordinate between 0 and 1.
        :param phi: The angular coordinate in radians.
        :return: The value of the polynomial at the specified coordinates.
        """
        # Add one axis to the left for broadcasting over the self.__polynomials representation
        rho = np.array(rho)[..., np.newaxis]
        phi = np.array(phi)[..., np.newaxis]

        mat = self._polynomials(rho, phi)
        result = mat @ self.coefficients
        return result

    def __str__(self):
        return f"zernike.Polynomial(coefficients={self.coefficients})"

    def __mul__(self, other: float) -> Polynomial:
        """Return a new Polynomial that equals this one multiplied by a scalar constant."""
        return Polynomial(coefficients=self.coefficients * other)

    def __div__(self, other: float) -> Polynomial:
        """Return a new Polynomial that equals this one divided by a scalar constant."""
        return Polynomial(coefficients=self.coefficients / other)

    def __truediv__(self, other: float) -> Polynomial:
        """Return a new Polynomial that equals this one divided by a scalar constant."""
        return self / other

    def __rdiv__(self, other: float) -> Polynomial:
        """Return a new Polynomial that equals this one divided by a scalar constant."""
        return self / other

    def __imul__(self, other: float):
        """In-place multiply (*=) this Polynomial by a scalar constant."""
        self.coefficients *= other
        return self

    def __idiv__(self, other: float):
        """In-place divide (/=) this Polynomial by a scalar constant."""
        self.coefficients /= other
        return self


# Some definitions for convenience. For more names, check the BasisPolynomial.name property.
piston = BasisPolynomial(n=0, m=0)

tip = BasisPolynomial(n=1, m=-1)
tilt = BasisPolynomial(n=1, m=1)

oblique_astigmatism = BasisPolynomial(n=2, m=-2)
defocus = BasisPolynomial(n=2, m=0)
vertical_astigmatism = BasisPolynomial(n=2, m=-2)

vertical_trefoil = BasisPolynomial(n=3, m=-3)
vertical_coma = BasisPolynomial(n=3, m=-1)
horizontal_coma = BasisPolynomial(n=3, m=1)
oblique_trefoil = BasisPolynomial(n=3, m=3)

primary_spherical = BasisPolynomial(n=4, m=0)


class Fit(Polynomial):
    """
    A class to fits Zernike polynomials up to the given order.
    """
    def __init__(self, z, y=None, x=None, rho=None, phi=None, weights=None, order: int = 15):
        """
        Construct an object to fit Zernike polynomials up to the given order.

        :param z: An nd-array of at least dimension 2, the right-most dimensions are indexed by
            either x and y or by rho and phi.
        :param y: The second Cartesian coordinate. Default: covering the range [-1, 1)
        :param x: The first Cartesian coordinate. Default: covering the range [-1, 1)
        :param rho: Alternative radial coordinate when not using Cartesian coordinates.
        :param phi: Alternative azimuthal coordinate when not using Cartesian coordinates.
        :param weights: An nd-array with per-value weights for the fit.
            Default: None = uniform weighting on the unit disk.
        :param order: The number of polynomial terms to consider.
        """
        super().__init__(coefficients=[])

        if np.isscalar(z):
            z = [z]
        z = np.asarray(z)

        cartesian = rho is None and phi is None
        if cartesian:
            if x is None and y is None:
                x, y = np.linspace(-1, 1, z.shape[-2]), np.linspace(-1, 1, z.shape[-1])

            rho, phi = cart2pol(y, x)

        calc_broadcast = np.broadcast(rho, phi)
        if weights is None:
            # Polar     => proportional to rho in unit disk, 0 outside
            # Cartesian => uniform in unit disk, 0 outside
            inside = np.broadcast_to(rho <= 1, shape=calc_broadcast.shape)
            weights = inside.astype(float)
        else:
            weights = np.broadcast_to(weights, shape=calc_broadcast.shape).astype(float)

        z = np.reshape(z, newshape=(*z.shape[:-2], -1))  # collapse the final (right-most) two dimensions.

        self.__rho = rho
        self.__phi = phi
        self.__z = z
        self.__weights = weights
        self.order = order

        # Lazily calculated
        self.__error = None

    @property
    def order(self) -> int:
        return self.coefficients.size

    @order.setter
    def order(self, new_order: int):
        """
        Sets the order and fits Zernike basis polynomials up to it.
        :param new_order: The number of polynomials to fit.
        """
        self.coefficients = np.zeros(new_order)  # Also determine the polynomials in the super class
        self.coefficients = self.contravariant

    @property
    def contravariant(self):
        basis_vectors = self._polynomials(self.__rho[..., np.newaxis], self.__phi[..., np.newaxis]) \
                        * self.__weights[..., np.newaxis]
        basis_vectors = basis_vectors.reshape(-1, self.order)

        # log.info('Fitting...')
        coefficients, residuals, rank, s = np.linalg.lstsq(basis_vectors, self.__z, rcond=None)
        # log.info(f'residuals={residuals}, rank={rank}, s={s}')

        return coefficients  # Set the coefficients of the underlying Polynomial

    @property
    def covariant(self) -> np.ndarray:
        coefficients = np.zeros(shape=(*self.__z.shape[:-2], self.order))
        for idx in range(self.order):
            basis_vector = (BasisPolynomial(idx)(self.__rho, self.__phi) * self.__weights).ravel()
            coefficients[..., idx] = basis_vector[np.newaxis, :] @ self.__z[..., np.newaxis]
            # log.debug(f"{idx}, {BasisPolynomial(idx).name}: {coefficients[idx]}")
        return coefficients  # Set the coefficients of the underlying Polynomial

    @property
    def error(self):
        """
        The root-mean-square (RMS) fitting error between `f` and `z`.
        ||z - f|| / sqrt(n), where `n` is the number of sample points.
        """
        if self.__error is None:
            self.__error = np.linalg.norm(self(rho=self.__rho, phi=self.__phi) - self.__z) / np.sqrt(self.__z.size)
        return self.__error

    def __str__(self):
        return f"zernike.Fit(coefficients={self.coefficients})"

    # def __mul__(self, other: float):
    #     return Fit(z=self.__z*other, rho=self.__rho, phi=self.__phi, weights=self.__weights, order=self.order)
    #
    # def __rdiv__(self, other):
    #     return Fit(z=self.__z/other, rho=self.__rho, phi=self.__phi, weights=self.__weights, order=self.order)


def fit(z, y=None, x=None, rho=None, phi=None, weights=None, order: int = 15) -> Fit:
    """
    Fits Zernike polynomial up to the given order and returns a Fit object with the coefficients,
     the polynomial, and the fitting error.

    :param z: An nd-array of at least dimension 2, the right-most dimensions are indexed by
    either x and y or by rho and phi.
    :param y: The second Cartesian coordinate. Default: covering the range [-1, 1)
    :param x: The first Cartesian coordinate. Default: covering the range [-1, 1)
    :param rho: Alternative radial coordinate when not using Cartesian coordinates.
    :param phi: Alternative azimuthal coordinate when not using Cartesian coordinates.
    :param weights: An nd-array with per-value coefficients for the fit.
    Default: None = uniform weighting on the unit disk, weighted by rho in case of polar coordinate specification.
    :param order: The number of polynomial terms to consider.
    :return: The Fit object representing the polynomial.
    """
    return Fit(z=z, y=y, x=x, rho=rho, phi=phi, weights=weights, order=order)


def fit_coefficients(z, y=None, x=None, rho=None, phi=None, weights=None, order: int=15) -> np.ndarray:
    """
    Fits (multiple) Zernike polynomials up to the given order.

    :param z: An nd-array of at least dimension 2, the right-most dimensions are indexed by
    either x and y or by rho and phi.
    :param y: The second Cartesian coordinate. Default: covering the range [-1, 1)
    :param x: The first Cartesian coordinate. Default: covering the range [-1, 1)
    :param rho: Alternative radial coordinate when not using Cartesian coordinates.
    :param phi: Alternative azimuthal coordinate when not using Cartesian coordinates.
    :param weights: An nd-array with per-value coefficients for the fit.
    Default: None = uniform weighting on the unit disk, weighted by rho in case of polar coordinate specification.
    :param order: The number of polynomial terms to consider.
    :return: A vector or nd-array of vectors with the polynomial coefficients.
    """
    cartesian = rho is None and phi is None
    if cartesian:
        if x is None and y is None:
            x, y = np.linspace(-1, 1, z.shape[-2]), np.linspace(-1, 1, z.shape[-1])

        rho, phi = cart2pol(y, x)

    calc_broadcast = np.broadcast(rho, phi)
    if weights is None:
        weights = np.broadcast_to((cartesian + (not cartesian) * rho) * (rho <= 1), shape=calc_broadcast.shape)
    else:
        weights = np.broadcast_to(weights, shape=calc_broadcast.shape)

    weights = weights / np.sum(weights)  # Make sure that the sum of the coefficients is 1

    z = np.reshape(z, newshape=(*z.shape[:-2], -1))  # collapse the final (right-most) two dimensions.
    coefficients = np.zeros(shape=(*z.shape[:-2], order))
    for idx in range(order):
        basis_vector = np.conj(BasisPolynomial(idx)(rho, phi) * weights).flatten()
        coefficients[..., idx] = basis_vector @ z
        # log.debug(f"{idx}, {BasisPolynomial(idx).name}: {coefficients[idx]}")

    return coefficients
