"""
Extends the root finding functionality of ``from numpy.polynomial import polynomial`` to arrays of polynomials.

At the moment, it cannot solve polynomials beyond 4th order. These could be solved numerically using an eigenvalue
solver on the characteristic polynomial, provided that a parallel implementation exists for numpy / scipy.

As it stands, polynomials must be of the order indicated by the number of coefficients. I.e. the highest order
coefficient should not be 0. This would cause division by 0s. todo: can a solution be found for this?
"""
import numpy as np
from typing import Union, Sequence, Optional
from numbers import Number
import logging
from optics.utils.display.subsup import superscript
from optics.utils.array import vector_to_axis

__all__ = ['Polynomial', 'find_roots']

log = logging.getLogger(__name__)

array_like = Union[np.ndarray, Sequence, Number]


class Polynomial:
    """
    A class to represent polynomials with coefficients and roots that are :py:func:`numpy.ndarray`s.
    """
    def __init__(self, coefficients: Optional[array_like], roots: Optional[array_like] = None, scale: array_like = 1,
                 axis: int = 0, dtype=None):
        """
        Constructs an object that represents an array of polynomials, either from their coefficients, or from their
        roots and scale.

        :param coefficients: The coefficients as an array-like object with the coefficient axis as per the axis argument.
        :param roots: Optional, instead of coefficients, the roots of the polynomial.
        :param scale: Optional, instead of coefficients, the coefficient of the largest order of the polynomial.
        :param axis: The axis of the polynomials in the array.
        :param dtype: The dtype of the roots. Must be complex.
        """
        self.__axis = axis
        self.__roots = None
        self.__scale = None

        if coefficients is not None:
            self.__roots = np.empty([], dtype=(np.asarray(coefficients).ravel()[0] + 0j).dtype if dtype is None else dtype)
            self.__roots = np.empty_like(self.__to_internal(coefficients)[:-1])
            self.coefficients = coefficients
            if roots is not None:
                log.warning('The coefficients and roots arguments are both specified. Ignoring roots.')
            if scale != 1:
                log.warning('The coefficients and scale arguments are both specified. Ignoring scale.')
        elif roots is not None:
            self.roots = roots
            self.scale = scale
        else:
            raise ValueError('Neither the argument roots nor the argument coefficients is specified.')

    @property
    def dtype(self):
        """The dtype of the values in the polynomial should always be complex."""
        return self.__roots.dtype

    @property
    def axis(self) -> int:
        """The axis of the coefficients and roots of this polynomial."""
        return self.__axis

    @axis.setter
    def axis(self, new_axis: int):
        """Set the axis of the coefficients and roots of this polynomial."""
        new_axis = new_axis % self.ndim
        self.__roots = np.moveaxis(self.__roots, self.__axis, new_axis)
        self.__axis = new_axis

    def __to_internal(self, arr: array_like) -> np.ndarray:
        """Converts an array-like object with coefficients or roots along the axis of this polynomial to the internal
        dtype == self.dtype and shape == (arr.shape[self.axis], *self.shape)."""
        arr = np.asarray(arr, dtype=self.dtype)
        while arr.ndim < self.ndim + 1:
            arr = arr[np.newaxis]
        return np.moveaxis(arr, self.axis, 0)

    def __to_external(self, arr: np.ndarray) -> np.ndarray:
        """Converts the shape to have the coefficient axis as self.axis. If self.axis == 0, and self.order == 1, the
        result will have a singleton dimension on the left."""
        return np.moveaxis(arr, 0, self.axis)

    @property
    def roots(self) -> np.ndarray:
        """The roots of this polynomial in arbitrary order."""
        return self.__to_external(self.__roots).copy()

    @roots.setter
    def roots(self, new_roots: array_like):
        """Set the roots of this polynomial in arbitrary order."""
        self.__roots = self.__to_internal(new_roots)

    @property
    def scale(self) -> np.ndarray:
        """The coefficient of the highest order term."""
        return self.__to_external(self.__scale[np.newaxis])

    @scale.setter
    def scale(self, new_scale: array_like):
        """Set the coefficient of the highest order term."""
        self.__scale = self.__to_internal(new_scale)

    @property
    def order(self) -> int:
        """The order of the polynomial."""
        return self.__roots.shape[0]

    @property
    def shape(self) -> np.ndarray:
        """The shape of the polynomial, without the axis of the coefficients."""
        return np.asarray(self.__roots.shape[1:])

    @property
    def ndim(self) -> int:
        """The number of dimensions of the polynomial array, excluding the axis of the coefficients."""
        return self.shape.size

    def __find_roots(self, coefficients: array_like) -> np.ndarray:
        """
        Finds the roots of a polynomial specified by coefficients.

        :param coefficients: An array with the first dimension one larger than the degree of the polynomial, listing the
        coefficients for low order to high order.
        :return: The roots as an array of the same dimensions of ``coefficients``, but a shape that is one less in the first dimension.
        """
        normalized_coefficients = np.asarray(coefficients[:-1]) / coefficients[-1]
        order = normalized_coefficients.shape[0]
        if order < 1:
            return np.full(normalized_coefficients.shape, np.NAN, dtype=self.dtype)
        phasors = vector_to_axis(np.exp(2j * np.pi / order * np.arange(order)), axis=0, ndim=self.ndim + 1)
        if order < 2:
            return -normalized_coefficients
        elif order == 2:
            c, b = normalized_coefficients
            disc_sqrt = (b**2 - 4*c) ** (1/2)
            return (-b + phasors * disc_sqrt) / order
        elif order == 3:
            d, c, b = normalized_coefficients
            delta_0 = b**2 - 3 * c
            delta_1 = 2 * b**3 - 9 * b * c + 27 * d
            disc_sqrt = (delta_1**2 - 4 * delta_0**3 + 0j) ** (1/2)  # * (delta_0 != 0) + delta_1 * (delta_0 == 0)
            delta2_plus_disc_sqrt = delta_1 + disc_sqrt  # when zero, use the next
            delta2_minus_disc_sqrt = delta_1 - disc_sqrt
            u = ((delta2_plus_disc_sqrt + delta2_minus_disc_sqrt * (delta_0 == 0)) / 2) ** (1/3)  # if delta_0 then delta_plus_disc_sqrt also
            u = u * phasors
            return - (b + u + delta_0 / (u + (u == 0.0))) / order  # when u == 0, so will be delta_0, and this term should be ignored
        elif order == 4:
            e, d, c, b = normalized_coefficients
            p = (8 * c - 3 * b**2) / 8
            q = (b**3 - 4 * b * c + 8 * d) / 8
            r = (-3 * b ** 4 + 256 * e - 64 * b * d + 16 * b ** 2 * c) / 256
            m_all = self.__find_roots([-q[np.newaxis] ** 2, 2 * p[np.newaxis] ** 2 - 8 * r[np.newaxis], 8 * p[np.newaxis], 8])
            m = m_all[0] + (m_all[1] + m_all[2] * np.isclose(m_all[1], 0)) * np.isclose(m_all[0], 0)
            left = 2 * p + 2 * m
            right = (2 / (m + (m==0))) ** (1/2) * q
            pos = (-left - right) ** (1/2)
            neg = (-left + right) ** (1/2)
            sqrt2m = (2 * m) ** (1/2)
            y_pp = sqrt2m + pos
            y_pn = sqrt2m - pos
            y_np = -sqrt2m - neg
            y_nn = -sqrt2m + neg
            return - b / order + np.stack([y_pp, y_pn, y_np, y_nn]) / 2
        elif order > 4:
            raise NotImplementedError('Cannot solve polynomials of order higher than 4.')

    def __poly_from_roots(self, roots: array_like) -> np.ndarray:
        """
        Calculate the polynomial corresponding to a set of roots.

        :param roots: An array with the roots in the first (left-most) dimension.
        :return: An array of coefficients with the first dimension one larger than the number of roots, listing the
        coefficients for low order to high order.
        """
        if roots.shape[0] < 1:
            coefficients = np.full([1, *roots.shape[1:]], np.NAN, dtype=self.dtype)
        elif roots.shape[0] == 1:
            coefficients = np.stack([-roots[0], np.ones_like(roots[0])])
            coefficients *= self.__scale
        else:
            z = np.zeros_like(roots[0])
            tail_coefficients = self.__poly_from_roots(roots[1:])
            coefficients = np.stack([z, *tail_coefficients]) - roots[0] * np.stack([*tail_coefficients, z])

        return coefficients

    @property
    def coefficients(self) -> np.ndarray:
        """The coefficients of this polynomial, from lowest to highest order."""
        return self.__to_external(self.__poly_from_roots(self.__roots))

    @coefficients.setter
    def coefficients(self, new_coefficients: array_like):
        """Set the coefficients of this polynomial, listed from lowest to highest order."""
        new_coefficients = self.__to_internal(new_coefficients)
        self.__scale = new_coefficients[-1]
        self.__roots = self.__find_roots(new_coefficients)

    def __exec__(self, x: array_like) -> np.ndarray:
        """
        Execute this polynomial at the point x.

        :param: x: The independent variable.
        :returns: The dependent variable y.
        """
        x = self.__to_internal(x)
        if self.order > 0:
            y = self.scale * (x - self.__roots[0])
            for _ in self.__roots[1:]:
                y *= x - _
        else:
            y = self.scale.copy()
        return self.__to_external(y)

    @classmethod
    def __format(cls, coefficients: array_like, format_spec: Optional[str]) -> str:
        """
        Formats polynomial coefficients as a unicode string. This is used by the __format__ method on the coefficients of this polynomial.

        :param coefficients: The coefficients of the polynomial.
        :param format_spec: Format specifier for the polynomial and its coefficients.
        :return: A unicode strings describing the polynomials.
        """
        if coefficients.ndim > 1:
            return '[' + (', '.join(cls.__format(_, format_spec) for _ in coefficients)) + ']'
        else:
            space = ' ' if len(format_spec) > 1 and format_spec[0] == ' ' else ''
            result = []
            for _, val in enumerate(coefficients):
                if np.isclose(val.real, 0):
                    val = 1j * val.imag
                if np.isclose(val.imag, 0):
                    val = val.real
                term_str = space if _ > 0 else ''
                if val != 0:
                    if _ < len(coefficients)-1 and val.real > 0:
                        term_str += '+'
                    if _ == 0 or val != 1:
                        if _ == 0 or val != -1:
                            term_str += format(val, format_spec)
                        else:
                            term_str += '-'
                    if _ > 0:
                        term_str += 'x'
                    if _ > 1:
                        term_str += superscript(_)
                    result.append(term_str)
            result_str = ''.join(result[::-1]) + space + '=' + format(0, format_spec)
            return result_str

    @classmethod
    def __format_root_form(cls, roots: array_like, scale: array_like, format_spec: Optional[str]) -> str:
        """
        Formats polynomial coefficients as a unicode string. This is used by the __format__ method on the coefficients of this polynomial.

        :param roots: The roots of the polynomial.
        :param scale: The scale of the polynomial.
        :param format_spec: Format specifier for the polynomial and its coefficients.
        :return: A unicode strings describing the polynomials.
        """
        if roots.ndim > 1:
            return '[' + (', '.join(cls.__format_root_form(r, s, format_spec) for r, s in zip(roots, scale))) + ']'
        else:
            space = ' ' if len(format_spec) > 1 and format_spec[0] == ' ' else ''
            result = []
            if np.isclose(scale.imag, 0):
                scale = scale.real
            if np.isclose(scale, 1):
                pass
            elif np.isclose(scale, -1):
                result.append(space + '-')
            else:
                result.append(format(scale.item(), format_spec))
            for _, val in enumerate(roots):
                if np.isclose(val.imag, 0):
                    val = val.real
                result.append(f'(x{space}{format(-val.item(), format_spec)})')
            result_str = ''.join(result) + space + '=' + format(0, format_spec)
            return result_str

    def __format__(self, format_spec: Optional[str]) -> str:
        """Formats a polynomial as a unicode string."""
        if format_spec.strip().startswith('*'):
            return self.__format_root_form(self.roots, self.scale, format_spec.strip()[1:])
        else:
            return self.__format(self.coefficients, format_spec)

    def __str__(self) -> str:
        return format(self)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.coefficients}, axis={self.axis})'


def find_roots(coefficients: Optional[array_like], axis: int = 0, dtype=None):
    """
    Finds the roots of an array of polynomials.

    :param coefficients: The coefficients as an array-like object with the coefficient axis as per the axis argument.
    :param axis: The axis of the polynomials in the array.
    :param dtype: The dtype of the roots. Must be complex.
    """
    return Polynomial(coefficients=coefficients, axis=axis, dtype=dtype).roots


if __name__ == '__main__':
    from optics.calc import log

    def check_roots(test_coeffs):
        for test_coeff in test_coeffs:
            poly = Polynomial(coefficients=test_coeff, axis=-1)
            x = poly.roots
            returned_coeff = poly.coefficients
            log.info(f'{Polynomial(test_coeff, axis=-1)}: x = {x}: {poly}')
            if np.linalg.norm(returned_coeff - test_coeff) / np.linalg.norm(test_coeff) > 1e-8:
                log.error(f'{Polynomial(test_coeff, axis=-1)} != {poly}')
                log.error(f'Relative error = {np.linalg.norm(returned_coeff - test_coeff)}')

    log.info('Calculating the roots of linear polynomials...')
    check_roots([
        [0, 1],
        [-1, 1],
        [0, 1],
        [-4, 1],
        [2, 2],
        [1, 1],
        [2, 1],
        [[6, 3], [2, 1], [1, 1]],
    ])

    log.info('Testing quadratics...')
    check_roots([
        [0, 0, 1],
        [-1, 0, 1],
        [1, 0, 1],
        [-4, 0, 1],
        [2, 0, 2],
        [1, 0, 1],
        [-1, 0, -1],
        [1, -2, 1],
        [2, -3, 1],
        [2, -4, 2],
    ])

    log.info('Testing cubics...')
    check_roots([
        [0, 0, 0, 1],
        [-1, 0, 0, 1],
        [1, 0, 0, 1],
        [-8, 0, 0, 1],
        [2, 0, 0, 2],
        [1, 0, 0, 1],
        [-1, 0, 0, -1],
        [-1, 1, 0, 1],
        [2, -3, 0, 1],
        [-1, 0, 1, 1],
        [2, 0, -3, 1],
        [-1, 0, 1, 1],
        [2, -3, -3, 1],
    ])

    log.info('Testing quartics...')
    check_roots([
        [0, 0, 0, 0, 1],
        [-1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [-8, 0, 0, 0, 1],
        [2, 0, 0, 0, 2],
        [1, 0, 0, 0, 1],
        [-1, 0, 0, 0, -1],
        [-1, 1, 0, 0, 1],
        [2, -3, 0, 0, 1],
        [-1, 0, 1, 0, 1],
        [2, 0, -3, 0, 1],
        [-1, 0, 1, 0, 1],
        [2, -3, -3, 0, 1],
        [2, -3, 6, -3, 1],
        [[2, -3, 6, -3, 1], [2, 0, -3, 0, 1]],
    ])

    # log.info('Testing random polynomials...')
    # check_roots(np.random.randn(2000, 5) + 1j * np.random.randn(2000, 5))

    log.info('Testing string display...')
    poly = Polynomial([2, -4, 2])
    log.info(f'[{poly:*.0f}] == [{poly:.0f}]')
    log.info(f'[{poly:* .0f}] == [{poly: .0f}]')
    poly = Polynomial([-2, 4, -2])
    log.info(f'[{poly:*.0f}] == [{poly:.0f}]')
    log.info(f'[{poly:* .0f}] == [{poly: .0f}]')

