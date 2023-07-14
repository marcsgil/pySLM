"""
A mix-in class that adds general arithmetic ability to beam functions.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Sequence, Optional, Callable
import logging
import numpy as np
from functools import reduce

from optics.utils import ft, polar
from optics.utils.display import subsup

__all__ = ['ArithmeticBeam']

log = logging.getLogger(__name__)


array_like = Union[np.ndarray, Sequence, complex, float, int]


class ArithmeticBeam(ABC):
    def __init__(self, wavelength: Optional[float] = 1.0, wavenumber: Optional[float] = None, propagation_axis: int = 0):
        """
        Construct a generic beam function with up to three arguments, one for each Cartesian coordinate. Use the
        `cylindrical` property to use polar or cylindrical coordinates instead.
        This class  must be inherited by a subclass that implements the call method.

        :param wavelength: (optional) The wavelength (default 1).
        :param wavenumber: (optional) The wavenumber.
        :param propagation_axis: The propagation axis (default 0).
        """
        if wavenumber is None:
            wavenumber = 2 * np.pi / wavelength
        self.__wavenumber = wavenumber
        self.__propagation_axis = propagation_axis

    @property
    def wavelength(self) -> float:
        r"""The wavelength in the material, i.e.~:math:`\lambda / n`."""
        return 2 * np.pi / self.__wavenumber

    @wavelength.setter
    def wavelength(self, new_wavelength: float):
        self.__wavenumber = 2 * np.pi / new_wavelength

    @property
    def wavenumber(self) -> float:
        """The wavenumber in the material, i.e. :math:`k_0 n`."""
        return self.__wavenumber

    @wavenumber.setter
    def wavenumber(self, new_wavenumber: float):
        self.__wavenumber = new_wavenumber

    @property
    def propagation_axis(self) -> int:
        """
        The propagation_axis of the beam at its focal plane, i.e. at z=0.
        The default axis is 0. When another axis is specified, the remaining axes are shifted without changing their order."""
        return self.__propagation_axis

    @propagation_axis.setter
    def propagation_axis(self, new_propagation_axis: int):
        self.__propagation_axis = new_propagation_axis

    """An abstract class of beams to which arithmetic can be applied."""
    def __add__(self, other: Union[complex, ArithmeticBeam]) -> Sum:
        """Add two beams together using the `+` operator."""
        return Sum(self, other)

    def __radd__(self, other: Union[complex, ArithmeticBeam]) -> Sum:
        """Add a constant and this beam together using the `+` operator."""
        return Sum(other, self)

    def __mul__(self, other: Union[complex, ArithmeticBeam]) -> Product:
        """Multiply a beam with a number or another beam using `*`."""
        return Product(self, other)

    def __rmul__(self, other: Union[complex, ArithmeticBeam]) -> Product:
        """Multiply a number with this beam using `*`."""
        return Product(other, self)

    def __invert__(self) -> ArithmeticBeam:
        """The reciprocal of the values of this beam. Use this with the unary operator ~"""
        return Reciprocal(self)

    def __div__(self, other: Union[complex, ArithmeticBeam]) -> Product:
        """Divide beam by a constant scalar or another beam. This is the same as __truediv__"""
        return self / other

    def __truediv__(self, other: Union[complex, ArithmeticBeam]) -> Product:
        """Divide beam by a constant scalar or another beam using `/`."""
        if isinstance(other, ArithmeticBeam):
            return self * ~other
        else:
            return self * (1 / other)

    def __rdiv__(self, other: Union[complex, ArithmeticBeam]) -> Product:
        """Divide a constant scalar by this beam using `/`."""
        return ~self * other

    def __neg__(self) -> Product:
        """Make this beam negative by prepending `-` to it."""
        return Product(self, -1.0)

    def __sub__(self, other: Union[complex, ArithmeticBeam]) -> Sum:
        """Subtract two beams using the `-`operator."""
        return Sum(self, -other)

    def __abs__(self) -> Abs:
        """Take the absolute value of this beam using `abs()` or `numpy.abs`"""
        return Abs(self)

    def __pow__(self, other: Union[complex, ArithmeticBeam]) -> Power:
        """Raise this beam to a power using the `**` operator."""
        return Power(self, other)

    def __call__(self, *args: Union[array_like, ft.Grid], **kwargs: array_like) -> np.ndarray:
        """
        Computes the beam's field values at the specified Cartesian coordinates.
        Coordinate vectors are broadcast and a Grid object can be used instead.

        This is a short-hand for the `cartesian` method.
        See also: `cylindrical` or `polar`

        :param z: The z-coordinate, usually the longitudinal coordinate.
        :param y: The y-coordinate, usually transverse.
        :param x: The x-coordinate, usually transverse.

        :return: The field at the specified coordinates.
        """
        return self.cartesian(*args, **kwargs)

    def _args_to_ranges(self, *args: Union[array_like, ft.Grid], **kwargs: array_like) -> Sequence[np.ndarray]:
        """Converts arguments to numerical ranges, insert default values as necessary."""
        # Get all the keyword ranges first
        range_dict = dict()
        for _, dim_labels in enumerate((['z'], ['y', 'rho', 'r'], ['x', 'phi', 'p', 'theta'])):
            for dim_label in dim_labels:
                if dim_label in kwargs:
                    range_dict[3 - _] = kwargs[dim_label]  # Use a negative key
        # List all the non-keyword ranges
        non_keyword_ranges = []
        for arg in args:
            if isinstance(arg, ft.Grid):
                non_keyword_ranges += [*arg]
            elif arg is not None:
                non_keyword_ranges.append(np.asarray(arg))
        # Combine non-keyword and keyword ranges
        ranges = []
        for _ in range(3):  # fill in backwards
            if _ in range_dict:
                ranges.append(_)
            elif len(non_keyword_ranges) > 0:
                ranges.append(non_keyword_ranges[-1])
                non_keyword_ranges = non_keyword_ranges[:-1]
            else:
                ranges.append(np.zeros(1))
        if len(non_keyword_ranges) > 0:
            raise ValueError(f'Maximum 3 ranges are allowed, including those in Grid objects and keyword arguments, got {len(non_keyword_ranges)}.')
        ranges = ranges[::-1]
        try:
            np.broadcast_shapes(*[_.shape for _ in ranges])
        except ValueError as ve:
            raise ValueError(f"{ve}\nMake sure that the beam coordinates are three-dimensional vectors or arrays that broadcast.")
        return ranges

    def _args_to_ranges_cart(self, *args: Union[array_like, ft.Grid], **kwargs: array_like) -> Sequence[np.ndarray]:
        """
        Converts arguments to numerical ranges, insert default values as necessary.
        Also sorts the axes when the propagation axis is not the left-most.
        """
        ranges = self._args_to_ranges(*args, **kwargs)
        # move the propagation axis to position 0
        ranges = (ranges[self.propagation_axis], *ranges[:self.propagation_axis], *ranges[self.propagation_axis+1:])
        return ranges

    def cartesian(self, *args: Union[array_like, ft.Grid], **kwargs: array_like) -> np.ndarray:
        """
        Computes the beam's field values at the specified Cartesian coordinates.

        Coordinate vectors are broadcast and a Grid object can be used instead.
        Subclasses should implement either the `cartesian` method or the `cylindrical` method.
        When arguments are omitted, those on their **left** are assumed to be 0. Keyword arguments take precedence.

        :param z: The z-coordinate, usually the longitudinal coordinate.
        :param y: The y-coordinate, usually transverse.
        :param x: The x-coordinate, usually transverse.

        :return: The field at the specified coordinates.
        """
        z, y, x = self._args_to_ranges_cart(*args, **kwargs)
        rho, phi = polar.cart2pol(y, x)
        return self.cylindrical(z, rho, phi)

    def cylindrical(self, *args: Union[array_like, ft.Grid], **kwargs: array_like) -> np.ndarray:
        """
        Computes the beam's field values at the specified cylindrical coordinates.

        Coordinate vectors are broadcast and a Grid object can be used instead. Use the :obj:`optics.polar` package
        to convert between Cartesian and polar or cylindrical coordinates. This method is synonym to `polar`.
        When arguments are omitted, those on their **left** are assumed to 0. Keyword arguments take precedence.

        :param z: The longitudinal coordinate.
        :param rho: The radial coordinate.
        :param phi: The azimuthal coordinate in radians.

        :return: The field at the specified coordinates.
        """
        z, rho, phi = self._args_to_ranges(*args, **kwargs)
        y, x = polar.pol2cart(rho, phi)
        return self.cartesian(z, y, x)

    def polar(self, *args, **kwargs):
        """
        Computes the beam's field values at the specified cylindrical coordinates.
        Coordinate vectors are broadcast and a Grid object can be used instead. Use the :obj:`optics.polar` package
        to convert between Cartesian and polar or cylindrical coordinates.

        Alternative name for `cylindrical`.

        :param z: The longitudinal coordinate.
        :param rho: The radial coordinate.
        :param phi: The azimuthal coordinate in radians.

        :return: The field at the specified coordinates.
        """
        return self.cylindrical(*args, **kwargs)

    @property
    def intensity(self) -> ArithmeticBeam:
        """
        Returns a new beam that represents the absolute-square (intensity) of this one.
        """
        return abs(self) ** 2

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class Reduction(ArithmeticBeam):
    """A beam that combines multiple beams using a binary operation."""
    def __init__(self, reduction_operation: Callable[[array_like, array_like], array_like],
                 reduction_identity: complex = 1.0,
                 str_reduction_operation: Callable[[str, str], str] = lambda a, b: a + b,
                 beams: Sequence[Union[ArithmeticBeam, complex]] = tuple()):
        print(f'1={reduction_identity}, beams = {beams}')
        self.__beams = []
        self.__scalar = None
        self.__reduction_operation: Callable[[array_like, array_like], array_like] = reduction_operation  # used for both the scalar and the beams.
        self.__reduction_identity: complex = reduction_identity
        self.__str_reduction_operation: Callable[[str, str], str] = str_reduction_operation

        self.beams += beams  # sets both beams and scalar
        super().__init__(wavenumber=self.beams[0].wavenumber, propagation_axis=self.beams[0].propagation_axis)

    @property
    def beams(self) -> Sequence[ArithmeticBeam]:
        """The component beams."""
        return self.__beams

    @beams.setter
    def beams(self, new_beams: Sequence[Union[ArithmeticBeam, complex]]):
        """Set the component beams."""
        def flatten_beams(*beams: Union[ArithmeticBeam, complex]):
            beam_list = []
            for _ in beams:
                if isinstance(_, ArithmeticBeam):
                    if not isinstance(_, self.__class__):
                        beam_list.append(_)
                    else:  # Another Sum or Product
                        beam_list += flatten_beams(*_.beams)
            return beam_list

        def flatten_scalars(*beams: Union[ArithmeticBeam, complex]):
            print(f'Flattening {beams}...')
            scalar = self.__reduction_identity
            for _ in beams:
                if isinstance(_, ArithmeticBeam):
                    if isinstance(_, self.__class__):  # Another Sum or Product
                        scalar = self.__reduction_operation(scalar, _.scalar)
                else:  # A scalar value
                    scalar = self.__reduction_operation(scalar, _)
            return scalar

        self.__beams = flatten_beams(*new_beams)
        self.scalar = flatten_scalars(*new_beams)

    @property
    def scalar(self) -> complex:
        """The scalar associated with these beams."""
        return self.__scalar

    @scalar.setter
    def scalar(self, new_value: complex):
        self.__scalar = new_value

    def cartesian(self, *args, **kwargs) -> np.ndarray:
        return reduce(self.__reduction_operation,
                      (_(*args, **kwargs) for _ in self.beams),
                      self.__reduction_operation(self.__reduction_identity, self.scalar))

    def __str__(self) -> str:
        # def format_real(number: float, first: bool = False, drop_unity: bool = False) -> str:
        #     if number % 1 == 0:
        #         number_str = f'{number:+0.0f}'  # remove the fraction
        #     else:
        #         number_str = f'{number:+0.6g}'
        #     if drop_unity and abs(number) == 1:
        #         number_str = number_str[0] + number_str[2:]  # drop the 1, leave the sign
        #     if first and number_str[0] == '+':
        #         number_str = number_str[1:]  # drop the sign
        #     return number_str

        scalar = self.scalar
        scalar_str = f'{scalar}'
        # if scalar == 0.0:
        #     scalar_str += '0'
        # else:
        #     if scalar.real != 0.0:
        #         scalar_str += format_real(scalar.real, first=True, drop_unity=scalar.imag == 0.0)
        #     if scalar.imag != 0.0:
        #         scalar_str += format_real(scalar.imag, first=scalar.real == 0.0, drop_unity=True) + 'i'
        #     if scalar.real != 0.0 and scalar.imag != 0.0:
        #         scalar_str = '(' + scalar_str + ')'
        # scalar_str += ' '

        return reduce(self.__str_reduction_operation, (str(_) for _ in self.beams), scalar_str)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}([{self.scalar}, {', '.join(repr(_) for _ in self.beams)}])"


class Product(Reduction):
    """A beam that is the product of multiple beams or scalars."""
    def __init__(self, *beams: Union[ArithmeticBeam, complex]):
        super().__init__(reduction_operation=lambda a, b: a * b, reduction_identity=1, beams=beams)


class Sum(Reduction):
    """A beam that is the sum of multiple beams or constants."""
    def __init__(self, *beams: Union[ArithmeticBeam, complex]):
        def str_reduction_operation(a: str, b: str) -> str:
            return a + ('+' if b[0] != '-' else '') + b
        super().__init__(reduction_operation=lambda a, b: a + b, reduction_identity=0,
                         str_reduction_operation=str_reduction_operation, beams=beams)


class Reciprocal(ArithmeticBeam):
    """A beam that is the reciprocal (1/) of another beam."""
    def __init__(self, beam: ArithmeticBeam):
        super().__init__(wavenumber=beam.wavenumber, propagation_axis=beam.propagation_axis)
        self.__beam = beam

    @property
    def beam(self) -> ArithmeticBeam:
        return self.__beam

    def __invert__(self) -> ArithmeticBeam:
        return self.beam

    def cartesian(self, *args, **kwargs) -> np.ndarray:
        return 1.0 / self.beam(*args, **kwargs)

    def __str__(self) -> str:
        return f'1/{self.beam}'

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.beam)})"


class Abs(ArithmeticBeam):
    """A beam that is the absolute value of another beam."""
    def __init__(self, beam: ArithmeticBeam):
        super().__init__(wavenumber=beam.wavenumber, propagation_axis=beam.propagation_axis)
        self.__beam = beam

    @property
    def beam(self) -> ArithmeticBeam:
        return self.__beam

    def cartesian(self, *args, **kwargs) -> np.ndarray:
        return np.abs(self.beam(*args, **kwargs))

    def __str__(self) -> str:
        return f'|{self.beam}|'

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.beam)})"


class Power(ArithmeticBeam):
    """A beam that is the absolute value of another beam."""
    def __init__(self, beam: ArithmeticBeam, exponent: Union[ArithmeticBeam, complex] = 1.0):
        super().__init__(wavenumber=beam.wavenumber, propagation_axis=beam.propagation_axis)
        self.__beam = beam
        self.__exponent = exponent

    @property
    def beam(self) -> ArithmeticBeam:
        return self.__beam

    @property
    def exponent(self) -> Union[ArithmeticBeam, complex]:
        return self.__exponent

    def cartesian(self, *args, **kwargs) -> np.ndarray:
        if isinstance(self.exponent, ArithmeticBeam):
            return self.beam(*args, **kwargs) ** self.exponent(*args, **kwargs)
        else:
            return self.beam(*args, **kwargs) ** self.exponent

    def __str__(self) -> str:
        exponent = self.exponent
        if isinstance(exponent, ArithmeticBeam):
            exp_str = f'^{exponent}'
        else:
            if np.isclose(exponent, round(exponent)):
                exponent = round(exponent)
            if exponent != 1:
                exp_str = subsup.superscript(exponent)
            else:
                exp_str = ''
        return f'{self.beam}{exp_str}'

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.beam)}, {repr(self.exponent)})"


