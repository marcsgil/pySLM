"""
General functions from https://en.wikipedia.org/wiki/Gaussian_beam

"""
import numpy as np
import scipy.special
from typing import Union, Sequence, Optional
import logging

from optics.calc.arithmetic_beam import ArithmeticBeam
from optics.utils import ft
from optics.utils.display import subsup

array_like = Union[np.ndarray, Sequence, complex, float, int]

log = logging.getLogger(__name__)


def hermite_gaussian(m: int = 0, n: int = 0, beam_waist: float = 1.0):
    """
    Create a specific instance of a Hermite-Gaussian function in Cartesian coordinates.

    :param m: The vertical mode.
    :param n: The horizontal mode.
    :param beam_waist: The beam waist at the focal plane.
    :return: The requested Hermite Gaussian function with the coordinates y and x as input arguments.
    """
    log.warning('A new version of this code exists that is more versatile and easier to use! It can be imported with: from optics.calc.gaussian import Hermite')
    normalization_factor = np.sqrt(2**(1 - (n + m))) / (np.sqrt(np.pi) * np.math.factorial(m) * np.math.factorial(n))

    def hermite_gaussian_function(vertical, horizontal) -> np.ndarray:
        return normalization_factor / beam_waist \
               * np.polynomial.hermite.Hermite.basis(m)(np.sqrt(2) * vertical / beam_waist) \
               * np.polynomial.hermite.Hermite.basis(n)(np.sqrt(2) * horizontal / beam_waist) \
               * np.exp(-sum(_ ** 2 for _ in (vertical, horizontal)) / beam_waist ** 2)

    return hermite_gaussian_function


class Airy(ArithmeticBeam):
    """
    The analytical Airy beam function. Currently only transverse.
    """
    def __init__(self, alpha: Optional[array_like] = None, alpha_rad: Optional[array_like] = None, wavelength: Optional[float] = 1.0, wavenumber: Optional[float] = None, propagation_axis: int = 0):
        """
        Construct a Gaussian function with up to three arguments, one for each Cartesian coordinate.

        :param alpha: The alpha value in units of wavelength.
        :param alpha_rad: Alternatively, the alpha value in radians.
        :param wavelength: (optional) The wavelength (default 1).
        :param wavenumber: (optional) The wavenumber.
        :param propagation_axis: The propagation axis (default 0).
        """
        super().__init__(wavelength=wavelength, wavenumber=wavenumber, propagation_axis=propagation_axis)
        self.__alpha_rad = None
        if alpha_rad is not None:
            self.alpha_rad = alpha_rad
        else:
            self.alpha = alpha

    @property
    def alpha_rad(self) -> np.ndarray:
        """The alpha values in radians, one for each transverse coordinate."""
        return self.__alpha_rad

    @alpha_rad.setter
    def alpha_rad(self, new_values: array_like):
        new_values = np.asarray(new_values)
        if new_values.size == 1:
            new_values = np.full(2, new_values.item())
        self.__alpha_rad = new_values

    @property
    def alpha(self) -> np.ndarray:
        """The alpha values in units of wavelength (2pi radians), one for each transverse dimension."""
        return self.__alpha_rad / (2.0 * np.pi)

    @alpha.setter
    def alpha(self, new_values: array_like):
        new_values = np.asarray(new_values)
        if new_values.size == 1:
            new_values = np.full(2, new_values.item())
        self.alpha_rad = new_values * 2.0 * np.pi

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

        # Calculate the Airy beam without an aperture using its analytical expression.
        # https://en.wikipedia.org/wiki/Airy_beam#Mathematical_description
        x0_inv = -self.wavenumber * (self.alpha_rad[:, np.newaxis, np.newaxis, np.newaxis] * 3) ** (-1/3)   # 1 / x0
        y = y * x0_inv[0]
        x = x * x0_inv[1]
        half_xi = z / self.wavenumber * (x0_inv ** 2) / 2
        half_xi_sqd = half_xi ** 2
        airy_field = scipy.special.airy(y - half_xi_sqd[-2])[0] * scipy.special.airy(x - half_xi_sqd[-1])[0] \
            * np.exp(1j * (half_xi[-2] * y + half_xi[-1] * x - np.sum(half_xi * half_xi_sqd) / 3))

        return airy_field

    def __str__(self) -> str:
        alphas = [round(_) if np.isclose(_, round(_)) else _ for _ in self.alpha]
        return f"A{subsup.subscript(alphas[0])}{subsup.superscript(alphas[1]) if alphas[1] != alphas[0] else ''}"

    def __repr__(self) -> str:
        return f'{__class__.__name__}({self.alpha})'


if __name__ == '__main__':
    from examples import log

    from optics.calc.gaussian import Gaussian, Laguerre, Hermite

    log.info(hermite_gaussian(0, 3)(0.1, 0.2))
    log.info(Hermite(0, 3)(0.1, 0.2))  # The new one is larger by sqrt(n!m!) == np.sqrt(np.math.factorial(n) * np.math.factorial(m))
