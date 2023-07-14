"""
General functions from https://en.wikipedia.org/wiki/Gaussian_beam

TODO: Ince-Gaussian and HyperGeometric-Gaussian modes.
"""
from __future__ import annotations

import numpy as np
import scipy.special
from typing import Union, Sequence, Optional
import logging

from optics.utils import ft
from optics.utils import polar
from optics.calc.arithmetic_beam import ArithmeticBeam
from optics.utils.display import subsup

__all__ = ['Gaussian', 'Hermite', 'Laguerre']

log = logging.getLogger(__name__)


array_like = Union[np.ndarray, Sequence, complex, float, int]


class Gaussian(ArithmeticBeam):
    """
    A class the represents a Gaussian beam and that serves as the basis for its generalizations.
    The amplitude of the Gaussian beam is such that its total intensity flux is 1 and its phase = 0 at the focal point.
    """
    def __init__(self, waist: float = 1.0, wavelength: Optional[float] = 1.0, wavenumber: Optional[float] = None, propagation_axis: int = 0):
        """
        Construct a Gaussian function with up to three arguments, one for each Cartesian coordinate.

        :param waist: The beam waist.
        :param wavelength: (optional) The wavelength (default 1).
        :param wavenumber: (optional) The wavenumber.
        :param propagation_axis: The propagation axis (default 0).
        """
        super().__init__(wavelength=wavelength, wavenumber=wavenumber, propagation_axis=propagation_axis)
        self.__waist = waist

    @property
    def waist(self) -> float:
        r"""
        The waist of the beam at its focal plane (z=0). It is defined as the radius at which the field amplitude and
         intensity fall to :math:`\frac{1}{e}` and :math:`\frac{1}{e^2}`, respectively.
        """
        return self.__waist

    @waist.setter
    def waist(self, new_waist: float):
        self.__waist = new_waist

    @property
    def full_width_at_half_maximum(self) -> float:
        """
        The full-width-at-half-maximum (FWHM) of the beam's field amplitude at the focal plane (z=0).
        """
        return self.__waist * 2 * np.sqrt(np.log(2))

    @full_width_at_half_maximum.setter
    def full_width_at_half_maximum(self, new_value: float):
        self.__waist = new_value / 2 / np.sqrt(np.log(2))

    @property
    def full_width_at_half_maximum_intensity(self) -> float:
        """
        The full-width-at-half-maximum (FWHM) of the beam's intensity at the focal plane (z=0).
        """
        return self.__waist * np.sqrt(2 * np.log(2))

    @full_width_at_half_maximum_intensity.setter
    def full_width_at_half_maximum_intensity(self, new_value: float):
        self.__waist = new_value / np.sqrt(2 * np.log(2))

    @property
    def rayleigh_range(self) -> float:
        """The Rayleigh range of the beam."""
        return self.wavenumber / 2.0 * self.waist ** 2

    @rayleigh_range.setter
    def rayleigh_range(self, new_value: float):
        """Set the Rayleigh range of the beam by adjusting the beam's waist."""
        self.waist = np.sqrt(new_value * 2.0 / self.wavenumber)

    @property
    def combined_order(self) -> int:
        """
        The combined order of a conventional Gaussian beam is 0.
        That of Hermite Gaussian beams is the sum of its horizontal and vertical indices, l and m.
        That of Laguerre Gaussian beams is the sum of twice its radial index, p, and the absolute value of its azimuthal index, l.
        """
        return 0

    @property
    def divergence(self) -> float:
        """The divergence angle of the beam in radians."""
        return np.arctan2(2.0, self.wavenumber * self.waist)

    @divergence.setter
    def divergence(self, new_value: float):
        """Set the angle of divergence by adjusting the beam waist."""
        self.waist = 2.0 / np.tan(new_value) / self.wavenumber

    def waist_at(self, distance_from_focal_plane: array_like) -> np.ndarray:
        """Compute the beam waist at one or more axial distances."""
        z = np.asarray(distance_from_focal_plane)
        return self.waist * np.sqrt(1.0 + (z / self.rayleigh_range) ** 2)

    def curvature_at(self, distance_from_focal_plane: array_like) -> np.ndarray:
        """Compute the wavefront's curvature at one or more axial distances."""
        z = np.asarray(distance_from_focal_plane)
        return z / (z ** 2 + self.rayleigh_range ** 2)

    def radius_of_curvature_at(self, distance_from_focal_plane: array_like) -> np.ndarray:
        """Compute the wavefront's radius of curvature at one or more axial distances."""
        return 1.0 / self.curvature_at(distance_from_focal_plane)

    def gouy_phase_at(self, distance_from_focal_plane: array_like) -> np.ndarray:
        """Compute the Gouy phase in radians at one or more axial distances."""
        z = np.asarray(distance_from_focal_plane)
        return (self.combined_order + 1) * np.arctan2(z, self.rayleigh_range)

    def complex_beam_parameter_at(self, distance_from_focal_plane: array_like) -> np.ndarray:
        """
        Compute the complex beam parameter at one or more axial distances.

        See: https://en.wikipedia.org/wiki/Gaussian_beam#Complex_beam_parameter
        """
        z = np.asarray(distance_from_focal_plane)
        return z + 1j * self.rayleigh_range

    def cylindrical(self, *args: Union[array_like, ft.Grid], **kwargs: array_like) -> np.ndarray:
        """
        Computes the beam's field values at the specified cylindrical coordinates.

        Coordinate vectors are broadcast and a Grid object can be used instead. Use the :obj:`optics.polar` package
        to convert between Cartesian and polar or cylindrical coordinates. This method is synonym to `polar`.
        When arguments are omitted, those on their **left** are assumed to 0.

        :param z: The longitudinal coordinate.
        :param rho: The radial coordinate.
        :param phi: The azimuthal coordinate in radians.

        :return: The field at the specified coordinates.
        """
        z, rho, phi = self._args_to_ranges(*args, **kwargs)

        rho2 = rho ** 2
        # field_normalization_factor = np.sqrt(2 / np.pi)
        # waist_inv_z = 1.0 / self.waist_at(z)
        # return field_normalization_factor * waist_inv_z \
        #     * np.exp(- (rho * waist_inv_z) ** 2
        #              - 1j * self.wavenumber * (z + self.curvature_at(z) / 2 * rho2)
        #              + 1j * self.gouy_phase_at(z)
        #              )
        q_z_inv = 1 / self.complex_beam_parameter_at(z)
        return (- 2 * np.pi) ** 0.5 * self.waist / self.wavelength \
            * q_z_inv * np.exp(-1j * self.wavenumber * z - 0.5j * self.wavenumber * rho2 * q_z_inv)

    def __str__(self) -> str:
        return f'G'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.waist}, wave_number={self.wavenumber}, propagation_axis={self.propagation_axis})'


class Hermite(Gaussian):
    """
    A class to represent Hermite-Gaussian beams, a.k.a. TEM modes.
    The amplitude of the beam is such that its total intensity flux is 1.
    """
    def __init__(self, m_index: int = 0, l_index: int = 0, *args, **kwargs):
        """
        Construct a Hermite-Gaussian beam with a cross-section intensity that integrates to 1. If another field-amplitude
        is required, just multiply it with a scalar (or even another beam).

        :param m_index: The index in the left-most transverse dimension, usually y, (default 0).
        :param l_index: The index in the right-most transverse dimension, usually x, (default 0).
        :param wavelength: (optional) The wavelength (default 1).
        :param wavenumber: (optional) The wavenumber.
        :param propagation_axis: The propagation axis (default 0).
        """
        super().__init__(*args, **kwargs)
        self.__indices = None
        indices = np.zeros(2, dtype=int)
        if m_index is not None:
            assert not isinstance(m_index, Sequence), 'Use *indices to specify both indices at once'
            indices[0] = m_index
        if l_index is not None:
            indices[1] = l_index
        self.indices = (m_index, l_index)

    @property
    def indices(self) -> np.ndarray:
        """The transverse indices of the beam, usually refered to as `m` and `l` in the order of the call arguments (usually y and x)."""
        return self.__indices

    @indices.setter
    def indices(self, new_value: array_like):
        new_value = np.ravel(new_value).astype(int)
        assert new_value.size == 2, 'Hermite Gaussian beams must have two indices, one for each transverse axis.'
        self.__indices = new_value

    @property
    def m_index(self) -> int:
        """The index for the left-most axis of the field, the `y`-axis."""
        return self.indices[-2]

    @m_index.setter
    def m_index(self, new_value: int):
        self.indices[-2] = new_value

    @property
    def l_index(self) -> int:
        """The index for the right-most axis of the field, the `x`-axis."""
        return self.indices[-1]

    @l_index.setter
    def l_index(self, new_value: int):
        self.indices[-1] = new_value

    @property
    def combined_order(self) -> int:
        """
        The combined order of a Hermite Gaussian beam is the sum of its horizontal and vertical indices, m and l.
        """
        return sum(np.abs(self.indices))

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

        # Calculate the transverse normalization factor
        waist_inv_z = 1.0 / self.waist_at(z)
        transverse_normalization_factor = np.sqrt(2) * waist_inv_z
        # Calculate the radial coordinates
        rho2 = y ** 2 + x ** 2
        rho = np.sqrt(rho2)
        indices = np.abs(self.indices)
        field_normalization_factor = np.sqrt(2.0 ** (1 - self.combined_order)) \
                                     / np.sqrt(np.pi * np.math.factorial(indices[-2]) * np.math.factorial(indices[-1]))  # * self.waist

        fld = field_normalization_factor * waist_inv_z \
            * np.exp(- (rho * waist_inv_z) ** 2
                     + 1j * (- self.wavenumber * (z + self.curvature_at(z) / 2 * rho2)
                             + self.gouy_phase_at(z)
                             )
                     )  # This is the regular Gaussian with an adapted Gouy phase
        if indices[-2] != 0:
            fld *= np.polynomial.hermite.Hermite.basis(indices[-2])(y * transverse_normalization_factor)
            fld *= np.sign(self.indices[-2])
        if indices[-1] != 0:
            fld *= np.polynomial.hermite.Hermite.basis(indices[-1])(x * transverse_normalization_factor)
            fld *= np.sign(self.indices[-1])
        return fld

    def __str__(self) -> str:
        return f'H{subsup.subscript(self.m_index)}{subsup.superscript(self.l_index)}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.indices}, waist={self.waist}, wave_number={self.wavenumber}, propagation_axis={self.propagation_axis})'


class Laguerre(Gaussian):
    """
    Represents Laguerre-Gaussian beams.
    The amplitude of the beam is such that its total intensity flux is 1.
    """
    def __init__(self, azimuthal_index: int = 0, radial_index: int = 0, *args, **kwargs):  # todo: Swap order of indices to be conform with cart2pol?
        """
        Creates a Laguerre-Gaussian function with unity intensity cross-section. If another field-amplitude is required,
        just multiply it with a scalar (or even another beam).

        :param azimuthal_index: The azimuthal index.
        :param radial_index: The radial index. When negative, the beam is mirrored around the axis. I.e. the l-index is inverted.
        :param waist: The beam waist.
        :param wavelength: (optional) The wavelength (default 1).
        :param wavenumber: (optional) The wavenumber.
        :param propagation_axis: The propagation axis (default 0).
        """
        super().__init__(*args, **kwargs)
        self.__indices = None
        self.indices = (azimuthal_index, radial_index)

    @property
    def indices(self) -> np.ndarray:
        """The azimuthal and radial index of the beam, usually refered to as `l` and `p`."""
        return self.__indices

    @indices.setter
    def indices(self, new_value: array_like):
        new_value = np.ravel(new_value).astype(int)
        assert new_value.size == 2, 'Laguerre Gaussian beams have two indices, an azimuthal one and a radial one.'
        self.__indices = new_value

    @property
    def azimuthal_index(self) -> int:
        """The azimuthal index of the beam, usually refered to as `l`."""
        return self.indices[0]

    @azimuthal_index.setter
    def azimuthal_index(self, new_value: int):
        self.indices[0] = new_value

    @property
    def radial_index(self) -> int:
        """The radial index of the beam, usually refered to as `p`."""
        return self.indices[1]

    @radial_index.setter
    def radial_index(self, new_value: int):
        self.indices[1] = new_value

    @property
    def combined_order(self) -> int:
        """
        The combined order of a Laguerre Gaussian beam is the sum of twice its radial index, p, and the absolute value of its azimuthal index, l.
        """
        return 2 * abs(self.radial_index) + abs(self.azimuthal_index)

    def cylindrical(self, *args: Union[array_like, ft.Grid], **kwargs: array_like) -> np.ndarray:
        """
        Computes the beam's field values at the specified cylindrical coordinates.

        Coordinate vectors are broadcast and a Grid object can be used instead. Use the :obj:`optics.polar` package
        to convert between Cartesian and polar or cylindrical coordinates. This method is synonym to `polar`.
        When arguments are omitted, those on their **left** are assumed to 0.

        :param z: The longitudinal coordinate.
        :param rho: The radial coordinate.
        :param phi: The azimuthal coordinate in radians.

        :return: The field at the specified coordinates.
        """
        z, rho, phi = self._args_to_ranges(*args, **kwargs)

        p = abs(self.radial_index)
        l = self.azimuthal_index
        if self.radial_index < 0:
            l = -l
        abs_l = abs(self.azimuthal_index)

        waist_inv_z = 1.0 / self.waist_at(z)
        field_normalization_factor = np.sqrt(2 * np.math.factorial(p) / (np.pi * np.math.factorial(p + abs_l)))
        return field_normalization_factor * waist_inv_z * (rho * np.sqrt(2) * waist_inv_z) ** abs_l \
            * scipy.special.genlaguerre(p, abs_l)(2 * (rho * waist_inv_z) ** 2) \
            * np.exp(- (rho * waist_inv_z)**2
                     + 1j * (- self.wavenumber * (z + self.curvature_at(z) / 2 * rho**2)
                             + self.gouy_phase_at(z)
                             - l * phi
                             )
                     )

    def __str__(self) -> str:
        return f'L{subsup.subscript(self.azimuthal_index)}{subsup.superscript(self.radial_index)}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.azimuthal_index}, {self.radial_index}, waist={self.waist}, wave_number={self.wavenumber}, propagation_axis={self.propagation_axis})'


class Hypergeometric(Laguerre):
    """
    Represents Hypergeometric-Gaussian beams.
    These can be an alternative to the Laguerre-Gaussian modes in a cylindrical coordinate system. In contrast to the
    Laguerre-Gaussian modes, the Hypergeometric modes are overdetermined.


    The amplitude of the beam is such that its total intensity flux is 1.
    """
    def __init__(self, *args, **kwargs):  # todo: Swap order of indices to be conform with cart2pol?
        """
        Creates a Laguerre-Gaussian function with unity intensity cross-section. If another field-amplitude is required,
        just multiply it with a scalar (or even another beam).

        :param azimuthal_index: The azimuthal index.
        :param radial_index: The radial index. When negative, the beam is mirrored around the axis. I.e. the l-index is inverted.
        :param waist: The beam waist.
        :param wavelength: (optional) The wavelength (default 1).
        :param wavenumber: (optional) The wavenumber.
        :param propagation_axis: The propagation axis (default 0).
        """
        super().__init__(*args, **kwargs)

    def cylindrical(self, *args: Union[array_like, ft.Grid], **kwargs: array_like) -> np.ndarray:
        """
        Computes the beam's field values at the specified cylindrical coordinates.

        Coordinate vectors are broadcast and a Grid object can be used instead. Use the :obj:`optics.polar` package
        to convert between Cartesian and polar or cylindrical coordinates. This method is synonym to `polar`.
        When arguments are omitted, those on their **left** are assumed to 0.

        :param z: The longitudinal coordinate.
        :param rho: The radial coordinate.
        :param phi: The azimuthal coordinate in radians.

        :return: The field at the specified coordinates.
        """
        z, rho, phi = self._args_to_ranges(*args, **kwargs)

        zn = z / self.rayleigh_range
        p = abs(self.radial_index)
        m = self.azimuthal_index
        if p < 0:
            m = -m
        abs_m = abs(self.azimuthal_index)
        gamma = scipy.special.gamma   # todo
        confluent_hypergeometric = scipy.special.hypergeometric(-p/2, abs_m+1)   # todo 1H1 confluent hypergeometric with args 1,b;x
        rho2 = rho ** 2
        normalization = np.sqrt((2**(p+abs_m+1)) / (np.pi * gamma(p + abs_m + 1))) \
                        * gamma(p/2 + abs_m + 1) / gamma(abs_m + 1) * (1j ** (abs_m + 1))
        longitudinal = (zn ** (p/2)) * ((zn + 1j) ** (- p/2 - abs_m - 1))
        radial = rho ** abs_m
        azimuthal = np.exp(1j * m * phi)
        rho2_over_zn_i = rho2 / (zn + 1j)
        radial_logitudinal = np.exp(-1j * rho2_over_zn_i) * confluent_hypergeometric(rho2_over_zn_i / zn)

        # todo: Test that at z==0, this simplifies to rho**(p+abs_m) np.exp(-\rho2 + 1j * m * phi)
        return normalization * longitudinal * radial * azimuthal * radial_logitudinal

    def __str__(self) -> str:
        return f'L{subsup.subscript(self.azimuthal_index)}{subsup.superscript(self.radial_index)}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.azimuthal_index}, {self.radial_index}, waist={self.waist}, wave_number={self.wavenumber}, propagation_axis={self.propagation_axis})'


class Ince(Laguerre):
    """
    Represents Ince-Gaussian beams, orthogonal modes in an elliptic coordinate system.

    The amplitude of the beam is such that its total intensity flux is 1.
    """
    def __init__(self, ellipticity: float, *args, **kwargs):  # todo: Swap order of indices to be conform with cart2pol?
        """
        Creates a Laguerre-Gaussian function with unity intensity cross-section. If another field-amplitude is required,
        just multiply it with a scalar (or even another beam).

        :param ellipticity: The ellipticity parameter of the elliptical coordinates. An ellipticity of 0 reproduces
        the Laguerre-Gaussian beams on a cylindrical coordinate system, while an ellipticity of ∞ reproduces the
        the Hermite-Gaussian beams on a Cartesian coordinate system.
        :param azimuthal_index: The azimuthal index.
        :param radial_index: The radial index. When negative, the beam is mirrored around the axis. I.e. the l-index is inverted.
        :param waist: The beam waist.
        :param wavelength: (optional) The wavelength (default 1).
        :param wavenumber: (optional) The wavenumber.
        :param propagation_axis: The propagation axis (default 0).
        """
        super().__init__(*args, **kwargs)
        self.__ellipticity = ellipticity

    @property
    def ellipticity(self) -> float:
        """The ellipticity of the Ince Gaussian beam."""
        return self.__ellipticity

    @ellipticity.setter
    def ellipticity(self, new_value: float):
        self.__ellipticity = new_value

    def elliptical(self, z, eta, xi):
        """
        Calculates the field that the specified elliptical coordinates.

        :param z: The longitudinal coordinate.
        :param eta:
        :param xi:
        :return: The field at the specified elliptical coordinates.
        """
        p = abs(self.radial_index)
        m = self.azimuthal_index
        if self.radial_index < 0:
            m = -m
        c = scipy.special.Ince(order=p, degree=m)   # todo

        rh = np.sqrt(self.ellipticity / 2) * self.waist_at(z)
        x = rh * np.cosh(xi) * np.cos(eta)
        y = rh * np.sinh(xi) * np.sin(eta)
        rho2 = x**2 + y**2
        return self.waist / self.waist_at(z) * c(1j * xi, self.ellipticity) * c(eta, self.ellipticity) \
            * np.exp(-1j * self.wavenumber * rho2 / (2 * qz) - (p+1)*scipy.special.zeta(z))

    # def cylindrical(self, z: Union[array_like, ft.Grid], rho: Optional[array_like] = None, phi: Optional[array_like] = None) -> np.ndarray:
    # """
    #     Computes the beam's field values at the specified cylindrical coordinates.
    #
    #     Coordinate vectors are broadcast and a Grid object can be used instead. Use the :obj:`optics.polar` package
    #     to convert between Cartesian and polar or cylindrical coordinates. This method is synonym to `polar`.
    #     When arguments are omitted, those on their **left** are assumed to 0.
    #
    #     :param z: The longitudinal coordinate.
    #     :param rho: The radial coordinate.
    #     :param phi: The azimuthal coordinate in radians.
    #
    #     :return: The field at the specified coordinates.
    #     """
    #     # eta = todo
    #     return self.elliptical(z, eta, xi)

    def __str__(self) -> str:
        return f'I({self.ellipticity}){subsup.subscript(self.azimuthal_index)}{subsup.superscript(self.radial_index)}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.ellipticity}, {self.azimuthal_index}, {self.radial_index}, waist={self.waist}, wave_number={self.wavenumber}, propagation_axis={self.propagation_axis})'


if __name__ == '__main__':
    from optics.experimental import log

    grid = ft.Grid(extent=[20, 10, 10], step=0.1)
    beam = Gaussian(waist=1, wavelength=0.5)
    # beam = Laguerre(0, 1)
    # beam = Laguerre(3, 2, waist=1)
    # beam = Hermite(2)
    # beam = Hermite(2, -1)
    # beam = Laguerre(azimuthal_index=3, radial_index=1)
    # beam = Hermite(0) + Hermite(2)
    # beam = Hermite(0, 4) + Laguerre(1, 3) / 2
    # beam = abs(Hermite(4) + Hermite(0) + Laguerre(2) + 2j * Gaussian()) ** 2

    # from special_beams import Airy
    # beam = Laguerre(3, 3) * 2 + 1 * Airy([1, 2]) + Airy(1) + 1j * Airy(2) - Airy(3) -1j * Airy(4) - 1 * Airy(5) + 0 * Airy(7) - 0 * Airy(8)
    # beam = - Airy(8) * 3

    beam = 3 * 3 * Gaussian() * Gaussian() * 2 * Gaussian() * 3.14
    beam = 2 + Gaussian() + 3 + Gaussian() + 4
    print(f"{beam}: {repr(beam)}")

    exit()

    fld = beam(grid)

    log.info(f'{beam} has an energy flux of {np.linalg.norm(fld[grid.shape[0] // 2]) ** 2 * np.prod(grid.step[1:])}')
    log.info(repr(beam))

    import matplotlib.pyplot as plt
    from optics.utils.display import complex2rgb, grid2extent

    fig, axs = plt.subplots(2, 2)
    axs = axs.ravel()
    axs[0].imshow(complex2rgb(fld[grid.shape[0] // 2, :, :], 4), extent=grid2extent(grid.project(axes_to_remove=0)))
    axs[0].set(xlabel='x', ylabel='y')
    axs[2].imshow(complex2rgb(fld[:, grid.shape[1] // 2, :], 4), extent=grid2extent(grid.project(axes_to_remove=1)))
    axs[2].set(xlabel='x', ylabel='z')
    axs[1].imshow(complex2rgb(fld[:, :, grid.shape[2] // 2], 4), extent=grid2extent(grid.project(axes_to_remove=2)))
    axs[1].set(xlabel='y', ylabel='z')
    section = np.abs(fld[grid.shape[-3] // 2, :, grid.shape[-1] // 2]) ** 2
    # axs[-1].plot(grid[-2].ravel(), section)
    # axs[-1].set(xlabel='y', ylabel='I')
    # axs[-1].plot(grid[0].ravel(), beam.gouy_phase_at(grid[0].ravel()))
    # axs[-1].set(xlabel='z', ylabel='Gouy phase [rad]')
    axs[-1].plot(grid[0].ravel(), beam(grid[0], 0, 0).ravel().real)
    axs[-1].plot(grid[0].ravel(), beam(grid[0], 0, 0).ravel().imag)
    axs[-1].set(xlabel='z', ylabel='E')
    axs[0].set_title(beam)

    plt.show()




