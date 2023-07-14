import numpy as np
from typing import Union, Optional, Callable, Sequence, Iterable, Generator, Type
from numbers import Complex, Number
import scipy.integrate
import scipy.special
import scipy.constants as const

from optics import log
from optics.utils.ft import Grid
from optics.instruments.objective import Objective
from optics.calc.beam import Beam, BeamSection
from optics.utils.polar import cart2pol

__all__ = ['PSF', 'RichardsWolfPSF', 'BesselPSF']


class PSF(Callable):
    """A class to represent generic vectorial point spread functions."""
    def __init__(self, objective: Optional[Objective] = None, pupil_grid: Optional[Grid] = None,
                 pupil_function: Union[None, float, Sequence, np.array,
                                       Callable[[Union[float, Sequence, np.ndarray], Union[float, Sequence, np.ndarray]],
                                                Union[float, Sequence, np.ndarray]]] = 1.0,
                 vacuum_wavenumber: Optional[float] = 1.0, vacuum_wavelength: Optional[float] = None,
                 pupil_intensity_correction_exponent: int = 1, dtype: Type[Number] = np.complex128
                 ):
        """
        Creates a Point-Spread Function object.

        :param objective: The Objective object to use.
        :param pupil_grid: (optional) The calculation grid to use on the pupil in coordinates that are centered in the
            pupil and normalized to its radius. The grid's extent should be at least 2x2 to avoid undersampling.
        :param pupil_function: The optional pupil function in two Cartesian variables, centered in the pupil, and normalized to the pupil radius.
            This can be a scalar or 3-vector function that takes vector arguments. E.g. `lambda u, v: exp(2j * np.pi * (u**2 + v**2))  # for defocus`
        :param vacuum_wavenumber: The vacuum wavenumber in radians per meter.
        :param vacuum_wavelength: The vacuum wavelength in meters if the wavenumber is not specified.
        :param pupil_intensity_correction_exponent: The exponent of the cosine variation of the pupil intensity correction.
            See the :py:obj:`pupil_intensity_correction_exponent` property for more details. Default: detection with
            infinity-corrected objective that obeys the `Abbe sine Condition <https://en.wikipedia.org/wiki/Abbe_sine_condition>`_.
        :param dtype: The dtype of the calculations.
        """
        if objective is None:
            objective = Objective(magnification=1.0, numerical_aperture=1.0, refractive_index=1.0)

        if vacuum_wavelength is not None:
            vacuum_wavenumber = 2 * np.pi / vacuum_wavelength

        self.__dtype = dtype
        self.__pupil_intensity_correction_exponent: int = 0
        self.pupil_intensity_correction_exponent = pupil_intensity_correction_exponent

        if not isinstance(pupil_function, Callable):
            value = np.asarray(pupil_function)
            if value.size > 2:
                raise ValueError('Argument pupil_function should be Callable, a scalar, or a 2-vector.')
            while value.ndim < 3:
                value = value[np.newaxis, ...]  # Add singleton dimensions
            pupil_array_function = lambda nu_y, nu_x: value  # normalized coordinates
        else:
            def pupil_array_function(nu_y, nu_x):
                pupil_array = np.asarray(pupil_function(nu_y, nu_x))
                while pupil_array.ndim < 3:
                    pupil_array = pupil_array[np.newaxis, ...]  # Add singleton dimensions
                return pupil_array

        test_pupil_array = pupil_array_function(np.zeros((1, 1)), np.zeros((1, 1)))
        self.__vectorial = test_pupil_array.ndim >= 3 and test_pupil_array.shape[-3] > 1

        def field_ft_function(ky, kx):
            """Returns a scalar or a polarization vector [Ey, Ex] for each (ky, kx) combination."""
            k_n = self.__vacuum_wavenumber * self.__objective.refractive_index
            # Find the angles theta and phi that correspond to each wavevector
            sin_y, sin_x = ky / k_n, kx / k_n  # The sine of the ray ray angle are the normalized coordinates to the wavenumber in the medium
            sin_theta_2 = sin_y ** 2 + sin_x ** 2  # theta is the angle with the optical axis (z, axis -3)
            cos_theta = np.sqrt(np.maximum(1.0 - sin_theta_2, 0.0))

            # Calculate the pupil function in normalized coordinates using pupil_array_functions.
            objective_sin_limit = np.minimum(self.__objective.numerical_aperture / self.__objective.refractive_index, 1.0)  # This should always be less than 1
            field_ft = pupil_array_function(sin_y / objective_sin_limit, sin_x / objective_sin_limit)[..., np.newaxis, :, :].astype(self.dtype)

            # Optionally apply the Sine or Herschel condition for imaging or focussing. See `pupil_intensity_correction_exponent` for more details.
            inside_aperture = sin_theta_2 < objective_sin_limit ** 2
            # Block the field outside the aperture and avoid division by 0.
            return field_ft * inside_aperture * (cos_theta + (1 - inside_aperture)) ** (-self.pupil_intensity_correction_exponent / 2.0)  # Fields vary with the sqrt of the intensity

        def vector_field_ft_function(ky, kx):
            """Returns a polarization vector [Ez, Ey, Ex] for each (ky, kx) combination."""
            field_ft = field_ft_function(ky, kx)  # should return a vector for each (ky, kx) combination containing [Ey, Ex] at the pupil.

            # Convert to spherical pupil coordinates
            # Find the angles theta and phi that correspond to each wavevector
            k_n = self.__vacuum_wavenumber * self.__objective.refractive_index
            k_dir_y, k_dir_x = ky / k_n, kx / k_n
            sin_theta_2 = k_dir_y ** 2 + k_dir_x ** 2  # theta is the angle with the optical axis (z, axis -3)
            sin_theta = np.sqrt(sin_theta_2)
            cos_theta = np.sqrt(np.maximum(1.0 - sin_theta_2, 0.0))

            # phi is angle from the x-axis towards the y-axis
            on_axis = np.isclose(sin_theta, 0.0)
            inv_sin_theta = 1.0 / (sin_theta + on_axis)
            sin_phi = k_dir_y * inv_sin_theta  # phi is angle from the x-axis towards the y-axis, or 0 if on-axis propagation
            cos_phi = k_dir_x * inv_sin_theta + on_axis  # No rotation for the wavevector along the optical axis
            # Rotate the electric field vectors by -phi
            Ea = cos_phi * field_ft[..., -2, :, :, :] - sin_phi * field_ft[..., -1, :, :, :]
            Er = sin_phi * field_ft[..., -2, :, :, :] + cos_phi * field_ft[..., -1, :, :, :]
            # Rotate polarization by theta
            Ez = -sin_theta * Er
            Er *= cos_theta
            # Convert to 3D Cartesian coordinates x, y, z
            Ey = cos_phi * Ea + sin_phi * Er
            Ex = -sin_phi * Ea + cos_phi * Er
            return np.concatenate((
                Ez[..., np.newaxis, :, :, :],
                Ey[..., np.newaxis, :, :, :],
                Ex[..., np.newaxis, :, :, :]), axis=-4)

        self.__objective = objective
        self.__vacuum_wavenumber = vacuum_wavenumber
        if pupil_grid is not None:
            self.__pupil_grid = pupil_grid  # todo: Unused at the moment
        else:
            self.__pupil_grid = Grid([128, 128], extent=2)
        f_grid = self.__pupil_grid * (self.__objective.numerical_aperture * self.__vacuum_wavenumber / (2 * np.pi))
        self.__grid = Grid([1, *f_grid.shape], step=[1, *(1/f_grid.extent)])
        if self.__grid.ndim < 3:
            self.__grid = Grid(1) @ self.__grid
        self.__field_ft_function = vector_field_ft_function if self.vectorial else field_ft_function

        self.__propagation_position_index = 0

    @property
    def vectorial(self) -> bool:
        """True when polarized calculations are requested, False for scalar calculations."""
        return self.__vectorial

    @property
    def shape(self) -> np.ndarray:
        """The default calculation shape"""
        return self.__grid.shape

    @property
    def dtype(self):
        """The dtype of the calculation results."""
        return self.__dtype

    @property
    def pupil_intensity_correction_exponent(self) -> int:
        r"""
        The exponent of the cosine variation of the pupil intensity correction. Use:

        *  `2` when imaging with an objective that obeys the `Herschel's Condition <https://doi.org/10.1088/0959-5309/58/1/309>`_

        *  `1` when imaging with an objective that obeys the `Abbe sine Condition <https://en.wikipedia.org/wiki/Abbe_sine_condition>`_

        *  `0` when appropriate amplitude corrections have already been incorporated into the supplied pupil function.

        * `-1` when illuminating with an objective that obeys the `Abbe sine Condition <https://en.wikipedia.org/wiki/Abbe_sine_condition>`_

        * `-2` when illuminating with an objective that obeys the `Herschel's Condition <https://doi.org/10.1088/0959-5309/58/1/309>`_

        The most common and default situation is `1`, imaging with an infinity-corrected objective that obeys the
        `Abbe sine Condition <https://en.wikipedia.org/wiki/Abbe_sine_condition>`_. This effectively increases the
        intensity as :math:`1/[\cos(\theta)]^{1}` towards the edge of the pupil, to compensate for the fact that those pupil
        coordinates capture a greater angle of the emitted light by an isotropically emitting point source.

        When imaging a point-source that emits equal intensity in all directions, that light is captured by the
        objective lens and re-imaged onto a detector using a tube lens. Assuming that the detection is an ideal system,
        the field at the back aperture pupil plane is Fourier transformed to the detector plane. When the intensity at
        the pupil is uniform, this results in a perfect Airy-disk point-spread function (scalar), or the vectorial
        equivalent. However, when detecting a point-source in the sample, the field at the back aperture is generally
        not uniform. If it is assumed that objective is loss-less, i.e. it captures all wave-vectors within the
        detection cone of the objective, with half angle :math:`\alpha` so that :math:`\sin(\alpha) = NA/n`, then the
        intensity of at the back aperture pupil plane will vary radially with angle :math:`\theta < \alpha` of the
        corresponding wave vector and the optical axis. Objectives that obey the `Abbe sine Condition <https://en.wikipedia.org/wiki/Abbe_sine_condition>`_
        project the Ewald (k-space) sphere axially so that intensity of a point source in the sample leads to a pupil
        intensity variation :math:`[\cos(\theta)]^{-1}`. Objectives that obey `Herschel's Condition <https://doi.org/10.1088/0959-5309/58/1/309>`_
        lead to an intensity variation :math:`[\cos(\theta)]^{-2}`. Conversely, when the objective is used to focus
        light into the sample, the intensity will vary as :math:`\cos(\theta)` and :math:`[\cos(\theta)]^2`, for the
        `Abbe sine Condition <https://en.wikipedia.org/wiki/Abbe_sine_condition>`_ and `Herschel's Condition <https://doi.org/10.1088/0959-5309/58/1/309>`_,
        respectively.

        References:

        - Colin J R Sheppard "The optics of microscopy" J. Opt. A: Pure Appl. Opt. 9(6) (2007) `doi: 10.1088/1464-4258/9/6/S01 <https://doi.org/10.1088/1464-4258/9/6/S01>`_. https://iopscience.iop.org/article/10.1088/1464-4258/9/6/S01

        - H. H. Hopkins "The Airy disc formula for systems of high relative aperture" Proc. Phys. Soc. 55 116–28 (1943). `doi: 10.1088/0959-5309/55/2/305 <https://doi.org/10.1088/0959-5309/55/2/305>`_. https://iopscience.iop.org/article/10.1088/0959-5309/55/2/305

        """
        return self.__pupil_intensity_correction_exponent

    @pupil_intensity_correction_exponent.setter
    def pupil_intensity_correction_exponent(self, new_exponent: int):
        """Set the exponent of the cosine variation of the pupil intensity correction."""
        self.__pupil_intensity_correction_exponent = new_exponent

    def beam(self, *ranges) -> Beam:
        """
        Get the beam exiting the objective as a beam.Beam object.

        :param ranges: The sample points in the z, y and x-axes. These must form a uniformly spaced grid. When less than
            3 dimensions are specified, those are interpreted as the right-most dimensions, while default values are
            taken for the left-most axes.
        :return: Returns a beam.Beam object for the requested grid points.
            The field can be extracted as an ndarray by calling field() on the returned value.
            The returned ndarray of which the final four dimensions are polarization, z, y, and x.
        """
        if len(ranges) == 1 and isinstance(ranges[0], Grid):
            ranges = ranges[0]
            log.warning('PSF() arguments should be ranges, or *grid, not a single Grid object! '
                        + 'Converting single Grid object to ranges.')
        all_ranges = [*self.__grid.flat]
        for _, rng in enumerate(reversed(ranges)):
            all_ranges[-_-1] = rng
        grid = Grid.from_ranges(*all_ranges)
        beam = Beam(grid=grid, propagation_axis=-3, vacuum_wavenumber=self.__vacuum_wavenumber,
                    background_refractive_index=self.__objective.refractive_index,
                    field_ft=self.__field_ft_function)
        return beam

    def __iter__(self, *ranges, infinite: bool = True) -> Generator[BeamSection, None, None]:
        """
        Iterator returning beam.BeamSections of the point-spread function.
        For each z-section, each BeamSection represents a y-x-slice.

        :param infinite: (default False) When set to True, this iterator will continue generating BeamSections in the
            background medium.

        :return: A Generator object producing BeamSections.
        """
        return self.beam(*ranges).__iter__(infinite=infinite)

    def field(self, *ranges) -> np.ndarray:
        """
        Calculate the field values at specific coordinates.

        :param ranges: The sample points in the z, y and x-axes. These must be uniformly spaced grids.

        :return: An ndarray of which the final four dimensions are the polarization, z, y, and x.
        """
        return np.asarray(self.beam(*ranges))

    def __call__(self, *ranges) -> np.ndarray:
        """
        Calculate the field values at specific coordinates.

        :param ranges: The sample points in the z, y and x-axes. These must be uniformly spaced grids.
            Use PSF(*grid) to distribute the Grid's ranges as arguments.

        :return: An ndarray of which the final four dimensions are the polarization, z, y, and x.
        """
        if len(ranges) == 1 and isinstance(ranges[0], Grid):
            ranges = ranges[0]
            log.warning('PSF() arguments should be ranges, or *grid, not a single Grid object! '
                        + 'Converting single Grid object to ranges.')
        return self.field(*ranges)

    def intensity(self, *ranges, polarization_axes=slice(None, None, None), keepdims: bool = False) -> np.ndarray:
        """
        Calculate the intensity values at the specific coordinates.

        :param ranges: The sample points in the z, y and x-axes. These must be uniformly spaced grids.
        :param polarization_axes: The polarization axes to include. Default: all.
        :param keepdims: (optional) When True, the polarization axes (-4) is kept as a singleton dimension.
            Default: False.

        :return: An ndarray of which the final three dimensions are the coordinates z, y, and x.
        """
        return np.sum(np.abs(self.field(*ranges)[..., polarization_axes, :, :, :])**2, axis=-4, keepdims=keepdims)

    # def __getitem__(self, key):
    #     if isinstance(key, Grid):
    #         index_grid = key
    #     elif isinstance(key, tuple):
    #         index_grid = Grid.from_ranges(*key)
    #     else:
    #         raise ValueError(f"Unknown key type: {type(key)}")
    #
    #     # Make sure that it is a 3-d Grid, add singleton dimensions as necessary
    #     while index_grid.ndim < 3:
    #         index_grid = Grid(1) @ index_grid
    #
    #     selected_grid = Grid(shape=self.__grid.shape, step=self.__grid.step, center=self.__grid.center)
    #
    #     return self(*selected_grid)


class RichardsWolfPSF(PSF):
    """
    The Richards and Wolf PSF.

    .. todo:: At the moment this implementation only works with a constant pupil function.

    """
    def __init__(self, objective: Optional[Objective] = None, pupil_grid: Optional[Grid] = None,
                 pupil_function: Union[None, float, Sequence, np.ndarray,
                                       Callable[[Union[float, Sequence, np.ndarray], Union[float, Sequence, np.ndarray]],
                                                Union[float, Sequence, np.ndarray]]]=1.0,
                 vacuum_wavenumber: Optional[float] = 1.0, vacuum_wavelength: Optional[float]=None,
                 ):
        if vacuum_wavelength is not None:
            vacuum_wavenumber = 2 * np.pi / vacuum_wavelength

        super().__init__(objective=objective, pupil_grid=pupil_grid, vacuum_wavenumber=vacuum_wavenumber,
                         pupil_function=pupil_function)

        self.__wavenumber = vacuum_wavenumber
        self.__objective = objective
        self.__pupil_grid = pupil_grid
        self.__pupil_function = pupil_function

    def field(self, *ranges) -> np.ndarray:
        if len(ranges) == 1 and isinstance(ranges[0], Grid):
            ranges = ranges[0]
            log.warning('PSF() arguments should be ranges, or *grid, not a single Grid object! '
                        + 'Converting single Grid object to ranges.')
        grid = Grid.from_ranges(*ranges).as_flat

        sin_theta = self.__objective.numerical_aperture / self.__objective.refractive_index
        dx = grid.step[-1] if grid.shape[-1] > 1 else 1.0
        dy = grid.step[-2] if grid.shape[-2] > 1 else dx

        if isinstance(self.__pupil_function, Complex):
            # Just scale the diffraction limited point spread function
            fl0 = self.__pupil_function / np.sqrt(np.pi*(sin_theta**2) / (dx*dy))
            field_at_point_function = lambda z, y, x: self.__diff_lim_at_point(self.__wavenumber, sin_theta, fl0, z, y, x)
        elif callable(self.__pupil_function):
            raise AttributeError('Only constant pupil functions are working at the moment. Please specify a number instead of a callable function.')
            # f times l_0 as in paper
            fl0 = lambda sx, sy: \
                np.broadcast_to(self.__pupil_function(sx / sin_theta, sy / sin_theta) / np.sqrt(np.pi * (sin_theta**2) / (dx*dy)),
                                [1, *np.broadcast_shapes(sx.shape, sy.shape)])
            field_at_point_function = lambda z, y, x: self.__generic_at_point(self.__wavenumber, sin_theta, fl0, z, y, x)
        else:
            raise TypeError(f'Unknown pupil function type: {type(self.__pupil_function)}.')

        # Iterate over all positions in image space
        field = np.zeros((3, *grid.shape), dtype=complex)
        for z_idx, z in enumerate(grid[-3]):
            for y_idx, y in enumerate(grid[-2]):
                for x_idx, x in enumerate(grid[-1]):
                    field[:, z_idx, y_idx, x_idx] = field_at_point_function(z, y, x)

        return field

    def __generic_at_point(self, k, sin_theta, fl0, z, y, x):
        def A(sy, sx, axis):
            sy = np.atleast_2d(sy)
            sx = np.atleast_2d(sx)
            ST = np.sqrt(sy**2 + sx**2)
            CT = np.sqrt(1 - ST**2)
            SP = sy / (ST + (ST == 0))
            CP = sx / (ST + (ST == 0))
            # todo: implement both Ex and Ey components!
            result = np.repeat(fl0(sy, sx)[0, :, :] * np.sqrt(CT), repeats=3, axis=0) \
                     * np.stack((CT + SP**2 * (1 - CT), (CT-1) * CP * SP, -ST * CP))
            return result[:, :, axis]

        integral = np.zeros(3)
        for axis in range(3):
            def integrand(sz, sy, sx):
                return A(sy, sx, axis) * (1 / sz) * np.exp(1j * k * (sx * x + sy * y + sz * z))
            integral[axis], abserr = scipy.integrate.dblquad(
                lambda sx, sy: integrand(np.sqrt(1 - (sx**2 + sy**2)), sy, sx),
                -sin_theta, sin_theta,
                lambda x: -sin_theta * np.sqrt(1 - (x / sin_theta) ** 2),
                lambda x: sin_theta * np.sqrt(1 - (x / sin_theta) ** 2),
                epsabs=1e-6)  # maxiter=5000

        return -1j * k / (2*np.pi) * integral

    def __diff_lim_at_point(self, k, sin_theta, fl0, z, y, x):
        rP = np.sqrt(x**2 + y**2 + z**2)
        tP = np.arctan2(np.sqrt(x**2 + y**2), z)
        phiP = np.arctan2(y, x)
        
        def integrand(t):
            result = [
                np.sqrt(np.cos(t)) * np.sin(t) * (1 + np.cos(t)) * scipy.special.jv(0, k * rP * np.sin(t) * np.sin(tP))
                * np.exp(1j * k * rP * np.cos(t) * np.cos(tP)),
                np.sqrt(np.cos(t)) * (np.sin(t)**2) * scipy.special.jv(1, k * rP * np.sin(t) * np.sin(tP))
                * np.exp(1j * k * rP * np.cos(t) * np.cos(tP)),
                np.sqrt(np.cos(t)) * np.sin(t) * (1 - np.cos(t)) * scipy.special.jv(2, k * rP * np.sin(t) * np.sin(tP))
                * np.exp(1j * k * rP * np.cos(t) * np.cos(tP))
            ]
            return np.asarray(result)

        I, int_error = scipy.integrate.quad_vec(integrand, 0.0, np.arcsin(sin_theta))
        # I = [scipy.integrate.quadrature(lambda t: integrand(t)[_, :, :], 0, np.arcsin(sin_theta)) for _ in range(3)]

        A = k * fl0 / 2
        E = 1j * A * np.array([(I[0] + I[2] * np.cos(2 * phiP)), I[2] * np.sin(2 * phiP), 2 * I[1] * np.cos(phiP)])
    
        return E


class BesselPSF(PSF):
    def __init__(self, objective: Objective = None, pupil_grid: Grid = None,
                 l_number: int = 0,
                 vacuum_wavenumber: Union[float, None] = 1.0, vacuum_wavelength: Union[float, None] = None,
                 ):
        """
        The analytic Bessel-beam point-spread function.

        .. todo:: This is work in progress!

        :param objective:
        :param pupil_grid:
        :param l_number:
        :param vacuum_wavenumber:
        :param vacuum_wavelength:
        """
        if vacuum_wavelength is not None:
            vacuum_wavenumber = 2 * np.pi / vacuum_wavelength
        self.__wavenumber = vacuum_wavenumber
        super().__init__(objective=objective, pupil_grid=pupil_grid, vacuum_wavenumber=vacuum_wavenumber)

        self.__objective = objective
        self.__pupil_grid = pupil_grid
        self.__l_number = l_number

    def field(self, *ranges) -> np.ndarray:
        grid = Grid(1) @ self.__pupil_grid  # todo: actually handle the key

        l_number = self.__l_number

        sin_theta = self.__objective.numerical_aperture / self.__objective.refractive_index
        sample_n = self.__objective.refractive_index
        pol = [1, 0]  # Polarization

        c = const.c

        rho, phi = cart2pol(grid[-2], grid[-1])  # Only the phase_vector of the field changes in the propagation direction.
        rho = np.maximum(rho, np.finfo(np.float).eps)
        k = self.__wavenumber * sample_n
        gamma = np.arcsin(sin_theta / sample_n)

        erho = (0.5j * np.exp(1j * (l_number * phi + grid[0] * k * np.cos(gamma))) * (2 * pol[0] * c * l_number * scipy.special.jv(l_number, rho * k * np.sin(gamma)) + pol[1] * rho * k * c * (scipy.special.jv(-1 + l_number,(rho * k * c * np.sin(gamma))/c) - scipy.special.jv(1 + l_number, rho * k * np.sin(gamma))) * np.cos(gamma) * np.sin(gamma)))/(c * rho)
        ephi = -(np.exp(1j * (l_number * phi + grid[0] * k * np.cos(gamma))) * (2 * pol[1] * c * l_number * scipy.special.jv(l_number, rho * k * np.sin(gamma)) * np.cos(gamma) + pol[0] * rho * k * c * (scipy.special.jv(-1 + l_number,(rho * k * c * np.sin(gamma))/c) - scipy.special.jv(1 + l_number, rho * k * np.sin(gamma))) * np.sin(gamma)))/(2*c * rho)
        ez = (pol[1] * np.exp(1j * (l_number * phi + grid[0] * k * np.cos(gamma))) * k * scipy.special.jv(l_number, rho * k * np.sin(gamma)) * np.sin(gamma)**2)
        # hrho = (np.exp((1j * (c * l_number * phi + grid[0] * k * c * np.cos(gamma)))/c) * (2 * c * l_number * scipy.special.jv(l_number,(rho * k * c * np.sin(gamma))/c) * (pol[1] - pol[0] * np.cos(gamma)) + pol[0] * rho * k * c * scipy.special.jv(-1 + l_number,(rho * k * c * np.sin(gamma))/c) * np.sin(2*gamma)))/(2*c * rho)
        # hphi = ((-1j) * np.exp((1j * (c * l_number * phi + grid[0] * k * c * np.cos(gamma)))/c) * (c * l_number * scipy.special.jv(l_number,(rho * k * c * np.sin(gamma))/c) * (pol[1] - pol[0] * np.cos(gamma)) - pol[1] * rho * k * c * scipy.special.jv(-1 + l_number,(rho * k * c * np.sin(gamma))/c) * np.sin(gamma)))/(c * rho)
        # hz = ((-1j) * pol[0] * np.exp((1j * (c * l_number * phi + grid[0] * k * c * np.cos(gamma)))/c) * k * c * scipy.special.jv(l_number,(rho * k * c * np.sin(gamma))/c) * np.sin(gamma)**2)/c

        efx = erho * np.cos(phi) - ephi * np.sin(phi)
        efy = erho * np.sin(phi) + ephi * np.cos(phi)
        efz = ez
        # hfx = hrho * np.cos(phi) - hphi * np.sin(phi)
        # hfy = hrho * np.sin(phi) + hphi * np.cos(phi)
        # hfz = hz

        psf = np.abs(efx)**2 + np.abs(efy)**2 + np.abs(efz)**2
        psf = np.repeat(psf, axis=0, repeats=grid.shape[0])  # Extend the beam along the optical axis.

        psf /= np.sum(psf[0, :, :], axis=(-2, -1))  #Normalize cross section intensity

        return psf


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from optics.utils.display import complex2rgb, grid2extent

    wavelength = 500e-9
    obj = Objective(40, 0.80, refractive_index=1.00)

    def pupil_function(u: Union[float, Sequence, np.array], v: Union[float, Sequence, np.array]):
        r = np.sqrt((u-0.0)**2 + v**2)
        pupil = np.logical_and(-0.9 < r, r <= 1)
        # pupil = np.exp(-0.5 * (r / 0.1)**2)

        # return pupil
        pupil = pupil[np.newaxis, :, :]

        return np.concatenate((pupil, 1j*pupil))

    # Determine sampling grid
    nb_slices = 128
    psf_grid_step = np.array([wavelength / 8, *np.ones(2) * wavelength / (4 * obj.numerical_aperture)])
    z_step = psf_grid_step[0]
    slice_step = psf_grid_step[1:]
    max_defocus = int(nb_slices / 2) * z_step
    geometric_extent = 2 * max_defocus / np.sqrt((obj.refractive_index / obj.numerical_aperture)**2 - 1.0)
    slice_shape = np.maximum(np.ceil(geometric_extent / slice_step), 128)
    psf_grid = Grid(shape=[nb_slices, *slice_shape], step=psf_grid_step)
    log.info(f"estimated_extent = {psf_grid.extent*1e6}um, grid.shape = {psf_grid.shape}")

    psf = PSF(obj, pupil_function=pupil_function, vacuum_wavelength=wavelength)
    psf_array = psf(*psf_grid)

    #
    # Display
    #
    fig, axs = plt.subplots(1, psf_array.shape[0], figsize=(5*psf_array.shape[0], 4))
    if not isinstance(axs, Iterable):
        axs = [axs]
    for slice_idx, ax in enumerate(axs):
        ax.imshow(complex2rgb(np.squeeze(psf_array[slice_idx, :, :, int(psf_array.shape[-1]/2)]), normalization=10),
                  extent=grid2extent(psf_grid[0]*1e6, psf_grid[1]*1e6))
        ax.set(xlabel='x [$\mu$m]', ylabel='z [$\mu$m]', title='$E_'+'xyz'[slice_idx]+'$')

    plt.show()
