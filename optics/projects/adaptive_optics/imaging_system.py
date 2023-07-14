import numpy as np
import logging
import scipy.special as sp
from typing import Optional
from optics.utils import ft, reference_object
from optics.calc import zernike


log = logging.getLogger(__name__)

# from optics.instruments import objective
# from optics.calc.psf import PSF


class ImagingSystem:
    def __init__(self, grid_2d: ft.Grid, numerical_aperture: float = 0.021, aperture_diameter: float = 0.0105, wavelength: float = 532e-9, simulate_noise: bool = True,
                 background_noise_level: float = 0.01, max_number_of_photons: int = 1000, bits_per_pixel: int = 8, point_source: bool = False):
        """
        TODO: Write this out...

        :param grid_2d:
        :param numerical_aperture:
        :param aperture_diameter:
        :param wavelength:
        :param simulate_noise:
        :param background_noise_level:
        :param max_number_of_photons:
        :param bits_per_pixel:
        :param point_source:
        """
        self.datatype = np.float32

        self.__grid = grid_2d.immutable
        self.numerical_aperture = numerical_aperture
        self.aperture_diameter = aperture_diameter
        self.focal_length = self.aperture_diameter / (2 * self.numerical_aperture)
        self.wavelength = wavelength
        self.simulate_noise = simulate_noise
        self.background_noise_level = background_noise_level
        self.max_number_of_photons = max_number_of_photons
        self.bits_per_pixel = bits_per_pixel
        self.wave_number = 2*np.pi/self.wavelength
        self.lens_radius = self.aperture_diameter / 2
        self.__rng = np.random.Generator(np.random.PCG64(seed=2))  # For the noise generation

        # log.info('Defining the optical system...')
        # obj = objective.InfinityCorrected(40, 0.80, refractive_index=1.33)

        self.__object_ft = None
        # original object: usaf1951 or other intensity image
        if point_source:  # TODO: Fix error before using this!
            self.object = np.zeros(self.grid.shape[-2:])
            self.object[self.object.shape[0]//4, self.object.shape[1]//4] = 1.0
        else:
            log.info('Loading reference...')
            self.object = np.asarray(reference_object.usaf1951(self.grid.shape[-2:], scale=1.0)) / 255.0  # Maximum value = 1.0
            # self.object = np.asarray(reference_object.boat(self.grid.shape)) / 255.0  # Maximum value = 1.0

        # create a aperture
        self.__pupil_grid = ft.Grid(self.grid.shape, extent=2*self.aperture_diameter)  # Twice to size to leave space for the wider OTF
        self.__phase_grid = ft.Grid(self.grid.shape, extent=8.0)  # todo: why 8 not 4?
        self.__k_max = 2 * np.pi / self.wavelength * self.numerical_aperture

        # self.__aperture = (sum(_ ** 2 for _ in self.grid.k) < self.__k_max ** 2).astype(self.datatype)
        self.__aperture = (sum(_ ** 2 for _ in self.__pupil_grid) < (self.aperture_diameter / 4) ** 2).astype(self.datatype)  # Why / 4 ?? TODO: This seems to use only half the pupil? This seems to correlated with the larger extent above
        self.__aberration_coefficients = None
        self.aberration_coefficients = 0

    @property
    def grid(self) -> ft.Grid:
        """The image grid."""
        return self.__grid

    @property
    def aberration_coefficients(self) -> np.ndarray:
        """The coefficients of the imaging system's aberrations."""
        return self.__aberration_coefficients

    @aberration_coefficients.setter
    def aberration_coefficients(self, new_coefficients):
        self.__aberration_coefficients = np.atleast_1d(new_coefficients)

    def __pad(self, a: np.ndarray) -> np.ndarray:
        return np.pad(a, [[0, _] for _ in a.shape])
        # padded_a = np.zeros_like(a, shape=np.array(a.shape) * 2)
        # padded_a[:padded_a.shape[0]//2, :padded_a.shape[1]//2] = a
        # return padded_a

    def __crop(self, padded_a):
        # print(padded_a.shape)
        return padded_a[:padded_a.shape[0]//2, :padded_a.shape[1]//2]

    # def aperture_pupil(self, padded_a):
    #     grid_pupil = ft.Grid(self.grid.shape, extent=4.0)
    #     aperture = (sum(_ ** 2 for _ in grid_pupil) < 1.0).astype(self.datatype)
    #     return aperture

    # def psf_simple_lens(self):
    #     xcor, ycor = np.meshgrid(*self.__pupil_grid)
    #     radial_cor = np.sqrt(xcor**2 + ycor**2)
    #     karif = self.wave_number*self.lens_radius*radial_cor/self.wavelength
    #     psf_lens = 4*(sp.jv(1, karif)/karif)**2
    #     return psf_lens
    def __otf(self, pupil) -> np.ndarray:
        psf = np.abs(ft.ifft2(pupil)) ** 2  # two times the number of photons
        psf = np.roll(self.__pad(ft.fftshift(psf)), -(self.grid.shape // 2), axis=(-2, -1))
        otf = ft.fft2(psf)
        return otf / otf.ravel()[0].real

    def psf(self, coefficients):
        """
        Only used for testing?

        :param coefficients:
        :return:
        """
        pupil = self.__aperture * np.exp(1j * self.phase_from_zernike_polynomial(coefficients))
        psf = ft.fftshift(np.abs(ft.ifft2(pupil)) ** 2)  # two times the number of photons
        # psf2 = np.roll(self.__pad(ft.fftshift(psf)), -(self.__pupil_grid.shape // 2), axis=(-2, -1))
        return pupil, psf

    def phase_from_zernike_polynomial(self, coefficients: Optional[np.ndarray] = None, include_system_aberration: bool = True) -> np.ndarray:
        """
        Calculates the pupil phase with an aberration (on top of the internal system aberration).

        :param coefficients: The additional Zernike coefficients.
        :param include_system_aberration: By default, include the system aberration.
        :return: The phase at the pupil.
        """
        if coefficients is None:
            coefficients = []
        combined_coefficients = [*coefficients]
        if include_system_aberration:
            for _, c in enumerate(self.aberration_coefficients):
                if _ < len(combined_coefficients):
                    combined_coefficients[_] += c
                else:
                    combined_coefficients.append(c)
        # pupil_grid = self.grid.k / self.__k_max
        phase = self.__aperture * (zernike.Polynomial(combined_coefficients).cartesian(*self.__phase_grid))
        return phase.astype(self.datatype)

    def image_from_zernike_coefficients(self, coefficients: Optional[np.ndarray] = None, include_system_aberration: bool = True) -> np.ndarray:
        """
        Calculates an image with an aberration (on top of the internal system aberration)

        :param coefficients: The additional Zernike coefficients
        :param include_system_aberration: By default, include the system aberration.
        :return: The aberrated image.
        """
        if coefficients is None:
            coefficients = []
        pupil_function = self.__aperture * np.exp(1j * self.phase_from_zernike_polynomial(coefficients, include_system_aberration=include_system_aberration))
        otf_array = self.__otf(pupil_function)

        # Convolve object with PSF
        blurred_image = np.maximum(0.0, ft.ifft2(self.__object_ft * otf_array).real)  # Non-negative, maximum value ~1
        blurred_image = self.__crop(blurred_image)

        if self.simulate_noise:
            # log.debug('Simulating Poisson photon noise...')
            # detected_image = self.__rng.poisson(
            #     (blurred_image * (1.0 - self.background_noise_level) + self.background_noise_level) * self.max_number_of_photons).astype(np.float32) / self.max_number_of_photons
            # detected_image = np.floor(detected_image * (2 ** self.bits_per_pixel)) / (2 ** self.bits_per_pixel)
            detected_image = blurred_image
            if self.background_noise_level > 0.0:
                log.debug('Simulating Gaussian noise as an approximation to Poisson noise...')
                detected_image += self.__rng.normal(scale=self.background_noise_level, size=blurred_image.shape).astype(self.datatype)  # todo: needs dtype? # Simplified noise model
            # else:
                # log.warning(f'Simulating noise requested but noise level is set is {self.background_noise_level}.')
        else:
            detected_image = blurred_image

        return detected_image

    def calc_sharpness_of_image_slow(self, img) -> float:
        img_ft = ft.fft2(img)
        img_ft /= img_ft.ravel()[0]  # Normalize
        return float(np.mean(np.abs(img_ft * self.__aperture) ** 2))

    @staticmethod
    def calc_sharpness_of_image_fast(img) -> float:
        return float(np.mean(img ** 2))

    @property
    def object(self) -> np.ndarray:
        return self.__crop(ft.ifft2(self.__object_ft))

    @object.setter
    def object(self, new_object: np.ndarray):
        self.__object_ft = ft.fft2(self.__pad(new_object))  # to speed up calc_image

    def pseudo_psf(self, coefficients: np.ndarray, bias_mode: int, bias: float) -> np.ndarray:
        bias_coeffs = np.zeros_like(coefficients)
        bias_coeffs[bias_mode] = bias
        images = [self.image_from_zernike_coefficients(coefficients + _) for _ in [-bias_coeffs, bias_coeffs]]
        img_ft = ft.fft2(images)

        min_denominator = 1e-9
        fractions_ft = [img_ft[0] / (img_ft[1] + (img_ft[1] == 0) * min_denominator),
                        img_ft[1] / (img_ft[0] + (img_ft[0] == 0) * min_denominator)]
        full_pseudo_psf = ft.fftshift(ft.ifft2(fractions_ft).real, axes=[-2, -1])
        return full_pseudo_psf

    def pseudo_psf_Wiener(self, coefficients: np.ndarray, bias_mode: int, bias: float) -> np.ndarray:
        bias_coeffs = np.zeros_like(coefficients)
        bias_coeffs[bias_mode] = bias
        images = [self.image_from_zernike_coefficients(coefficients + _) for _ in [-bias_coeffs, bias_coeffs]]
        img_ft = ft.fft2(images)

        snr_level = 5.0
        f_max = max(np.amax(np.abs(_)) for _ in self.grid.f)  # Assuming that the Grid is exactly Nyquist sampled
        nu = np.sqrt(sum((_ / f_max) ** 2 for _ in self.grid.f))
        pnsr = (nu / snr_level) ** 2

        def wiener(img_ft: np.ndarray) -> np.ndarray:
            # return 1 / img_ft
            return img_ft.conj() / (np.abs(img_ft) ** 2 + pnsr)

        fractions_ft = [img_ft[0] * wiener(img_ft[1]), img_ft[1] * wiener(img_ft[0])]
        full_pseudo_psf = ft.fftshift(ft.ifft2(fractions_ft).real, axes=[-2, -1])
        return full_pseudo_psf

    def pseudo_psf_from_images(self, images) -> np.ndarray:
        img_ft = ft.fft2(images)
        min_denominator = 1e-9
        fractions_ft = [img_ft[0] / (img_ft[1] + (img_ft[1] == 0) * min_denominator),
                        img_ft[1] / (img_ft[0] + (img_ft[0] == 0) * min_denominator)]
        full_pseudo_psf = ft.fftshift(ft.ifft2(fractions_ft).real, axes=[-2, -1])
        return full_pseudo_psf
