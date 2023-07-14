import logging
import numpy as np
from typing import Optional, Union, Sequence, Tuple, Iterable
from numbers import Complex

from optics.utils import ft
from optics.utils.ft import Grid
from optics.utils.ft.subpixel import register, roll_ft, Registration


log = logging.getLogger(__file__)

array_like = Union[Sequence, np.ndarray]


class Interferogram:
    """
    Represents complex phase images as encoded in interference patters.

    todo: Implement automated test_interferogram to make this more robust.

    Consider using the newer optics.calc.interferogram.Interferogram instead.
    """

    def __init__(self, raw_interference_image: np.ndarray,
                 registration: Optional[Registration] = None, approximate_registration: Optional[Registration] = None,
                 maximum_fringe_period: float = 10.0, grid: Optional[Grid] = None):
        """
        :param raw_interference_image: The raw 2D interference pattern image.
        :param registration: Exact registration that will be used to calculate the phase image
        :param approximate_registration: Approximate registration that will be used to get a new (exact) registration
        :param maximum_fringe_period: [units px or grid.step] Alternative to approximate_registration. The maximum
        period of interference fringes to look for.
        :param grid: a Cartesian grid object representing the pixel locations in the interference image
        """
        self.__raw_interference_image = raw_interference_image
        self.__registration = registration
        self.__approximate_registration = approximate_registration
        self.__maximum_fringe_period = maximum_fringe_period

        self.__grid = Grid(shape=self.raw_interference_image.shape[-2:]) if grid is None else grid

        self.__abs_f = np.sqrt(sum(_ ** 2 for _ in self.grid.f))

    @property
    def raw_interference_image(self) -> np.ndarray:
        """The raw interference fringe pattern (real and non-negative)."""
        return self.__raw_interference_image

    @property
    def registration(self) -> Registration:
        """
        The Fourier-space representation of the interference fringe pattern as a Registration object.

        This property is expensive to compute and cached if not provided during the construction of the Interferogram.
        """
        if self.__registration is None:
            interference_image_ft = ft.fft2(ft.ifftshift(self.__raw_interference_image))

            approx_reg = self.__approximate_registration
            if approx_reg is None:
                # Calculate the registration from interference image in Fourier space around the first order frequencies
                # Registration from an approximate interference period and high passes the Fourier space around DC
                # Of the two first orders, pick the one in the first half of space.
                search_region = self.__abs_f[..., self.grid.shape[-1] // 2:] > 0.5 / (self.__maximum_fringe_period * self.grid.step[0])
                approx_reg = register(interference_image_ft[..., self.grid.shape[-1] // 2:] * search_region)

            # Uses the registration from approximate interference period to calculate a new more precise registration
            approx_freq = np.linalg.norm(approx_reg.shift * self.grid.f.step)
            df2_array = sum((f - s) ** 2 for f, s in zip(self.grid.f, approx_reg.shift * self.grid.f.step))
            self.__registration = register(interference_image_ft * (df2_array <= (0.5 * approx_freq) ** 2))  # Low-pass around peak

        return self.__registration

    @property
    def fringe_period(self) -> array_like:
        """The spatial frequency of the interference fringe in pixels."""
        return self.grid.shape / self.registration.shift

    @property
    def amplitude(self) -> Complex:
        """The amplitude of the object in the interference pattern (complex)."""
        return self.registration.factor

    @property
    def grid(self) -> Grid:
        """Grid object representing the location of the image pixels (raw and complex image)."""
        return self.__grid

    @property
    def complex_object(self) -> np.ndarray:
        """The complex object image, extracted from the raw fringe pattern."""
        return self.__array__()

    def __array__(self, dtype=None, low_pass: bool = True,
                  keep_magnitude: bool = False, keep_phase: bool = False) -> np.ndarray:
        """
        Compute the complex object image from the raw fringe pattern.

        :param dtype: (optional) The dtype of the output.
        :param low_pass: (optional) When set to False, frequency-shift but do not low-pass the image to keep the fringe pattern.
        :param keep_magnitude: (optional) When set to False, in addition to aligning the data, rescale its (real) amplitude.
        :param keep_phase: (optional) When set to False, in addition to aligning the data, rescale its phase.

        :return: The complex object image, extracted from the raw fringe pattern.
        """
        # Demodulate the image by shifting its spectrum so that the fringe-frequency is centred and normalized to the
        # fringe amplitude and phase. The raw_interference_image is a frequency space object, so we need to have the
        # centre in the top-left corner for the following.
        correction_factor = 1.0
        if not keep_magnitude or not keep_phase:
            if keep_phase:
                correction_factor = 1.0 / abs(self.amplitude)
            elif keep_magnitude:
                correction_factor = np.exp(-1j * np.angle(self.amplitude))
            else:
                correction_factor = 1.0 / self.amplitude
        result = ft.ifftshift(self.raw_interference_image, axes=(-2, -1))
        result = roll_ft(result, self.registration.shift) * correction_factor  # Sub-pixel shifts cause phase discontinuities in the fourier-space image. We want these to be at the edge, not the center.
        result = ft.fftshift(result, axes=(-2, -1))  # Undo the earlier ifftshift.

        if low_pass:
            # Construct low-pass filter
            frequency_separation = np.linalg.norm(self.registration.shift * self.grid.f.step)
            low_pass_filter = self.__abs_f < 0.5 * frequency_separation
            # Low-pass filter
            low_passed_result_ft = ft.ifftn(result, axes=(-2, -1)) * low_pass_filter
            result = ft.fftn(low_passed_result_ft, axes=(-2, -1))

        if dtype is not None:
            result = result.astype(dtype)

        return result

    def __iter__(self) -> Iterable[np.ndarray]:
        """Iterates through the rows of the complex image."""
        return (_ for _ in self.complex_object)

    @property
    def shape(self) -> np.ndarray:
        """The shape of the raw (real) and processed (complex) image."""
        return np.asarray(self.__raw_interference_image.shape)

    @property
    def ndim(self) -> int:
        """
        The number of dimensions of the raw (real) and processed (complex) image.

        This is currently restricted to 2
        """
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of scalars in the interference image."""
        return np.prod(self.shape).item()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.raw_interference_image}, {self.registration}, grid={self.grid})'


class InterferogramOld:
    """
    Deprecated class to represent complex phase images as encoded in interference patterns. Use Interferogram instead.

    This is only used in:
                  - precision_stability_test
                  - process_pol_scan
                  - process_polarization_memory_effect

    Based on the class with the same name in `projects.confocal_interference_contrast.image_scanning_microscopy`.
    """

    def __init__(self, raw_interference_image: array_like,
                 fringe_frequency: Optional[array_like] = None, fringe_amplitude: Complex = 1.0,
                 minimum_fringe_frequency: Optional[array_like] = 0.1):
        """
        Represents an interferogram image.

        It allows to construct on Interferogram object from a fringe pattern image, which can than be used as a complex
        array, `np.asarray(Interferogram(raw_fringe_pattern_image))`, of the complex phase-amplitude object.

        :param raw_interference_image: The raw 2D interference pattern image, showing intereference fringes.
        :param minimum_fringe_frequency: The minimum fringe frequency to look for. This is ignored when
            `fringe_frequency` is specified.
        :param fringe_frequency: The optional fringe frequency to use to interprete the interference image.
            This equals [1/T_v, 1/ Th], where T_v and T_h are the vertical and horizontal periods, respectively.
            The spatial frequency can be obtained as `Registration.shift` when the registration is done on the spectrum.
        :param fringe_amplitude: The optional fringe amplitude (and phase) to enable absolute measurements between
            images. This can be obtained as `Registration.factor` when the registration is done on the spectrum.
        """
        log.warning('This class is deprecated, please use the Interference class instead with a Registration object instead of fringe_frequency and amplitude.')
        self.__raw_interference_image = raw_interference_image
        self.__minimum_fringe_frequency = minimum_fringe_frequency
        self.__fringe_frequency = fringe_frequency
        self.__fringe_amplitude = fringe_amplitude

        self.__grid = Grid(shape=self.shape)  # a pixel grid for now, centred in the center
        self.__ft_axes = np.arange(self.__grid.ndim)
        self.__complex_object = None

    @property
    def raw_interference_image(self) -> np.ndarray:
        """Returns the raw image data of the interference pattern as a real array."""
        return self.__raw_interference_image

    @property
    def fringe_frequency(self) -> array_like:
        """Interference image registration in Fourier space with a low pass around the first order frequencies."""
        if self.__fringe_frequency is None:
            self.__fringe_frequency, self.__fringe_amplitude = self.__detect_fringe()
        elif not isinstance(self.__fringe_frequency, np.ndarray):
            self.__fringe_frequency = np.array(self.__fringe_frequency)

        return self.__fringe_frequency

    @fringe_frequency.setter
    def fringe_frequency(self, new_frequency: array_like):
        self.__fringe_frequency = new_frequency

    @property
    def fringe_amplitude(self) -> Complex:
        """Interference image registration in Fourier space with a low pass around the first order frequencies."""
        if self.__fringe_frequency is None:
            self.__fringe_frequency, self.__fringe_amplitude = self.__detect_fringe()
        return self.__fringe_amplitude

    @fringe_amplitude.setter
    def fringe_amplitude(self, new_amplitude: Complex):
        self.__fringe_amplitude = new_amplitude

    @property
    def grid(self) -> Grid:
        """The Cartesian pixel grid."""
        return self.__grid

    @property
    def complex_object(self) -> np.ndarray:
        """The complex (amplitude and phase) image values that cause the interference pattern."""
        if self.__complex_object is None:  # Calculate on first read
            log.debug('Demodulating interference fringe image as a complex phase and amplitude object...')
            # Frequency shift the input (i.e. heterodyne detection)
            ishifted_img = ft.fftshift(
                roll_ft(
                    ft.ifftshift(self.raw_interference_image) / self.fringe_amplitude,
                    -self.fringe_frequency / self.grid.f.step
                )
            )
            # Block the shifted DC and other high spatial frequency components
            fringe_freq_norm = np.linalg.norm(self.fringe_frequency)
            f_proj = sum(_ * ff for _, ff in zip(self.grid.f, self.fringe_frequency / fringe_freq_norm))
            bandpass = np.abs(f_proj) < fringe_freq_norm / 2
            self.__complex_object = ft.ifftn(ft.fftn(ishifted_img, axes=self.__ft_axes) * bandpass, axes=self.__ft_axes)
        return self.__complex_object

    def __array__(self, dtype=None) -> np.ndarray:
        """The `complex_object` image (amplitude and phase) values that produced the interference pattern."""
        if dtype is None:
            return self.complex_object
        else:
            return self.complex_object.astype(dtype)

    def __iter__(self) -> Iterable[np.ndarray]:
        """Iterates through the rows of the complex image."""
        return (_ for _ in self.complex_object)

    @property
    def shape(self) -> np.ndarray:
        """The shape of the raw (real) and processed (complex) image."""
        return np.asarray(self.__raw_interference_image.shape)

    @property
    def ndim(self) -> int:
        """
        The number of dimensions of the raw (real) and processed (complex) image.

        This is currently restricted to 2
        """
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of scalars in the interference image."""
        return np.prod(self.shape).item()

    def __detect_fringe(self) -> Tuple[array_like, Complex]:
        """A private helper function to detect the periodic pattern, its frequency, phase, and amplitude."""
        log.debug('Detecting interference fringe pattern...')
        abs_f2 = sum(_ ** 2 for _ in self.grid.f)
        interference_image_ft = ft.fftn(self.raw_interference_image, axes=self.__ft_axes)
        # Registration from an approximate interference period and high passes the Fourier space around DC
        registration = register(interference_image_ft * (abs_f2 > (0.5 * self.__minimum_fringe_frequency) ** 2))
        fringe_amplitude = registration.factor / np.prod(self.shape[self.__ft_axes])
        fringe_frequency = registration.shift * self.grid.f.step
        if fringe_frequency[0] < 0 or (fringe_frequency[0] == 0 and fringe_frequency[1] < 0):
            fringe_frequency *= -1
        log.debug(f'Detected interference pattern with amplitude {np.abs(fringe_amplitude)}, ' +
                    f'phase {np.angle(fringe_amplitude) * 180 / np.pi:0.1f}, and period {1 / fringe_frequency}px.')
        return fringe_frequency, fringe_amplitude
