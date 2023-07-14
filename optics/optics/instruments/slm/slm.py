from abc import abstractmethod

import numpy as np
from typing import Union, Sequence, Optional, Callable, List
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import logging

from optics.instruments import InstrumentError, Instrument
from optics.instruments.display import Display
from optics.instruments.display import TkDisplay as FullScreenImpl
from optics.instruments.display import SimulatedDisplay, DisplayDescriptor
from optics.utils.array import pad_to_length
from optics.utils import Roi
from optics.utils.ft import Grid

log = logging.getLogger(__name__)

array_like = Union[complex, Sequence, np.ndarray]

__all__ = ['SLM', 'DisplaySLM', 'SLMError']


class SLMError(InstrumentError):
    """
    An Exception class for Spatial Light Modulators.
    """
    pass


class SLMDescriptor(DisplayDescriptor):
    pass


class SLM(Instrument):
    """
    A super class for spatial light modulators.
    """
    def __init__(self, deflection_frequency=(0, 0), two_pi_equivalent: float = 1.0, pixel_pitch=(1, 1), shape=(256, 256)):
        """
        Construct an abstract SLM object.

        :param deflection_frequency: The deflection frequency in the vertical and horizontal direction [1/pixels].
        A third argument can be given to indicate the axial shift (defocus) [1/pixel^2]
        :param two_pi_equivalent: The fraction of the dynamic range to use.
        :param pixel_pitch: The pixel pitch for the grid. Default 1.
        :param shape: Optional, the shape of the SLM to use, if impossible to determine automatically or when using only
        a subsection.
        """
        super().__init__()
        self.__max_shape = np.array(shape)
        self.__roi = Roi(shape=self.shape)  # Use the full SLM size
        # If the deflection_frequency frequency is a single number, assume it is diagonal
        deflection_frequency = np.atleast_1d(deflection_frequency)
        if deflection_frequency.size < 2:
            deflection_frequency = np.array((deflection_frequency, deflection_frequency, 0.0))
        elif deflection_frequency.size < 3:  # If no third number is given, no defocus is assumed
            deflection_frequency = np.array((*deflection_frequency, 0.0))
        else:
            deflection_frequency = deflection_frequency[:3]
        self.__deflection_frequency = deflection_frequency
        self.__two_pi_equivalent = two_pi_equivalent
        self.__pixel_pitch = None
        self.pixel_pitch = pixel_pitch

        self.__complex_field = np.zeros(self.roi.shape, dtype=complex)  # store the target field so that it can be reported back

        self.__pre_modulation_callback = None
        self.pre_modulation_callback = None
        self.__post_modulation_callback = None
        self.post_modulation_callback = None

        self.__corrected_deflection_phase = 0.0
        self.__corrected_deflection_amplitude = 1.0  # must be non-negative
        self.__correction = 1.0
        self.__update_corrected_deflection()

    @property
    def roi(self):
        with self._lock:
            return self.__roi

    @roi.setter
    def roi(self, new_roi: Roi):
        with self._lock:
            if new_roi is None:
                new_roi = Roi(shape=self.shape)  # Set region of interest to the maximum available
            self.__roi = new_roi
            self.__complex_field = np.zeros(self.roi.shape, dtype=complex)
            self.__corrected_deflection_phase = 0.0
            self.__corrected_deflection_amplitude = 1.0
            self.__update_corrected_deflection()

    @property
    def shape(self):
        """
        The maximum shape of this SLM.

        Note that the active region of interest may be smaller. Use slm.roi.shape to check that.
        """
        return self.__max_shape

    @property
    def pixel_pitch(self) -> np.array:
        return self.__pixel_pitch

    @pixel_pitch.setter
    def pixel_pitch(self, new_pixel_pitch: Optional[float]):
        if new_pixel_pitch is None:
            new_pixel_pitch = 1.0
        new_pixel_pitch = np.atleast_1d(new_pixel_pitch)
        if new_pixel_pitch.size < 2:
            new_pixel_pitch = np.concatenate((new_pixel_pitch, new_pixel_pitch))
        self.__pixel_pitch = new_pixel_pitch

    @property
    def grid(self):
        """
        A grid with the origin with the size of the roi at the center in metric units (if the pixel_pitch is specified).
        If pixel indices are required. Use self.roi.grid instead.
        """
        with self._lock:
            return Grid(shape=self.roi.shape, step=self.pixel_pitch)

    @property
    def two_pi_equivalent(self):
        with self._lock:
            return self.__two_pi_equivalent

    @two_pi_equivalent.setter
    def two_pi_equivalent(self, new_two_pi_equivalent):
        with self._lock:
            self.__two_pi_equivalent = np.clip(new_two_pi_equivalent, 1/1000, 10)
            self.__update_corrected_deflection()

    @property
    def correction(self):
        """
        The correction pattern as a complex nd-array.
        """
        with self._lock:
            return self.__correction

    @correction.setter
    def correction(self, new_correction):
        with self._lock:
            self.__correction = new_correction
            self.__update_corrected_deflection()

    @property
    def deflection_frequency(self):
        """
        The first-order deflection frequency of the blazed grating.
        This is a sequence of 3 real values, indicating the horizontal and vertical spatial frequencies in cycles/pixel,
        followed by the defocus of the first order (optional parabolic grating).
        """
        with self._lock:
            return self.__deflection_frequency

    @deflection_frequency.setter
    def deflection_frequency(self, new_deflection_frequency):
        """
        The first-order deflection frequency of the blazed grating.
        This is a sequence of 2 or 3 real values, indicating the horizontal and vertical spatial frequencies in
        cycles/pixel, followed by the defocus of the first order (optional parabolic grating).
        """
        with self._lock:
            new_deflection_frequency = np.array(new_deflection_frequency, dtype=float).flatten()
            new_deflection_frequency = pad_to_length(new_deflection_frequency, 3)

            self.__deflection_frequency = new_deflection_frequency
            self.__update_corrected_deflection()

    @property
    def complex_field(self):
        """
        A complex 2d-array of this shape, with the complex field at each pixel of the SLM.
        This does not include the aberration correction or the blazed grating deflection.
        """
        return self.__complex_field

    @complex_field.setter
    def complex_field(self, new_complex_field: Union[array_like, Callable]):
        """
        Sets the target complex field using either a complex 2d array of the shape of the SLM, a scalar value, or a
        function of the SLM.grid coordinates.

        :param new_complex_field: The target array or a function that produces it, without correction or 1st order deflection.
        """
        if callable(new_complex_field):
            new_complex_field = new_complex_field(*self.grid)
        new_complex_field = np.atleast_1d(new_complex_field)
        if new_complex_field.ndim < 2 or np.any(np.asarray(new_complex_field.shape) < self.roi.shape):
            new_complex_field = np.broadcast_to(new_complex_field, self.roi.shape)

        with self._lock:
            self.__complex_field = new_complex_field

            self.pre_modulation_callback(self)  # signal start of modulation

            amplitude = np.abs(self.complex_field) * self.__corrected_deflection_amplitude
            amplitude = np.clip(amplitude, 0.0, 1.0)  # Clip the amplitude just to be sure
            phase = np.angle(self.complex_field) * (amplitude > np.sqrt(np.finfo(amplitude.dtype).eps)) + self.__corrected_deflection_phase
            self._modulate(phase, amplitude)

            self.post_modulation_callback(self)  # signal end of modulation

    def modulate(self, new_complex_field):
        """
        Programs the SLM to produce the target field. Can be used as an alternative of the complex_field setter.

        :param new_complex_field: A complex 2d-array or a function in x-y pixel coordinates that generates one.
        """
        with self._lock:
            self.complex_field = new_complex_field  # Update target complex field and modulate
            return self.complex_field

    @property
    def pre_modulation_callback(self):
        with self._lock:
            return self.__pre_modulation_callback

    @pre_modulation_callback.setter
    def pre_modulation_callback(self, cb):
        with self._lock:
            if cb is None:
                cb = lambda slm: None
            self.__pre_modulation_callback = cb

    @property
    def post_modulation_callback(self):
        with self._lock:
            return self.__post_modulation_callback

    @post_modulation_callback.setter
    def post_modulation_callback(self, cb):
        with self._lock:
            if cb is None:
                cb = lambda slm: True
            self.__post_modulation_callback = cb

    def _quantize_phase(self, phase: np.ndarray) -> np.ndarray:
        """
        Convert an array of floating point phase values to integer grey-level values.

        * -pi will be converted to a grey-level of 0

        *   0 is converted to 128 * :py:obj:`two_pi_equivalent`

        *  pi will be converted to a grey-level equal to 256 * :py:obj:`two_pi_equivalent` == 0

        :param phase: The target phase as a 2d array of float values in radians.
        :return: Returns a 2d array of integer grey-level values, representing the phase at each pixel.
        """
        nb_grey_levels = 2 ** 8
        nb_grey_levels_2pi = self.two_pi_equivalent * nb_grey_levels  # if the slm can do more than 2pi, scale down to a fraction of the dynamic range
        # Convert phase [-pi, pi] to optical path difference in units of wavelength: -pi -> 0, 0 -> 0.5, 3.141 -> 1, pi -> 0
        # Scale to new dynamic range: -pi -> 0, 0 -> zero_point, 3.141 -> 1, pi -> nb_grey_levels_2pi  unless two_pi_equivalent > 1
        opd = np.mod(phase / (2 * np.pi) + 0.5 + 0.5 / nb_grey_levels_2pi, 1.0) - 0.5  # map phases to [-0.5, 0.5), slightly offset by half a gray-level for rounding later
        opd *= nb_grey_levels_2pi  # Convert to gray levels floating point values
        opd += round(min(nb_grey_levels_2pi, nb_grey_levels) / 2)  # Shift so that the zero phase is in the middle of the 2pi dynamic range, but no greater than half of the dynamic range of the SLM

        # Scale and clip to the SLM's dynamic range
        quantized_phase = np.clip(opd, 0, nb_grey_levels - 1).astype(np.uint8)  # saturate if outside range of display grey-levels

        return quantized_phase

    @abstractmethod
    def _modulate(self, phase: np.ndarray, amplitude: Union[float, np.ndarray]):
        """
        The raw modulation function that places the target phase and amplitude on the spatial light modulator.
        This is an abstract function that must be implemented by the subclasses.

        :param phase: A 2d array of shape SLM.roi.shape with the phases [-pi, pi] at each pixel.
        :param amplitude: A 2d array of shape SLM.roi.shape with the amplitudes [0, 1] at each pixel or a scalar.
        """
        pass

    def __update_corrected_deflection(self):
        log.info(f'Updating corrected deflection for {self.grid.shape} px...')
        deflection_phase = 2.0 * np.pi * (
                self.grid[0] * self.__deflection_frequency[0] + self.grid[1] * self.__deflection_frequency[1]
                + self.__deflection_frequency[2] * (self.grid[0] ** 2 + self.grid[1] ** 2)
        )
        corrected_deflection = np.exp(1j * deflection_phase) * self.__correction
        self.__corrected_deflection_phase = np.angle(corrected_deflection)
        self.__corrected_deflection_amplitude = np.abs(corrected_deflection)


class DisplaySLM(SLM):
    """
    A super class for spatial light modulators that use a video display of figure window as output.
    """
    @classmethod
    def list(cls, recursive: bool = False, include_unavailable: bool = False) -> List[Union[List, SLMDescriptor]]:
        """
        Return all constructors.
        :param recursive: List also constructors of subclass instruments.
        :param include_unavailable: List also the instruments that are not currently available.
        :return: A dictionary with as key the class and as value, a dictionary with subclasses.
        """
        # Detect the available display devices
        display_descriptors = Display.list(recursive=True, include_unavailable=include_unavailable)
        # display_descriptors = list(itertools.chain.from_iterable(display_descriptors))  # flatten nested lists
        # Each display has a potential SLM associated with it:

        def get_slm_desc(display_descs: List[DisplayDescriptor]) -> List[SLMDescriptor]:
            return [SLMDescriptor(f'{cls.__name__}_on_' + _.id, lambda: cls(_.index))
                    if isinstance(_, DisplayDescriptor) else get_slm_desc(_)
                    for _ in display_descs]

        return get_slm_desc(display_descriptors)

    def __init__(self, display_target: Union[None, int, Axes, AxesImage] = None,
                 deflection_frequency=(0, 0), two_pi_equivalent: float = 1.0, pixel_pitch=(1, 1), shape=None):
        """
        Construct an abstract SLM object that uses a monitor display.

        :param display_target: The full screen display, its number, or a figure axis or axis image to use. Default: None.
        :param deflection_frequency: The deflection frequency in the vertical and horizontal direction [1/pixels].
            A third argument can be given to indicate the axial shift (defocus) [1/pixel^2]
        :param two_pi_equivalent: The fraction of the dynamic range to use.
        :param pixel_pitch: The pixel pitch for the grid. Default 1.
        :param shape: Optional, the shape of the SLM to use, if impossible to determine automatically or when using only
            a subsection.
        """

        if display_target is None or isinstance(display_target, (Axes, AxesImage)):
            # Creates a new figure if axes_image == None
            self._full_screen = SimulatedDisplay(image_or_axes=display_target, shape=shape)
        else:
            if isinstance(display_target, Display):
                self._full_screen = display_target
            elif isinstance(display_target, int):
                self._full_screen = FullScreenImpl(display_target)
            else:
                self._full_screen = None
            if self._full_screen is not None:
                log.info(f'Using fullscreen display {display_target} with dimensions {self._full_screen.shape} for SLM.')

        if self._full_screen is not None:
            shape = self._full_screen.shape
        if shape is None:
            shape = (300, 400)

        super().__init__(shape=shape, deflection_frequency=deflection_frequency,
                         two_pi_equivalent=two_pi_equivalent, pixel_pitch=pixel_pitch)

        # Initialize an RGB image buffer for the entire SLM screen, irrespectively of ROI
        self.__image_on_slm = np.zeros((*self.shape, 3), dtype=np.uint8)

    @property
    def image_on_slm(self):
        """
        The three-dimensional numpy.ndarray of uint8 values that is sent to the display.
        """
        return self.__image_on_slm[slice(self.roi.top, self.roi.bottom), slice(self.roi.left, self.roi.right)]

    @image_on_slm.setter
    def image_on_slm(self, new_image_on_slm):
        """
        Sets the image on the SLM using either a 3d array of red-green-blue pixels of shape [*SLM.roi.shape, 3], or a
        function of the SLM.grid coordinates.

        :param new_image_on_slm: The target array, without correction or 1st order deflection.
        """
        if callable(new_image_on_slm):
            new_image_on_slm = new_image_on_slm(*self.grid)
        new_image_on_slm = np.atleast_1d(new_image_on_slm)
        if new_image_on_slm.ndim < 2 or np.any(np.asarray(new_image_on_slm.shape[:2]) < self.roi.shape):
            new_image_on_slm = np.broadcast_to(new_image_on_slm, [*self.roi.shape, 3])
        with self._lock:
            self.__image_on_slm[slice(self.roi.top, self.roi.bottom), slice(self.roi.left, self.roi.right)] = new_image_on_slm
            self._full_screen.show(self.__image_on_slm)

    def _disconnect(self):
        with self._lock:
            if self._full_screen is not None:
                self._full_screen.disconnect()
                self._full_screen = None
