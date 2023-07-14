from __future__ import annotations

from abc import abstractmethod, ABC
import numpy as np
from typing import Union, Sequence, Optional, Iterator, ContextManager, Callable

import logging
log = logging.getLogger(__name__)
from optics.instruments import InstrumentError, Instrument, InstrumentDescriptor
from optics.utils import Roi


class CamError(InstrumentError):
    """
    An Exception for Cam objects.
    """
    pass


class CamDescriptor(InstrumentDescriptor):
    def __init__(self, id: str, constructor: Callable, available: bool = True, serial: str = '', model: str = ''):
        super().__init__(id, constructor, available)
        self.__serial: str = serial
        self.__model: str = model

    @property
    def serial(self) -> str:
        """The serial number of the camera"""
        return self.__serial

    @property
    def model(self) -> str:
        """The model of the camera."""
        return self.__model


class Cam(Instrument):
    """
    A super-class for camera objects.
    """
    class ImageStream(Iterator, ContextManager):
        """
        An inner class for streaming images as an iterable context manager,
        while starting and stopping continuous acquisition as required.
        """
        def __init__(self, cam: Cam, nb_frames: Optional[int] = None):
            self.__cam = cam
            self.__max_nb_frames = nb_frames
            self.__frame_index = 0

        def __enter__(self):
            self.__cam._start_continuous()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.__cam._stop_continuous()

        def __iter__(self) -> Iterator[np.ndarray]:
            return self

        def __next__(self) -> np.ndarray:
            if self.__max_nb_frames is not None:
                self.__frame_index += 1
                if self.__frame_index > self.__max_nb_frames:
                    raise StopIteration
            return self.__cam.acquire()

        def acquire(self):
            return self.__next__()

        def __len__(self):
            if self.__max_nb_frames is None:
                return np.inf
            else:
                return self.__max_nb_frames

        def __str__(self):
            return f'ImageStream from {str(self.__cam)}.'

    def __init__(self, shape: Union[None, Sequence, np.ndarray] = None,
                 exposure_time: Optional[float] = 1e-3, frame_time: Optional[float] = None, gain: float = 1.0,
                 normalize: bool = True, color: bool = False, pixel_pitch: Union[float, Sequence, np.ndarray] = 1e-6):
        """
        Base class for camera objects.

        :param shape: The total width and height of the sensor area in pixels.
        :param exposure_time: The integration or exposure time in seconds.
        :param frame_time: The time in seconds between consecutive frames. None: the minimum time for the set exposure
        time. Default: None.
        :param gain: The gain as a floating point value.
        :param normalize: When True, the dynamic range of the returned images is scaled to the interval [0, 1], with 1.0
        as the maximum value of the dynamic range.
        :param color: When True, the image is returned as a 3d-array where the third dimension is the color channel.
        :param pixel_pitch: A 2-element vector indicating the pixel spacing in meters.
        """
        super().__init__()

        self.__exposure_time = exposure_time if exposure_time is not None else frame_time
        self.__frame_time = frame_time if frame_time is not None else exposure_time
        self.__gain = gain
        self.__hardware_roi = None
        self.__roi = None  # Invalid, fixed later
        self.__hardware_bin = np.ones(2, dtype=int)
        self.__bin = np.ones(2, dtype=int)
        self.__bit_depth = 8
        self.__color = color
        self.__normalize = normalize
        self.__background = None
        self.__image_stream = None

        if np.isscalar(pixel_pitch):
            pixel_pitch = np.full(2, pixel_pitch)
        self._pixel_pitch = np.array(pixel_pitch).flatten()

        if shape is not None:
            self.__shape = np.array(shape, dtype=int).flatten()
            self.roi = None  # Default the region-of-interest to the full sensor area
            if exposure_time is not None:  # Only update when explicitly set
                self._set_hardware_exposure_time(exposure_time)  # if None, set to the maximum possible
            if frame_time is not None:  # Only update when explicitly set
                self._set_hardware_frame_time(frame_time)  # if None, set to the minimum possible

    @property
    def shape(self) -> np.ndarray:
        """
        The physical shape of this sensor array sets a maximum for the region-of-interest.
        """
        with self._lock:
            return self.__shape

    @property
    def roi(self) -> Roi:
        """
        The region-of-interest as a Rect object.
        """
        with self._lock:
            return self.__roi

    @roi.setter
    def roi(self, new_roi):
        with self._lock:
            self.background = None
            if new_roi is None:
                new_roi = Roi(shape=self.shape)

            self.__roi = Roi(new_roi, dtype=int)

            cropped_roi = self.__roi % Roi(shape=self.shape)
            # log.debug(f"Cropping roi:\n{cropped_roi} = {self.__roi} % {Roi(shape=self.shape)}")
            self.__hardware_roi = self._set_hardware_roi(cropped_roi)
            # log.debug(f"Hardware roi: {self.__hardware_roi}")

    @property
    def bin(self) -> np.ndarray:
        return self.__bin

    @bin.setter
    def bin(self, bin_shape: Union[int, Sequence, np.ndarray]):
        if bin_shape is None:
            bin_shape = 1
        bin_shape = np.atleast_1d(bin_shape, dtype=int).flatten()
        if bin_shape.size < 2:
            bin_shape = np.repeat(bin_shape, repeats=2)

        self.__bin = bin_shape
        self.__hardware_bin = self._set_hardware_bin(bin_shape)

    @property
    def color(self) -> bool:
        """"""
        with self._lock:
            return self.__color

    @color.setter
    def color(self, use_color: bool):
        """"""
        with self._lock:
            self.__color = use_color

    @property
    def normalize(self) -> bool:
        with self._lock:
            return self.__normalize

    @normalize.setter
    def normalize(self, state: bool):
        with self._lock:
            self.background = None
            self.__normalize = state

    @property
    def background(self) -> Union[int, np.ndarray]:
        with self._lock:
            return self.__background

    @background.setter
    def background(self, img: Union[int, np.ndarray]):
        with self._lock:
            if img is None:
                log.debug('Not using background subtraction.')
            else:
                log.debug('Using background subtraction.')

            self.__background = img

    def acquire(self) -> np.ndarray:
        with self._lock:
            # Get the raw data from the protected specific implementation
            if np.all(self.roi.shape > 0):
                # Delegate the actual acquisition to the specialized _acquire()
                image_array = self._acquire()
                if image_array.shape[0] != self.__hardware_roi.shape[0] \
                        or image_array.shape[1] != self.__hardware_roi.shape[1]:
                    log.error(f"The camera returned an image of shape {image_array.shape[:2]},"
                              + f" while the hardware region-of-interest is {self.__hardware_roi}."
                              + " Either the camera object didn't set the correct hardware ROI, or it returned"
                              + " incorrect image data.")

                # Adjust region-of-interest if needed
                if self.roi != self.__hardware_roi:
                    # log.debug(f"Region of interest {self.roi} of hardware {self.__hardware_roi}" +
                    #           f" with maximum {self.shape}.")
                    relative_roi = self.roi / self.__hardware_roi  # Determine relative ROI
                    image_array = image_array[
                        np.clip(relative_roi.grid[0], 0, self.__hardware_roi.shape[0] - 1).astype(dtype=int),
                        np.clip(relative_roi.grid[1], 0, self.__hardware_roi.shape[1] - 1).astype(dtype=int)]
            else:
                image_array = np.zeros(self.roi.shape, dtype=np.uint8)

            # Convert to the desired format
            if image_array.ndim < 3:
                image_array = image_array[:, :, np.newaxis]
            if not self.color and image_array.shape[2] > 1:
                if image_array.dtype == np.uint8:
                    new_type = np.uint16
                    nb_extra_bits = 8
                elif image_array.dtype == np.uint16:
                    new_type = np.uint32
                    nb_extra_bits = 16
                elif image_array.dtype == np.uint32:
                    new_type = np.uint64
                    nb_extra_bits = 32
                else:
                    new_type = np.uint128
                    nb_extra_bits = 64
                image_array = np.sum(image_array, axis=2, keepdims=False, dtype=new_type) * int(2**nb_extra_bits / 3)
            if self.normalize:
                image_array = image_array.astype(float) / np.iinfo(image_array.dtype).max

            if image_array.ndim >= 3 and image_array.shape[2] == 1:
                image_array = image_array[:, :, 0]

            # Subtract the background in the end
            if self.background is not None:
                positive_pixels = image_array > self.background
                image_array[positive_pixels] -= self.background[positive_pixels]
                image_array[np.logical_not(positive_pixels)] = 0  # avoid negative values

            return image_array

    #
    # Override the following properties and methods in the child class
    #

    @property
    def bit_depth(self):
        with self._lock:
            return self.__bit_depth

    @property
    def exposure_time(self) -> float:
        """
        Gets the exposure (a.k.a. integration) time in seconds.
        """
        with self._lock:
            return self._get_hardware_exposure_time()

    @exposure_time.setter
    def exposure_time(self, exposure_time: Optional[float]):
        """
        Sets the exposure (a.k.a. integration) time in seconds.
        If None, then the maximum exposure is set for the current frame time.

        This property is implementation independent. A sub-class should override _set_hardware_exposure_time.
        """
        with self._lock:
            self.background = None
            self.__exposure_time = self._set_hardware_exposure_time(exposure_time)  # Store the result in case _get_hardware_exposure time is not redefined

    @property
    def frame_time(self) -> float:
        """
        Gets the inter-frame time or interval in seconds.
        """
        with self._lock:
            return self._get_hardware_frame_time()

    @frame_time.setter
    def frame_time(self, frame_time: Optional[float]):
        """
        Sets the inter-frame time or interval in seconds.
        If None, then the minimum frame time is set for the current exposure time.
        """
        with self._lock:
            self.background = None
            self.__frame_time = self._set_hardware_frame_time(frame_time)  # Store the result in case _get_hardware_frame_time time is not redefined

    @property
    def gain(self) -> float:
        """Gets the gain value as a floating point number in [0, 1]."""
        with self._lock:
            return self.__gain

    @gain.setter
    def gain(self, gain: float):
        """Sets the gain value, a floating point number in [0, 1]."""
        with self._lock:
            self.background = None
            self.__gain = gain

    @property
    def pixel_pitch(self) -> np.ndarray:
        return self._pixel_pitch

    def stream(self, nb_frames: Optional[int] = None) -> ImageStream:
        """
        Get a stream of images as an interable context manager.

        Usage:
        with cam.stream as s:
            for img in s:
                plt.image(img)
                plt.show()
                plt.sleep(0.001)

        :param nb_frames: (optional) The maximum number of frames to return.
        :return: A Cam.ImageStream object.
        """
        if self.__image_stream is not None:
            # Make sure that the previous stream is stopped
            self.__image_stream.__exit__(exc_type=None, exc_val=None, exc_tb=None)
        self.__image_stream = Cam.ImageStream(cam=self, nb_frames=nb_frames)
        return self.__image_stream

    def center_roi_around_peak_intensity(self, shape: Union[int, Sequence, np.ndarray] = (25, 25),
                                         target_graylevel_fraction: float = 0.75) -> Cam:
        """
        Centers the region of interest around the peak intensity in the current region of interest.

        The algorithm acquires an image from the camera, finds the position of maximal intensity and centers the cam's
        region of interest around it. If the image is saturated it halves the cam's integration time until it is not
        saturated anymore.

        :param shape: The shape of the roi (optional).
        :param target_graylevel_fraction: The camera's exposure is changed until the gray level is approximately this value
            times its maximum.

        :return: self
        """
        with self._lock:
            if np.all(self.roi.shape > 0):
                normalize_preset = self.normalize
                color_preset = self.color

                self.normalize = False
                self.color = False
                img = self.acquire()

                peak_index = np.argmax(img)
                max_graylevel = (2**self.bit_depth - 1)
                # if the image is saturated:
                for attempt in range(32):
                    if img.ravel()[peak_index] >= max_graylevel:
                        self.exposure_time /= 2
                        img = self.acquire()
                        peak_index = np.argmax(img)
                    else:
                        break
                # Set exposure so that the dynamic range is at a given level
                self.exposure_time /= img.ravel()[peak_index] / max_graylevel / target_graylevel_fraction

                center = np.unravel_index(peak_index, shape=img.shape)
                relative_roi = Roi(shape=shape, center=center, dtype=int)

                absolute_roi = relative_roi * self.roi

                self.roi = absolute_roi
                # reset the state
                self.color = color_preset
                self.normalize = normalize_preset

        return self

    def _set_hardware_roi(self, minimum_roi: Roi) -> Roi:
        """
        Sets the region-of-interest in the hardware so that it covers the required minimum_roi if possible.
        The actual region-of-interest that is set is returned.
        
        :param minimum_roi: A Roi object that indicates the minimum required region-of-interest.
        :return: The region-of-interest as set, which may be different from the requested one.
        """
        with self._lock:
            return Roi(shape=self.shape)

    def _set_hardware_bin(self, maximum_bin_shape: Union[int, Sequence, np.ndarray]) -> np.ndarray:
        """
        Sets the binning in the hardware so that it covers the required minimum_roi if possible.
        The actual region-of-interest that is set is returned.

        :param maximum_bin_shape: A 2-vector that indicates the maximum bin shape.
        :return: The bin shape as set, which may be different from the requested one.
        """
        maximum_bin_shape = np.atleast_1d(maximum_bin_shape).astype(int).flatten()
        if maximum_bin_shape.size < 2:
            maximum_bin_shape = np.repeat(maximum_bin_shape, repeats=2)
        with self._lock:
            return maximum_bin_shape

    def _set_hardware_exposure_time(self, exposure_time: Optional[float] = None) -> Optional[float]:
        """
        Sets the hardware exposure (a.k.a. integration) time in seconds.
        If None, then the maximum exposure is set for the frame time.

        This method can be overridden in a sub-class
        """
        with self._lock:
            if exposure_time is None:
                exposure_time = 1.0  # Maximum is 1s for the default implementation.
            self.__exposure_time = exposure_time
            if exposure_time > self.frame_time:
                self.__frame_time = self.__exposure_time
            return self.__exposure_time

    def _get_hardware_exposure_time(self) -> float:
        """
        Gets the hardware exposure (a.k.a. integration) time in seconds.

        This method can be overridden in a sub-class
        """
        with self._lock:
            return self.__exposure_time

    def _set_hardware_frame_time(self, frame_time: Optional[float] = None) -> float:
        """
        Sets the hardware interval between frame time in seconds.
        If None, then the minimum interval is set for the exposure time.

        This method can be overridden in a sub-class
        """
        with self._lock:
            if frame_time is None:
                frame_time = 1e-6  # Maximum is 1us for the default implementation.
            self.__frame_time = frame_time
            if self.__frame_time < self.__exposure_time:
                self.__exposure_time = self.__frame_time
            return self.__frame_time

    def _get_hardware_frame_time(self) -> float:
        """
        Gets the hardware interval between frame time in seconds.

        This method can be overridden in a sub-class
        """
        with self._lock:
            return self.__frame_time

    @abstractmethod
    def _acquire(self) -> np.ndarray:
        """
        Protected method to get a raw frame from the camera.

        :return: An ndarray with dimensions: vertical x horizontal x color
        """
        with self._lock:
            dtype = np.uint8
            img = np.zeros(self.__hardware_roi.shape, dtype=dtype)
            if self.__color:
                img = np.repeat(img, repeats=3, axis=2)  # Simulate 3 color channels
            return img

    def _start_continuous(self):
        pass

    def _stop_continuous(self):
        pass

