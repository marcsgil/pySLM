import numpy as np
import time
from typing import Callable, Union, Sequence, Optional, List

from .cam import Cam, CamDescriptor
import logging
log = logging.getLogger(__name__)
from optics.utils.roi import Roi


class SimulatedCamDescriptor(CamDescriptor):
    def __init__(self, id: str, constructor: Callable, available: bool = True):
        super().__init__(id, constructor, available)


class SimulatedCam(Cam):
    """"
    A class to simulate a camera.
    """
    @classmethod
    def list(cls, recursive: bool = False, include_unavailable: bool = False) -> List[Union[SimulatedCamDescriptor, List]]:
        """
        Return all constructors.
        :return: A dictionary with as key the class and as value, a dictionary with subclasses.
        """
        return [SimulatedCamDescriptor('CGA', lambda: SimulatedCam(shape=(240, 320))),
                SimulatedCamDescriptor('VGA', lambda: SimulatedCam(shape=(480, 640), get_frame_callback=lambda c: np.ones(c.shape)))
        ]

    def __init__(self, normalize: bool = True, exposure_time: Optional[float] = None, color: Optional[bool] = None,
                 shape: Union[None, Sequence, np.ndarray] = (300, 400),
                 get_frame_callback: Callable[[Cam], np.ndarray] = None):
        self.__auto_exposure = False
        self.__shape = np.array(shape, dtype=int)
        self.__hardware_roi = Roi(shape=self.__shape)

        self.__get_frame_callback = None

        super().__init__(shape=shape, exposure_time=exposure_time)

        if get_frame_callback is None:
            get_frame_callback = lambda _: np.zeros(self.shape)

        self.get_frame_callback = get_frame_callback

        # Do a test to see whether the callback returns color images or not
        if color is None:
            img = self.get_frame_callback(self)
            color = img.ndim >= 3

        super().__init__(shape=shape, normalize=normalize, color=color)

    @property
    def get_frame_callback(self) -> Callable[[Cam], np.ndarray]:
        with self._lock:
            return self.__get_frame_callback

    @get_frame_callback.setter
    def get_frame_callback(self, new_callback):
        with self._lock:
            if new_callback is None:
                # value = lambda: np.zeros(self.shape, dtype=np.uint8)
                new_callback = lambda _: (np.random.rand(*self.shape) * 256.0 * 0.20).astype(np.uint8)
            self.__get_frame_callback = new_callback

    def _set_hardware_roi(self, minimum_roi: Roi) -> Roi:
        """
        Sets the region-of-interest in the hardware so that it covers the required minimum_roi if possible.
        The actual region-of-interest that is set is returned.

        :param minimum_roi: A Roi object that indicates the minimum required region-of-interest.
        :return: The region-of-interest as set.
        """
        with self._lock:
            # Do exactly as requested
            self.__hardware_roi = minimum_roi
            return self.__hardware_roi

    def _acquire(self):
        with self._lock:
            start_time = time.perf_counter()

            # Get the raw data from a user-specified callback
            raw_frame = self.__get_frame_callback(self)
            # Convert to integers if necessary
            if raw_frame.dtype == float:
                raw_frame = (np.clip(raw_frame * 256, 0, 255)).astype(np.uint8)
            # Make sure that the number of dimensions is correct
            if self.color and raw_frame.ndim < 3:
                raw_frame = np.repeat(raw_frame, repeats=3, axis=2)
            elif not self.color and raw_frame.ndim >= 3:
                raw_frame = raw_frame[:, :, 0]
            # Make sure that the region-of-interest is correct
            if np.all(raw_frame.shape == self.roi.shape):
                returned_roi = self.roi
            else:
                # Assume it is centered
                returned_roi = Roi(shape=raw_frame.shape, center=np.array(self.shape / 2, dtype=int))
            # Adjust region of interest as necessary
            if np.any(self.__hardware_roi.shape != returned_roi.shape):
                # log.info(f"Region of interest {self.roi} of {returned_roi}")
                relative_roi = self.__hardware_roi / returned_roi  # Determine relative ROI
                # log.info(f"Relative roi {relative_roi}")
                raw_frame = raw_frame[
                    np.clip(relative_roi.grid[0], 0, raw_frame.shape[0] - 1).astype(dtype=int),
                    np.clip(relative_roi.grid[1], 0, raw_frame.shape[1] - 1).astype(dtype=int), ...]

            remaining_time = self.exposure_time - (time.perf_counter() - start_time)
            if remaining_time > 0 and not np.isinf(remaining_time):
                time.sleep(remaining_time)

            return raw_frame

    def __str__(self):
        return f'SimulatedCam with maximum image shape {self.shape} ' + ('in color ' if self.color else '') + \
               f'configured for exposure time {self.exposure_time * 1e3:0.1f} ms and frame time {self.frame_time * 1e3:0.1f} ms.'