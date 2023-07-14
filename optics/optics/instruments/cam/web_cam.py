import logging
log = logging.getLogger(__name__)

try:
    from optics.external import v4l2
    import fcntl
    vid = open('/dev/video0')
    cap = v4l2.v4l2_capability()
    fcntl.ioctl(vid, v4l2.VIDIOC_QUERYCAP, cap)
    log.info(cap.driver)
    log.info(cap.card)
    log.info(f'{cap.capabilities:032b}')
    log.info(f'{v4l2.V4L2_CAP_VIDEO_CAPTURE:032b} v4l2.V4L2_CAP_VIDEO_CAPTURE')
    log.info(f'{v4l2.V4L2_CAP_STREAMING:032b} v4l2.V4L2_CAP_STREAMING')
    _hardware_exposure_time_exponential = False  # On Linux?
    _hardware_exposure_time_scaling_factor = 2.0 / 65536
    _use_direct_capture = False
except Exception as err:
    log.debug(f'Could not import v4l2 for WebCam: {err}, probably on Windows')
    _hardware_exposure_time_exponential = True  # On Windows
    _hardware_exposure_time_scaling_factor = 1.0
    _use_direct_capture = True

import cv2
import numpy as np
from typing import Optional, Callable, List, Union

from .cam import Cam, CamDescriptor


class WebCamDescriptor(CamDescriptor):
    def __init__(self, id: str, constructor: Callable, available: bool = True):
        super().__init__(id, constructor, available)


class WebCam(Cam):
    """
    A class to control webcams.
    """
    @classmethod
    def list(cls, recursive: bool = False, include_unavailable: bool = False) -> List[Union[WebCamDescriptor, List]]:
        """
        Return all constructors.
        :return: A dictionary with as key the class and as value, a dictionary with subclasses.
        """
        # Until I know how to detect webcams
        return [WebCamDescriptor(_, lambda: WebCam(_)) for _ in range(4)]

    def __init__(self, index: int = 0, normalize: bool = True, color: Optional[bool] = None, exposure_time: Optional[float] = None):
        self.__camera_index = index

        # Conversion factors between hardware exposure value and seconds
        # This is needed because OpenCV does not convert exposure times.
        self.hardware_exposure_time_exponential = _hardware_exposure_time_exponential
        self.hardware_exposure_time_scaling_factor = _hardware_exposure_time_scaling_factor
        self.hardware_exposure_time_exponent = 2.0

        self.__video_capture = cv2.VideoCapture(index, cv2.CAP_DSHOW if _use_direct_capture else cv2.CAP_ANY)

        # fps = self.__video_capture.get(cv2.CAP_PROP_FPS)
        if exposure_time is None:
            register_value = self.__video_capture.get(cv2.CAP_PROP_EXPOSURE)
            if self.hardware_exposure_time_exponential:
                exposure_time = self.hardware_exposure_time_exponent ** np.clip(register_value, -10, 10)
            else:
                exposure_time = register_value
            exposure_time *= self.hardware_exposure_time_scaling_factor
        # self.__video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.__gain = 1.0  # self.__video_capture.get(cv2.CAP_PROP_GAIN)
        height = self.__video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = self.__video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        shape = np.array((height, width), dtype=int)

        super().__init__(shape=shape, exposure_time=exposure_time, normalize=normalize, color=color)

        self.__auto_exposure = True

    def _get_hardware_exposure_time(self) -> float:
        with self._lock:
            register_value = self.__video_capture.get(cv2.CAP_PROP_EXPOSURE)
            if self.hardware_exposure_time_exponential:
                exposure_time = self.hardware_exposure_time_exponent ** np.clip(register_value, -10, 10)
            else:
                exposure_time = register_value
            return self.hardware_exposure_time_scaling_factor * exposure_time

    def _set_hardware_exposure_time(self, exposure_time: Optional[float] = None) -> Optional[float]:
        log.info(f'Setting exposure to {exposure_time}...')
        with self._lock:
            if exposure_time is None:
                exposure_time = self.frame_time if self.frame_time is not None else 1.0
            self.background = None
            if self.__video_capture is not None:
                # self.__video_capture.set(cv2.CAP_PROP_EXPOSURE, integration_time * 1e3)
                # integration_time = self.__video_capture.get(cv2.CAP_PROP_EXPOSURE) * 1e-3
                exposure_time = exposure_time / self.hardware_exposure_time_scaling_factor
                if self.hardware_exposure_time_exponential:
                    register_value = np.log2(np.clip(exposure_time, 1e-6, 10.0)) / np.log2(self.hardware_exposure_time_exponent)
                    register_value = np.clip(int(register_value), -9, 0)
                else:
                    register_value = exposure_time
                result = self.__video_capture.set(cv2.CAP_PROP_EXPOSURE, register_value)
                if not result:
                    log.warning(f'Error setting exposure time to {exposure_time}.')

            return exposure_time

    @property
    def gain(self) -> float:
        with self._lock:
            return self.__gain

    @gain.setter
    def gain(self, gain: float):
        log.info(f'Setting gain to {gain}...')
        with self._lock:
            self.background = None
            if self.__video_capture is not None:
                self.__video_capture.set(cv2.CAP_PROP_GAIN, gain)
                # self._gain = self.__video_capture.get(cv2.CAP_PROP_GAIN)
                self.__gain = gain  # Because the above doesn't seem to work

    @property
    def auto_exposure(self):
        with self._lock:
            return self.__auto_exposure

    @auto_exposure.setter
    def auto_exposure(self, value: bool):
        with self._lock:
            self.background = None
            if self.__video_capture is not None:
                values = (0.25, 0.75)  # (auto exposure OFF, auto exposure ON)
                self.__video_capture.read()  # Capture a reference image
                result = self.__video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, values[value])
                if not result:
                    log.warning(f'Could not set auto-exposure to value {values[value]}, likely not supported by OpenCV2 for this camera. You may want to reset this using, e.g. v4l2-ctl or the OS webcam settings.')

                self.__auto_exposure = value

    @property
    def camera_index(self):
        with self._lock:
            return self.__camera_index

    def _acquire(self):
        with self._lock:
            raw_frame = None
            if self.__video_capture is not None:
                success, raw_frame = self.__video_capture.read()
                if success:
                    raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)  # OpenCV has BGR ordering instead of RGB
            if raw_frame is None:
                log.warning('WebCam could not capture the frame, defaulting to all zeros!')
                raw_frame = np.zeros((*self.shape, 3), dtype=np.uint8)
            return raw_frame

    def _disconnect(self):
        with self._lock:
            if self.__video_capture is not None:
                try:
                    self.exposure_time = 1.0 / 24
                    self.auto_exposure = True
                    self.__video_capture.release()
                    self.__video_capture = None
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(f"WebCam error while closing {e}")
                    pass
