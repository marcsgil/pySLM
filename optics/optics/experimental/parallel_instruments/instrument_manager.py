from typing import Union, Sequence, Optional
import logging
import numpy as np
import msgpack_numpy as m
m.patch()  # For NumPy use with Pyro5

import Pyro5.api
import Pyro5.server
# Pyro5.config.COMPRESSION = False
Pyro5.config.SERIALIZER = 'msgpack'

from optics.instruments.cam import Cam
from optics.instruments.cam.web_cam import WebCam
from optics.utils import Roi

from nameserver_proxy_provider import NameServerProxyProvider

log = logging.getLogger(__name__)

array_like = Union[Sequence, np.ndarray]


@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode='single')
class BroadcastCam(Cam):
    def __init__(self, cam: Cam):
        self.__cam = cam
        super().__init__(shape=self.__cam.shape, exposure_time=self.__cam.exposure_time,
                         frame_time=self.__cam.frame_time, gain=self.__cam.gain, normalize=self.__cam.normalize,
                         color=self.__cam.color, pixel_pitch=self.__cam.pixel_pitch)

    @Pyro5.api.expose
    def acquire(self) -> np.ndarray:
        return super().acquire()

    @Pyro5.api.expose
    def stream(self, nb_frames: Optional[int] = None) -> Cam.ImageStream:
        return super().stream(nb_frames=nb_frames)

    def _set_hardware_roi(self, minimum_roi: Roi) -> Roi:
        """
        Sets the region-of-interest in the hardware so that it covers the required minimum_roi if possible.
        The actual region-of-interest that is set is returned.

        :param minimum_roi: A Roi object that indicates the minimum required region-of-interest.
        :return: The region-of-interest as set, which may be different from the requested one.
        """
        with self._lock:
            return self.__cam._set_hardware_roi(minimum_roi)

    def _set_hardware_bin(self, maximum_bin_shape: Union[int, Sequence, np.ndarray]) -> np.ndarray:
        """
        Sets the binning in the hardware so that it covers the required minimum_roi if possible.
        The actual region-of-interest that is set is returned.

        :param maximum_bin_shape: A 2-vector that indicates the maximum bin shape.
        :return: The bin shape as set, which may be different from the requested one.
        """
        with self._lock:
            return self.__cam._set_hardware_bin(maximum_bin_shape)

    def _set_hardware_exposure_time(self, exposure_time: Optional[float] = None) -> Optional[float]:
        """
        Sets the hardware exposure (a.k.a. integration) time in seconds.
        If None, then the maximum exposure is set for the frame time.
        """
        with self._lock:
            return self.__cam._set_hardware_exposure_time(exposure_time)

    def _get_hardware_exposure_time(self) -> float:
        """
        Gets the hardware exposure (a.k.a. integration) time in seconds.
        """
        with self._lock:
            return self.__cam._get_hardware_exposure_time()

    def _set_hardware_frame_time(self, frame_time: Optional[float] = None) -> float:
        """
        Sets the hardware interval between frame time in seconds.
        If None, then the minimum interval is set for the exposure time.
        """
        with self._lock:
            return self.__cam._set_hardware_frame_time(frame_time)

    def _get_hardware_frame_time(self) -> float:
        """
        Gets the hardware interval between frame time in seconds.
        """
        with self._lock:
            return self.__cam._get_hardware_frame_time()

    def _acquire(self) -> np.ndarray:
        """
        Protected method to get a raw frame from the camera.

        :return: An ndarray with dimensions: vertical x horizontal x color
        """
        with self._lock:
            return self.__cam._acquire()

    def _start_continuous(self):
        self.__cam._start_continuous()

    def _stop_continuous(self):
        self.__cam._stop_continuous()


if __name__ == '__main__':
    with NameServerProxyProvider() as nspp:
        with Pyro5.api.Daemon() as daemon:
            log.info('Creating remote object...')
            broadcast_cam = BroadcastCam(WebCam(normalize=False))
            uri = daemon.register(broadcast_cam)
            log.info(f'Registering uri = {uri}')
            nspp.proxy.register('experimental.camera', uri)

            log.info('The camera is ready...')
            daemon.requestLoop()
            log.info('The camera exited.')
