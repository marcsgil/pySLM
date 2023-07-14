import numpy as np
import logging

log = logging.getLogger(__name__)

from optics.instruments.cam import Cam
from . import Detector


class CamDetector(Detector):
    def __init__(self, cam: Cam = None, dt: float = 0.05):
        super().__init__(dt)
        self.__cam = cam
        self.__dt = dt

    @property
    def cam(self):
        return self.__cam

    @cam.setter
    def cam(self, new_cam):
        self.__cam = new_cam()

    def automatic_roi(self):  # TODO ???
        pass

    def read(self, nb_values: int) -> np.ndarray:
        # Do something, wait until values arrive, return
        frames = self.cam.stream(nb_values)
        return frames[0, 0, :]
