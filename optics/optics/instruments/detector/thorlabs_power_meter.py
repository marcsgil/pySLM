import numpy as np
import logging

log = logging.getLogger(__name__)

from . import Detector


class ThorlabsPowerMeterDetector(Detector):
    def __init__(self, dt: float = 0.05):
        super().__init__(dt)

    def read(self, nb_values: int) -> np.ndarray:
        pass
