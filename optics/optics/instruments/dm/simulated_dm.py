import numpy as np

import logging
log = logging.getLogger(__name__)
from .dm import DM, DMError


class SimulatedDM(DM):
    """
    A class to simulate deformable mirrors.
    """
    def __init__(self, radius: float = 1.0, nb_actuators: int = 1, max_stroke: float = 1.0):
        super().__init__(radius=radius, nb_actuators=nb_actuators, max_stroke=max_stroke)

    def _modulate(self, actual_stroke_vector: np.ndarray):
        pass
