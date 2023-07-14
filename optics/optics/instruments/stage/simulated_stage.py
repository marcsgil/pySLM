import numpy as np
from typing import Union, Sequence

from .stage import Stage


class SimulatedStage(Stage):
    """"
    A class to simulate a stage.
    """
    def __init__(self, ranges: Union[float, Sequence, np.ndarray] = None, ndim=None,
                 max_velocity: Union[float, Sequence, np.ndarray] = None,
                 max_acceleration: Union[float, Sequence, np.ndarray] = None):
        super().__init__(ranges=ranges, ndim=ndim, max_velocity=max_velocity, max_acceleration=max_acceleration)

    def __str__(self) -> str:
        return f'SimulatedStage with range {self.ranges} m, maximum velocity {self.max_velocity} m/s and acceleration '\
               + f' {self.max_acceleration} m/s^2. Current position {self.position} m and speed {self.velocity} m/s.'

