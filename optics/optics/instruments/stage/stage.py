from __future__ import annotations

from contextlib import AbstractContextManager
import logging
import numpy as np
from typing import Union, Sequence, Callable
import traceback
import time
from optics.utils.ft import Grid

from optics.instruments import InstrumentError, Instrument
from optics.instruments.detector import Detector

log = logging.getLogger(__name__)

__all__ = ['StageError', 'Translation', 'Stage']


class StageError(InstrumentError):
    """
    An Exception for Stage objects.
    """
    pass


class Translation(AbstractContextManager):
    """
    A ContextManager class to represent a stage translation.
    """
    def __init__(self, stage: Stage, max_acceleration: Union[None, float, Sequence, np.ndarray],
                 velocity: Union[None, float, Sequence, np.ndarray] = None,
                 origin: Union[None, float, Sequence, np.ndarray] = None,
                 destination: Union[None, float, Sequence, np.ndarray] = None):
        # Save mandatory parameters
        self.__stage: Stage = stage
        # Set default parameters
        self.__target_velocity = np.atleast_1d(velocity).flatten()
        if origin is not None:
            origin = np.clip(np.atleast_1d(origin), self.__stage.ranges[:, 0], self.__stage.ranges[:, -1])
            log.debug(f'Moving to origin {origin*1e3} mm...')
            self.__stage.position = origin
        else:
            origin = self.__stage.position
        if destination is not None:
            destination = np.clip(np.atleast_1d(destination), self.__stage.ranges[:, 0], self.__stage.ranges[:, -1])
        else:
            destination = np.inf * np.sign(velocity)
        # Store for later use
        self.__origin = origin
        self.__destination = destination

        # Set velocity in mm/s and acceleration in mm/s^2
        self.__acceleration = np.sign(velocity) * max_acceleration

        # The initial time and position
        self.__previous_time = time.perf_counter()
        self.__previous_position = self.__stage.position
        self.__previous_velocity = np.zeros(self.ndim)

        self.__in_progress = False

    @property
    def ndim(self):
        return self.__stage.ndim

    @property
    def origin(self) -> np.ndarray:
        return self.__origin

    @property
    def destination(self) -> np.ndarray:
        return self.__destination

    @property
    def velocity(self) -> np.ndarray:
        """
        Get the target velocity.
        When 0.0, the stage has reached its destination.
        """
        return self.__target_velocity

    @property
    def position(self) -> np.ndarray:
        # Determine the elapsed time since the previous position update
        current_time = time.perf_counter()
        elapsed_time = current_time - self.__previous_time
        if np.any(self.__previous_velocity != self.__target_velocity):
            velocity_difference = self.__target_velocity - self.__previous_velocity
            time_to_target_velocity = np.amax(velocity_difference / self.__acceleration)
        else:
            time_to_target_velocity = 0.0
        # Accelerate until the target velocity is reached
        acceleration_time = np.minimum(elapsed_time, time_to_target_velocity)
        pos = self.__previous_position + self.__previous_velocity * acceleration_time
        if acceleration_time > 0:
            pos += self.__acceleration * 0.5 * (acceleration_time**2)
        # Cruise at target velocity
        if elapsed_time >= time_to_target_velocity:
            pos += self.__target_velocity * (elapsed_time - time_to_target_velocity)
            self.__previous_time = current_time
            self.__previous_position = pos
            self.__previous_velocity = self.__target_velocity
        if np.all((self.destination - pos) * self.__target_velocity <= 0):
            self.stop()
        return pos

    def start(self):
        """
        Start the stage translation. This is automatically triggered by the Context Manager.
        """
        self.__in_progress = True
        self.__stage.velocity = self.__target_velocity
        # Store the current state to simulate movement
        self.__previous_time = time.perf_counter()
        self.__previous_position = self.position
        self.__previous_velocity = np.zeros(self.__stage.ndim)

    def stop(self):
        """
        Stop the stage translation. This is triggered the Context Manager when self.position is called and the
        destination is reached.
        """
        self.__stage.velocity = np.zeros(self.ndim)
        # Break
        self.__target_velocity = np.zeros(self.ndim)
        self.__acceleration *= -1
        self.__in_progress = False

    @property
    def in_progress(self) -> bool:
        return self.__in_progress

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
        if exc_type is not None:
            log.error('Exception while using Stage object: ' + str(exc_type))
            traceback.print_tb(exc_tb)
            raise exc_val
        return True

    def __del__(self):
        if self.__stage is not None:
            self.stop()
        self.__stage = None

    def __str__(self):
        return f'stage.Translation(velocity={self.velocity}, origin={self.origin}, destination={self.destination})'


class Stage(Instrument):
    """"
    A super-class for stage objects.
    """
    def __init__(self, ranges: Union[float, Sequence, np.ndarray] = None, ndim: int = None,
                 max_velocity: Union[float, Sequence, np.ndarray] = None,
                 max_acceleration: Union[float, Sequence, np.ndarray] = None,
                 power_down_on_disconnect: bool = True):
        """
        Base class for stage objects.

        :param ranges: The travel ranges as a sequence of 2-vectors containing the minimum and maximum value for each
        axis. This can be specified as a 2d-array of shape [self.ndim, 2]. If a scalar is specified, it is interpreted
        as the maximum value of a range starting at 0. In this case the number of axis is 1 unless specified otherwise
        by ndim. If not specified, unlimited travel is assumed (e.g. for rotation stages): [(-np.inf, np.inf)]
        :param ndim: Optional. The number of dimensions. Default: as given by ranges.shape[0].
        :param max_velocity: Optional. The maximum velocity per axes. If a single number is provided, the same value is
        assumed for all dimensions. Default: infinite
        :param max_acceleration: Optional. The maximum velocity per axes. If a single number is provided, the same value is
        assumed for all dimensions. Default: infinite
        :param: power
        """
        super().__init__(power_down_on_disconnect=power_down_on_disconnect)

        if ranges is None:
            ranges = [(-np.inf, np.inf)]
        elif np.isscalar(ranges):
            ranges = [(0, ranges)]
        ranges = np.atleast_2d(ranges)
        if ranges.shape[1] != 2:
            raise TypeError(f'The ranges of a stage should be specified as a 2d-array with one row per axis and two elements per row, not {ranges.shape[1]}.')
        # Replace None by the appropriate infinity
        ranges[ranges[:, 0] == None, 0] = -np.inf
        ranges[ranges[:, 1] == None, 1] = np.inf

        if ndim is None:
            ndim = ranges.shape[0]
        else:
            if ranges.shape[0] == 1 and ndim > 1:
                ranges = np.tile(ranges, (ndim, 1))

        if max_velocity is None:
            max_velocity = np.inf
        max_velocity = np.atleast_1d(max_velocity).flatten()
        if max_velocity.size == 1:
            max_velocity = np.repeat(max_velocity, ndim)
        elif max_velocity.size != ndim:
            raise TypeError(f'The number of maximum velocities, {max_velocity.size}, should match the number of dimensions, {ndim}.')
        if max_acceleration is None:
            max_acceleration = np.inf
        max_acceleration = np.atleast_1d(max_acceleration).flatten()
        if max_acceleration.size == 1:
            max_acceleration = np.repeat(max_acceleration, ndim)
        elif max_acceleration.size != ndim:
            raise TypeError(f'The number of maximum accelerations, {max_acceleration.size}, should match the number of dimensions, {ndim}.')

        self.__ndim = ndim
        self.__ranges = ranges
        self.__max_velocity = max_velocity
        self.__max_acceleration = max_acceleration

        self.__start_time = 0.0
        self.__position = np.zeros(self.ndim)
        self.__velocity = np.zeros(self.ndim)
        self.__translation = None

    def __str__(self) -> str:
        return f'{__class__.__name__} with range {self.ranges} m, maximum velocity {self.max_velocity} m/s and acceleration {self.max_acceleration} m/s^2.'

    @property
    def ndim(self) -> int:
        return self.__ndim

    @property
    def ranges(self) -> int:
        return self.__ranges

    @property
    def max_velocity(self) -> np.ndarray:
        return self.__max_velocity

    @property
    def max_acceleration(self) -> np.ndarray:
        return self.__max_acceleration

    @property
    def position(self) -> np.ndarray:
        with self._lock:
            current_time = time.perf_counter()
            elapsed_time = current_time - self.__start_time
            position = self.__position + elapsed_time * self.__velocity
            for axis in range(self.ndim):
                position[axis] = np.clip(position[axis], self.ranges[axis][0], self.ranges[axis][-1])
            self.__start_time = current_time
            self.__position = position
            return self.__position

    @position.setter
    def position(self, new_position: Union[float, Sequence, np.ndarray]):
        with self._lock:
            new_position = np.atleast_1d(new_position)
            if new_position.size != self.ndim:
                raise TypeError(f'This stage has {self.ndim} axes, but the position is a {new_position.size}-element vector!')
            new_position = np.clip(new_position, self.ranges[:, 0], self.ranges[:, 1])
            distance = new_position - self.position
            time_difference = np.amax(np.abs(distance) / self.max_velocity)  # Wait until the position is reached for all axes.
            # simulate a delay
            time.sleep(time_difference)
            self.__position = new_position

    @property
    def velocity(self) -> np.ndarray:
        return self.__velocity

    @velocity.setter
    def velocity(self, new_velocity: Union[float, Sequence, np.ndarray]):
        with self._lock:
            new_velocity = np.atleast_1d(new_velocity)
            if new_velocity.size != self.ndim:
                raise TypeError(f'This stage has {self.ndim} axes, but the velocity is a {new_velocity.size}-element vector!')
            # Update current internal position
            self.position = self.position
            # Update internal velocity
            self.__velocity = np.minimum(np.abs(new_velocity), self.max_velocity) * np.sign(new_velocity)

    def translate(self, velocity: Union[None, float, Sequence, np.ndarray] = None,
                  origin: Union[None, float, Sequence, np.ndarray] = None,
                  destination: Union[None, float, Sequence, np.ndarray] = None) -> Translation:
        """
        Translate the stage.
        Usage:

        ::

            with stage.translation(1e-3) as t:
                while t.position < 10e-3:
                    print(t.position)

        :param velocity: The translation velocity in m/s (default: the maximum velocity for this stage).
        :param origin: (optional) The origin position in meters (default: the current position).
        :param destination: (optional) The destination in meters (default: the end of the stage's range).
        :return: A context manager object for the stage Translation.
        """
        with self._lock:
            if velocity is None:
                velocity = self.max_velocity
                if origin is not None and destination is not None:
                    velocity = velocity * np.sign(destination - origin)
            else:
                velocity = np.atleast_1d(velocity)
                velocity = np.minimum(self.max_velocity, np.abs(velocity)) * np.sign(velocity)
            # Make sure that there is only one Translation object associated with this stage
            if self.__translation is not None:
                self.__translation.stop()
            self.__translation = Translation(self, self.max_acceleration,
                                             velocity=velocity, origin=origin, destination=destination)
            return self.__translation

    def scan(self, grid: Grid, detector: Detector, callback: Callable = None) -> np.ndarray:  # Currently need to specify a 3D grid
        scan_order = np.argsort(grid.shape)[::-1]  # scanning along the longest axis
        scan_extent = np.array([grid.center - grid.extent/2, grid.center + grid.extent/2])[scan_order[0]]

        v = np.zeros(3)
        v[scan_order[0]] = grid.step[scan_order[0]] / detector.dt  # line scan velocity
        n = grid.shape[scan_order[0]]
        # this variable will be updated for the purposes of un-timed stage movement, but is an inaccurate representation of the current position
        position = grid.center

        # preparing to save the result
        result = np.zeros(grid.shape)
        slc_result = [0, 0, 0]
        slc_result[scan_order[0]] = slice(None)

        for count, line in enumerate(grid[scan_order[1]]):

            # moving the stage to the start of the line
            scan_extent = scan_extent[::-1] if (count % 2) else scan_extent
            position[scan_order[0]] = scan_extent[0]
            position[scan_order[1]] = line
            self.position = position

            # new position for line destination
            position[scan_order[0]] = scan_extent[1]

            # scanning the line
            slc_result[scan_order[1]] = count
            with self.translate(velocity=v, origin=self.position, destination=position) as t:
                result[tuple(slc_result)] = detector.read(nb_values=n)
                if callback is not None:
                    if not callback(result):
                        break
        return result

    def home(self):
        self.position = np.zeros(self.ndim)

    def power_down(self):
        self.home()


