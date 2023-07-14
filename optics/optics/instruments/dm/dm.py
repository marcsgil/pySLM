from abc import abstractmethod
from typing import Union, Sequence, Callable
import numpy as np
import logging

from optics.instruments import InstrumentError, InstrumentDescriptor, Instrument
from optics.calc import zernike
from optics.utils.ft import Grid

log = logging.getLogger(__name__)


class DMError(InstrumentError):
    """
    An Exception class for Deformable Mirrors.
    """
    pass


class DMDescriptor(InstrumentDescriptor):
    def __init__(self, id: str, constructor: Callable, available: bool = True, index: int = -1, nb_actuators: int = 0):
        super().__init__(id, constructor, available)
        self.__index: int = index
        self.__nb_actuators: int = nb_actuators

    @property
    def index(self) -> int:
        """The index of the deformable mirror."""
        return self.__index

    @property
    def nb_actuators(self) -> int:
        """The number of actuators of this deformable mirror."""
        return self.__nb_actuators


class DM(Instrument):
    """
    A base class for Deformable Mirrors.
    """
    def __init__(self, radius: float = 1.0, nb_actuators: int = 1, max_stroke: float = 1.0, wavelength: float = 1.0):
        super().__init__()
        self.__nb_actuators = nb_actuators

        self.__radius = radius

        self.__modulation_vector = np.zeros(nb_actuators)  # in meters
        self.__correction_vector = np.zeros(nb_actuators)  # in meters
        self.__deflection_vector = np.zeros(nb_actuators)  # in meters

        self.__max_stroke = max_stroke

        self.__wavelength = wavelength

        nb_actuators_side = int(np.round(np.sqrt(nb_actuators * 4 / np.pi)))
        grid = Grid(np.full(2, nb_actuators_side), extent=2, center_at_index=False)
        actuator_at_indices = sum(_**2 for _ in grid) <= 1
        if np.sum(actuator_at_indices) != nb_actuators:
            raise ValueError(f'Expected {np.sum(actuator_at_indices)} instead of {nb_actuators}!')
        self.__actuator_position = np.stack([np.broadcast_to(_, grid.shape)[actuator_at_indices] for _ in grid], axis=-1)

    @property
    def wavelength(self) -> float:
        """ The wavelength in meters used to determine the stroke-phase-relationship. """
        with self._lock:
            return self.__wavelength

    @wavelength.setter
    def wavelength(self, new_wavelength: float):
        """ The wavelength in meters used to determine the stroke-phase-relationship. """
        with self._lock:
            self.__wavelength = new_wavelength

    @property
    def max_stroke(self) -> float:
        """The maximum stroke of this mirror in meters."""
        with self._lock:
            return self.__max_stroke

    @property
    def radius(self) -> float:
        """
        The radius of the aperture in meters.
        """
        with self._lock:
            return self.__radius

    @radius.setter
    def radius(self, new_radius: float):
        """
        The radius of the aperture in meters.
        """
        with self._lock:
            self.__radius = new_radius

    @property
    def actuator_position(self) -> np.array:
        """
        The positions of the actuators as an array with shape (nb_actuators, 2), in the order of the modulation,
        deflection, and correction vectors. All coordinates are normalized to the radius.
        """
        with self._lock:
            return self.__actuator_position

    def modulate(self, stroke: Union[Sequence, np.ndarray, Callable[[float, float], float]]):
        """
        Modulates the deformable mirror.

        :param stroke: The stroke to apply. It can be specified as a function in Cartesian coordinates
        or as a vector of actuator values. The actuators are positions at DM.actuator_position in the normalized pupil.
        """
        with self._lock:
            self.modulation = stroke

    @property
    def modulation(self) -> np.ndarray:
        """
        The target stroke of the mirror in meters as a vector per actuator, not including deflection and correction.
        """
        with self._lock:
            return self.__modulation_vector

    @modulation.setter
    def modulation(self, stroke: Union[Sequence, np.ndarray, Callable[[float, float], float]]):
        """ Sets the target stroke of the mirror (in meters?), not including deflection and correction. """
        with self._lock:
            self.__modulation_vector = self.__as_vector(stroke)
            # print(f'modulation={self.modulation}, deflection={self.deflection}, correction={self.correction}, actual={self.actual_stroke_vector}')
            self._modulate(self.actual_stroke_vector)  # Delegate implementation to sub-class

    @abstractmethod
    def _modulate(self, actual_stroke_vector: np.ndarray):
        pass  # Override in sub-class

    @property
    def correction(self) -> np.ndarray:
        """ The correction vector in meters. """
        with self._lock:
            return self.__correction_vector

    @correction.setter
    def correction(self, new_correction: Union[Sequence, np.ndarray, Callable[[float, float], float]]):
        """ The correction in meters. """
        with self._lock:
            self.__correction_vector = self.__as_vector(new_correction)

    @property
    def deflection(self):
        """ This is the additional correction in meters that we can manually adjust (e.g. tip, tilt, and defocus) """
        with self._lock:
            return self.__deflection_vector

    @deflection.setter
    def deflection(self, new_deflection: Union[Sequence, np.ndarray, Callable[[float, float], float]]):
        """ This is the additional correction that we can manually adjust (e.g. tip, tilt, and defocus). """
        with self._lock:
            self.__deflection_vector = self.__as_vector(new_deflection)

    @property
    def actual_stroke_vector(self) -> np.ndarray:
        """ The actual stroke of the mirror, including deflection and correction [in meters]."""
        return np.clip(self.__deflection_vector + self.__correction_vector + self.__modulation_vector,
                       -self.max_stroke, self.max_stroke)

    #
    # Methods to simulate and display
    #
    def field(self, *ranges, actual: bool=False) -> np.array:
        """
        The field modulation caused by the mirror at interpolated points.
        Values outside of the pupil are 0.

        :param ranges: [nu_y, nu_x], the coordinates to sample the phase at, normalized to the pupil radius.
        :param actual: bool indicating whether the target or the actual field is returned.
        :return: The phase as an ndarray in nu_y, nu_x
        """
        with self._lock:
            phases = self.phase(*ranges, actual=actual)
            outside = np.isnan(phases)
            phases[outside] = 0.0

            return np.exp(1j * phases) * (1 - outside)

    def phase(self, *ranges, actual: bool=False) -> np.array:
        """
        The mirror phase in radians at interpolated points.
        Values outside of the pupil are indicated by not-a-number (np.nan).

        :param ranges: [nu_y, nu_x], the coordinates to sample the phase at, normalized to the pupil radius.
        :param actual: bool indicating whether the target or the actual phase is returned.
        :return: The phase as an ndarray in nu_y, nu_x
        """
        with self._lock:
            return self.stroke(*ranges, actual=actual) * (2 * np.pi / self.wavelength)

    def stroke(self, *ranges, actual: bool = False) -> np.array:
        """
        The target mirror stroke in meters at interpolated points.
        Values outside of the pupil are indicated by not-a-number (np.nan).

        :param ranges: [nu_y, nu_x], the coordinates to sample the stroke at coordinates normalized to the pupil radius.
        :param actual: bool indicating whether the target or the actual stroke is returned.
        :return: The stroke as an ndarray in nu_y, nu_x in units of meters.
        """
        modulation = self.actual_stroke_vector if actual else self.modulation
        with self._lock:
            zernike_fit = zernike.fit(
                modulation / self.wavelength, self.__actuator_position[:, 0], self.__actuator_position[:, 1],
                order=self.__nb_actuators)
            # print(f'Fitting error: {zernike_fit.error}. coeff = {zernike_fit.coefficients}')
            zernike_fit *= self.wavelength

            result = zernike_fit.cartesian(*ranges)

            # block everything outside of the deformable mirror
            result[(ranges[0]**2 + ranges[1]**2) > 1] = np.nan

            return result

    #
    # Private method
    #
    def __as_vector(self, other) -> np.ndarray:
        """
        Converts inputs to vectors of target or phase values.

        :param other: An array or vector with one floating point value per actuator in the order of the actuators, or
            a function that will be evaluated at the Cartesian coordinates of the actuators.
        :return: A vector of values.
        """
        if isinstance(other, Callable):
            nu_y = self.__actuator_position[:, 0]
            nu_x = self.__actuator_position[:, 1]
            other = other(nu_y, nu_x)
        else:
            other = np.asarray(other).ravel()
        if np.isscalar(other):
            other = np.full(self.__nb_actuators, other)
        elif other.size != self.__nb_actuators:
            raise DMError(f'The number of actuators is {self.__nb_actuators}, instead {other.size} were received.')

        return other.real.astype(float)
