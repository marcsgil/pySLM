import numpy as np
from typing import Callable, Union, Sequence

import logging
log = logging.getLogger(__name__)
from optics.instruments import InstrumentError, Instrument

__all__ = ['SourceError', 'Source']


class SourceError(InstrumentError):
    """
    An Exception for Source objects.
    """
    pass


class Source(Instrument):
    """
    A super class for light sources.
    """
    def __init__(self, spectrum: Callable[[Union[float, Sequence, np.ndarray]], np.ndarray] = None,
                 power_down_on_disconnect: bool = True):
        super().__init__(power_down_on_disconnect=power_down_on_disconnect)
        if spectrum is None:
            spectrum = lambda f: np.ones_like(np.asarray(f))

        self.__spectrum = spectrum
        self.__emitting = False

    @property
    def spectrum(self) -> Callable[[Union[float, Sequence, np.ndarray]], np.ndarray]:
        return self.__spectrum

    @property
    def emitting(self) -> bool:
        with self._lock:
            return self.__emitting

    @emitting.setter
    def emitting(self, new_status: bool):
        with self._lock:
            self.__emitting = new_status

    def power_down(self):
        self.emitting = False

    def __str__(self):
        return 'Light Source ' + ('' if self.emitting else 'not ') + 'emitting.'
