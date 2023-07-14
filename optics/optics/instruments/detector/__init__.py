import numpy as np
import logging

log = logging.getLogger(__name__)

from optics.instruments import Instrument


class Detector(Instrument):
    def __init__(self, dt=None, nb_samples=None):
        super().__init__()

        self.__nb_samples = nb_samples
        self.__dt = dt

    @property
    def nb_samples(self) -> int:
        return self.__nb_samples

    @nb_samples.setter
    def nb_samples(self, nb_samples):
        self.__nb_samples = nb_samples

    @property
    def dt(self) -> float:
        return self.__dt

    @dt.setter
    def dt(self, new_dt):
        self.__dt = new_dt

    def read(self, nb_values: int) -> np.ndarray:
        # Do something, wait until values arrive, return
        return np.zeros(nb_values)


