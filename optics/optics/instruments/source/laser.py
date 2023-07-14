import numpy as np
from typing import Union, Optional, Callable, Sequence
import scipy.constants as const

import logging
log = logging.getLogger(__name__)
from optics.instruments.source import Source


class Laser(Source):
    def __init__(self, max_power: float = 1.0, power_down_on_disconnect: bool = True,
                 frequency: Optional[float] = None, frequency_bandwidth_fwhm: Optional[float] = None,
                 wavelength: Optional[float] = None, wavelength_bandwidth_fwhm: Optional[float] = None,
                 spectrum: Callable[[Union[float, Sequence, np.ndarray]], np.ndarray] = None):
        # Initialize the super class, even though we don't know the spectrum yet.
        super().__init__(power_down_on_disconnect=power_down_on_disconnect)

        # Convert everything to frequencies [Hz]
        if frequency is None:
            if wavelength is None:
                wavelength = 1000e-9
            frequency = self.wavelength2frequency(wavelength)
        if frequency_bandwidth_fwhm is None:
            if wavelength_bandwidth_fwhm is None:
                frequency_bandwidth_fwhm = 1e-6 * frequency
            else:
                frequency_bandwidth_fwhm = np.diff(self.wavelength2frequency(
                    self.frequency2wavelength(frequency) + np.array([-0.5, 0.5]) * wavelength_bandwidth_fwhm))
        if spectrum is None:
            # Assume a Gaussian spectrum
            standard_deviation = frequency_bandwidth_fwhm / np.sqrt(8 * np.log(2))
            peak_power = max_power / np.sqrt(2 * np.pi)

            def spectrum(f: Union[float, Sequence, np.ndarray]):
                return peak_power * np.exp(-0.5 * ((np.asarray(f) - frequency) / standard_deviation) ** 2)
        else:
            frequency = 1.0

        self.__max_power = max_power
        self.__power = self.__max_power  # If the power cannot be changed, it will be at the maximum power.
        self.__frequency = frequency

        # Call it again, now with the correct spectrum
        super().__init__(spectrum=spectrum, power_down_on_disconnect=power_down_on_disconnect)

    @property
    def max_power(self) -> float:
        return self.__max_power

    @property
    def power(self) -> float:
        return self.__power

    @property
    def frequency(self) -> float:
        return self.__frequency

    @property
    def wavelength(self) -> float:
        return float(self.frequency2wavelength(self.__frequency))

    @staticmethod
    def wavelength2frequency(wavelength: Union[float, Sequence, np.ndarray]) -> np.ndarray:
        return const.c / np.asarray(wavelength)

    @staticmethod
    def frequency2wavelength(frequency: Union[float, Sequence, np.ndarray]) -> np.ndarray:
        return const.c / np.asarray(frequency)

    def __str__(self):
        return 'Laser ' + ('' if self.emitting else 'not ') \
               + f'emitting {self.power*1e3:0.3f} mW at {self.wavelength*1e9:0.0f} nm.'


class SimulatedLaser(Laser):
    def __init__(self, max_power: float = np.inf, wavelength: float = 1.0e-6, wavelength_bandwidth_fwhm: float = 1.0e-9,
                 power_down_on_disconnect: bool = True):

        self.__power = max_power
        # Initialize the super class correctly.
        super().__init__(max_power=max_power,
                         wavelength=wavelength, wavelength_bandwidth_fwhm=wavelength_bandwidth_fwhm,
                         power_down_on_disconnect=power_down_on_disconnect)

    @property
    def power(self):
        return self.__power

    @power.setter
    def power(self, new_power: float):
        self.__power = np.clip(new_power, 0.0, self.max_power)

    def __str__(self):
        return f'SimulatedLaser ' + \
               f'wavelength {self.wavelength * 1e9:0.1f} nm and maximum power {self.max_power*1e3:0.0f} mW. ' + \
               f'Currently ' + ('' if self.emitting else 'not ') + f'emitting at {self.power*1e3:0.1f} mW.'

