"""
Script to calculate the (number of) wavelengths in a diode laser cavity.
"""
import scipy.constants as const
import numpy as np

from optics import log


if __name__ == '__main__':
    opd = 1.2e-3
    tau = opd / const.c
    delta_f = 1 / tau

    center_wavelength = 488e-9
    band_width = 5e-9
    wavelength_limits = center_wavelength + np.array([-1, 1]) * 0.5 * band_width

    frequency = lambda w: const.c / w
    wavelength = lambda f: const.c / f

    frequency_limits = frequency(wavelength_limits[::-1])
    frequencies = np.arange(frequency_limits[0], frequency_limits[1], delta_f)
    wavelengths = wavelength(frequencies[::-1])

    print(wavelengths * 1e9)
    delta_wavelength = (wavelengths[-1] - wavelengths[0]) / (wavelengths.size - 1)

    log.info(f'Number of wavelengths in cavity = {wavelengths.size}, difference between wavelengths = {delta_wavelength * 1e9:0.6f} nm ')
