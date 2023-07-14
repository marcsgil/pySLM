from __future__ import annotations

from typing import Union, Callable, Sequence
from numbers import Complex, Real

import numpy as np
import scipy.constants as const
import logging

from .light import Wavefront
from .surface import Surface

from . import array_type, array_like, asarray_r, asarray_c
import torch

log = logging.getLogger(__name__)


class Material:
    def __init__(self, refractive_index: Union[Complex, Callable[[Real], Complex]]):
        """
        Represents a material with a homogeneous isotropic refractive index. The refractive index can be complex to
        account for absorption or gain.

        :param refractive_index: A function of vacuum wavenumber, k0, or a scalar, indicating a constant function.
        """
        super().__init__()
        self.__refractive_index__description = refractive_index.__repr__()
        if not callable(refractive_index):
            self.__refractive_index = lambda _: torch.ones_like(asarray_c(_)) * refractive_index
        else:
            self.__refractive_index = refractive_index

    def n(self, vacuum_wavenumber: array_like = None, angular_frequency: array_like = None,
          frequency: array_like = None, vacuum_wavelength: array_like = None) -> array_type:
        """
        Returns the complex refractive index as a function of one of the following input arguments. The real refractive
        index can be obtained as the property .real and the extinction coefficient, kappa, can be obtained as the .imag property.

        :param vacuum_wavenumber: (optional) The wavenumber in rad/m, or an ndarray thereof.
        :param angular_frequency: (optiona) The angular frequency in rad/s, or an ndarray thereof.
        :param frequency: (optiona) The frequency in $s^{-1}$, or an ndarray thereof.
        :param vacuum_wavelength: (optiona) The wavelength in vacuum, or an ndarray thereof.
        All input arguments can be a real number, a sequence thereof, or an ndarray of such numbers.
        :return: The complex refractive indices for each scalar in the input argument, return as an numpy.ndarray of
        the same shape as the input argument.
        """
        if vacuum_wavenumber is None:
            if angular_frequency is None:
                if frequency is None:
                    frequency = const.c / vacuum_wavelength
                angular_frequency = 2 * np.pi * frequency
            vacuum_wavenumber = const.c * angular_frequency
        # log.info(f'k0 = {vacuum_wavenumber}')
        k0 = asarray_r(vacuum_wavenumber)
        k0 = torch.atleast_1d(k0)
        return self.__refractive_index(k0)

    def extinction_coefficient(self, vacuum_wavenumber: array_like = None, angular_frequency: array_like = None,
                               frequency: array_like = None, vacuum_wavelength: array_like = None) -> array_type:
        return self.n(vacuum_wavenumber=vacuum_wavenumber, angular_frequency=angular_frequency, frequency=frequency,
                      vacuum_wavelength=vacuum_wavelength).imag

    def k(self, vacuum_wavenumber: array_like = None, angular_frequency: array_like = None,
          frequency: array_like = None, vacuum_wavelength: array_like = None) -> array_type:
        """
        Returns the wavenumber in the medium as a function of one of the following input arguments.

        :param vacuum_wavenumber: (optional) The wavenumber in rad/m, or an ndarray thereof.
        :param angular_frequency: (optiona) The angular frequency in rad/s, or an ndarray thereof.
        :param frequency: (optiona) The frequency in $s^{-1}$, or an ndarray thereof.
        :param vacuum_wavelength: (optiona) The wavelength in vacuum, or an ndarray thereof.
        All input arguments can be a real number, a sequence thereof, or an ndarray of such numbers.
        :return: The wavenumbers for each scalar in the input argument, return as an numpy.ndarray of
        the same shape as the input argument.
        """
        if vacuum_wavenumber is None:
            if angular_frequency is None:
                if frequency is None:
                    frequency = const.c / vacuum_wavelength
                angular_frequency = 2 * np.pi * frequency
            vacuum_wavenumber = const.c * angular_frequency
        return vacuum_wavenumber * self.n(vacuum_wavenumber=vacuum_wavenumber).real

    def propagate(self, wavefront: Wavefront, surface: Surface) -> Wavefront:
        """
        Propagate a wavefront in this material to the given surface.

        :param wavefront: The wavefront to propagate, must be in this material.
        :param surface: The surface to propagate to.
        :return: The wavefront at the surface, prior to refraction.
        """
        wavefront = wavefront.propagate(surface=surface)   # propagate in isotropic material
        return wavefront

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(refractive_index={self.__refractive_index__description})"


vacuum = Material(refractive_index=1.0)

