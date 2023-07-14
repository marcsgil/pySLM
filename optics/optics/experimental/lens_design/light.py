from __future__ import annotations

import logging

import numpy as np
from typing import Optional
import typing

log = logging.getLogger(__name__)

from . import array_like, array_type, asarray_r, asarray_c, copy_array, einsum

import torch

from .surface import Surface


class Wavefront:
    def __init__(self, E: array_like, k0: Optional[array_like], k: Optional[array_like] = None, p: Optional[array_like] = None,
                 opd: Optional[array_like] = None, direction: Optional[array_like] = None):
        """
        Container class to represent a wavefront with a ray bundle.

        :param E: The complex electric-field polarization direction in 3D in V/m as an array of 3D vectors.
        The complex argument indicates the phase.
        :param k0: The wavenumber (= 2 pi / wavelength) for each ray in rad / meters as an array or scalars.
        If not specified, it is taken as the length of the k-vector, i.e. vacuum is assumed.
        :param k: The k-vector in the material in 3D in rad / meters as an array of 3D vectors.
        :param p: (default: origin) The position on the ray in 3D in meters as an array of 3D vectors.
        :param opd: (default: 0) The optical path difference within the bundle of rays in units of wavelength,
        as an array of scalars.
        :param direction: (optional, default k-vector) The direction of the ray in 3D in rad / meters in an anisotropic
        material.
        """
        self.__k0 = None
        self.__k = None
        self.__polarization = None
        self.__p = None
        self.__opd = None
        self.__direction = None
        self.__p_history = []

        if k0 is None:
            k0 = np.linalg.norm(k, axis=-1, keepdims=True)
        self.k0 = k0
        if k is None:
            k = self.k0 * direction
        self.k = k
        if direction is None:
            direction = self.k
        self.direction = direction
        self.E = E
        self.p = p
        if opd is None:
            opd = 0.0
        self.opd = opd

    def __asscalar_r(self, arr: array_like) -> array_type:
        # TODO: Check broadcast dimensions valid?
        arr = asarray_r(arr)
        return arr

    def __asvector_c(self, arr: array_like) -> array_type:
        # TODO: Check broadcast dimensions valid?
        arr = asarray_c(arr)
        if arr.shape[-1] != 3:
            raise ValueError('Expected vector array, not a scalar array!')
        return arr

    @property
    def k0(self) -> array_type:
        return self.__k0

    @k0.setter
    def k0(self, new_k0: array_like):
        self.__k0 = asarray_r(new_k0)

    @property
    def k(self) -> array_type:
        return self.__k

    @k.setter
    def k(self, new_k: array_like):
        self.__k = asarray_c(new_k)

    @property
    def E(self) -> array_type:
        return self.__polarization

    @E.setter
    def E(self, new_polarization: array_like):
        self.__polarization = self.__asvector_c(new_polarization)

    @property
    def p(self) -> array_type:
        return self.__p

    @p.setter
    def p(self, new_p: array_like):
        if self.__p is not None:
            self.__p_history.append(copy_array(self.__p))  # store the previous coordinates
        if new_p is None:
            new_p = np.zeros(3)
        self.__p = asarray_r(new_p)

    @property
    def p_history(self):
        return self.__p_history

    @property
    def opd(self) -> array_type:
        return self.__opd

    @opd.setter
    def opd(self, new_opd: array_like):
        self.__opd = asarray_r(new_opd)

    @property
    def direction(self) -> array_type:
        if self.__direction is not None:
            return self.__direction
        else:
            direction = self.k.real / torch.linalg.norm(self.k.real, axis=-1, keepdims=True)
            return direction

    @direction.setter
    def direction(self, new_direction: array_like):
        if new_direction is not None:
            new_direction = asarray_c(new_direction)
        self.__direction = new_direction

    @property
    def wavelength(self) -> array_type:
        return 2 * np.pi / self.k

    @wavelength.setter
    def wavelength(self, new_wavelength: array_like):
        self.k = 2 * np.pi / asarray_c(new_wavelength)

    def copy(self) -> Wavefront:
        """Return an independent copy of this Wavefront object."""
        return Wavefront(E=copy_array(self.E), k0=copy_array(self.k0), k=copy_array(self.k), p=copy_array(self.p),
                         opd=copy_array(self.opd), direction=copy_array(self.direction))

    def propagate(self, surface: Surface) -> Wavefront:
        """
        Propagates this wavefront to the given surface.

        :param: surface: The target surface to trace each ray to.
        """
        p_after = surface.intersect(self.p, self.direction)
        distance_rad = einsum('...i,...i->...', p_after - self.p + 0j, self.k)[..., np.newaxis]  # Can be complex
        distance_int = torch.round(distance_rad.real / (2 * np.pi)).to(torch.int32)  # round, not floor
        self.p = p_after
        self.opd = self.opd + distance_int
        self.E = self.E * torch.exp(1j * distance_rad)  # imaginary part accounts for extinction coefficient
        longitudinal_field = einsum('...i,...i->...', self.E, self.direction + 0j)[..., np.newaxis] * self.direction
        self.E -= longitudinal_field  # * (1.0 - np.exp(-distance / wavefront.k))  # Only keep transversal
        return self

    def plot(self, ax):
        transmitted = np.abs(self.k.detach().numpy()) > 0.0
        for p_before, p_after in zip(self.p_history, [*self.p_history[1:], self.p]):
            p_before = p_before.detach().numpy()
            p_after = p_after.detach().numpy()
            ax.scatter(p_before[..., 2], p_before[..., 1], marker='.', color=[0.0, 0.5, 0.0])
            connections = np.stack(np.broadcast_arrays(p_before, p_after), axis=-2)
            connections = connections.reshape([-1, 2, 3]).swapaxes(0, 1)
            ax.plot(connections[..., 2], connections[..., 1], color=[1.0, 0.0, 0.0], linewidth=1)
        ax.scatter(self.p[..., 2].detach().numpy(), self.p[..., 1].detach().numpy(), marker='.', color=[0.0, 1.0, 0.0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k0={self.k0}, k={self.k}, E={self.E}, p={self.p}, opd={self.opd})'
