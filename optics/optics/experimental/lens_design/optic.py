from __future__ import annotations
import logging
from . import array_like, array_type, asarray_r, asarray_c, asnumpy, Param
import torch

from abc import abstractmethod
from typing import Union, Callable, Sequence, Optional
from numbers import Complex, Real
from dataclasses import dataclass

import numpy as np

from .material import Material, vacuum
from .surface import Surface
from .light import Wavefront
from .geometry import Positionable

log = logging.getLogger(__name__)


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Optic(Positionable):
    """Represents an abstract Lens."""
    surfaces: Sequence[Surface]
    diameter: Real = np.inf

    @property
    def front_surface(self) -> Surface:
        return self.surfaces[0]

    @property
    def rear_surface(self) -> Surface:
        return self.surfaces[-1]

    @property
    def thickness(self) -> Real:
        return self.rear_surface.position[2] - self.front_surface.position[2]

    @property
    def parameters(self) -> Sequence[Param]:
        raise NotImplementedError

    @abstractmethod
    def propagate(self, wavefront: Wavefront, exit_material: Optional[Material] = None) -> Wavefront:
        """
        Traces the wavefront to each Surface in sequence. The input argument will be destroyed by this method.

        :param: wavefront: The wavefront before reaching the front surface.
        Note that this input argument will be destroyed and should not be reused.
        :param: exit_material: (optional) The Material to refract into after the last Surface. If None is specified,
        the wavefront propagation stops at the last surface.

        :return: The wavefront at the rear surface, right after refraction when exit_material is specified, or right
        before if None is specified.
        """
        raise NotImplementedError

    def plot(self, ax, color=(0, 0, 0)):
        raise NotImplementedError(f'for class {type(self)}')


class MultipletLens(Optic):
    """Class to represent a lens, consisting of a sequence of N Surfaces separated by N-1 Materials."""
    def __init__(self, surfaces: Sequence[Surface], materials: Sequence[Material],
                 thicknesses: Optional[Sequence[Real]] = None, diameter: Real = None):
        """
        Constructs a lens.

        :param: surfaces: The surfaces separating the materials from front to rear.
        :param: materials: The sequence of materials. This should be 1 lens than the number of surfaces.
        :param: thicknesses: (optional) The thickness of each material. This should equal the number of materials.
        If thicknesses is specified, the surfaces' offset is overwritten so that the first surface is at the origin, and
        the following surfaces are separated by the given thickness.
        :param: diameter: (optional) The maximum aperture diameter.
        """
        if thicknesses is not None:
            offsets = np.cumsum(thicknesses)
            for surface, offset in zip(surfaces[1:], offsets):
                surface.position = offset
        if diameter is not None:
            for surface in surfaces:
                surface.diameter = diameter
        super().__init__(surfaces)
        self.__materials = materials
        self.__position = torch.zeros(3)

    @property
    def materials(self) -> Sequence[Material]:
        return self.__materials

    @materials.setter
    def materials(self, new_materials: Sequence[Material]):
        self.__materials = new_materials

    @property
    def thicknesses(self) -> Sequence[Real]:
        return np.diff([_.position[2] for _ in self.surfaces])

    @property
    def position(self) -> array_type:
        return self.__position

    @position.setter
    def position(self, new_position: array_like):
        new_position = asarray_r(new_position)
        if new_position.ndim == 0:
            new_position = asarray_r([0, 0, new_position])
        elif new_position.shape[0] == 1:
            new_position = asarray_r([0, 0, *new_position])
        translation = new_position - self.position
        for _ in self.surfaces:
            _.position += translation
        self.__position = new_position

    @property
    def parameters(self) -> Sequence[Param]:
        return sum([list(_.parameters) for _ in self.surfaces], [])

    def propagate(self, wavefront: Wavefront, exit_material: Optional[Material] = None) -> Wavefront:
        """
        Traces the wavefront to each Surface in sequence. The input argument will be destroyed by this method.

        :param: wavefront: The wavefront before reaching the front surface.
        Note that this input argument will be destroyed and should not be reused.
        :param: exit_material: (optional) The Material to refract into after the last Surface. If None is specified,
        the wavefront propagation stops at the last surface.

        :return: The wavefront at the rear surface, right after refraction when exit_material is specified, or right
        before if None is specified.
        """
        for surface, material in zip(self.surfaces, [*self.materials, exit_material]):
            # Propagate in same material to the next surface
            wavefront = wavefront.propagate(surface)
            if material is not None:
                # Refract at surface into the next material
                k_before = wavefront.k
                normal = surface.normal(wavefront.p) + 0j  # Make complex for later projections on complex vectors
                k_transverse = k_before - torch.einsum('...i,...i->...', k_before, normal)[..., np.newaxis] * normal
                k_transverse_norm2 = torch.einsum('...i', torch.abs(k_transverse)**2)
                k_after_norm = material.k(vacuum_wavenumber=wavefront.k0)
                k_normal_norm2 = k_after_norm**2 - k_transverse_norm2
                non_propagating = k_normal_norm2 <= 0.0
                normal_E = torch.einsum('...i,...i->...', wavefront.E, normal)[..., np.newaxis] * normal
                wavefront.E -= normal_E
                wavefront.E = torch.logical_not(non_propagating) * wavefront.E  # make sure to broadcast
                k_normal_norm2[non_propagating] = 0.0
                wavefront.k = k_transverse + normal * torch.sign(k_after_norm) * (k_normal_norm2**0.5)
                wavefront.direction = None  # Assume material isotropic, so wavefront.direction is the same as wavefront.k

        return wavefront

    def plot(self, ax, color=(0.5, 0.0, 0.0)):
        # plot surfaces
        for _ in self.surfaces:
            _.plot(ax, color=color)
        # plot boundaries
        for before, material, after in zip(self.surfaces[:-1], self.materials, self.surfaces[1:]):
            if material != vacuum:
                pos = [[surface.intersect(position=asnumpy(surface.position) + [0.0, y, 0.0], direction=[0, 0, 1]).detach().numpy()
                        for y in (-surface.diameter / 2.0, surface.diameter / 2.0)]
                       for surface in (before, after)]
                pos = np.asarray(pos)
                ax.plot(pos[:, :, 2], pos[:, :, 1], color=color, linewidth=2)

    def __reversed__(self) -> MultipletLens:
        return MultipletLens(surfaces=self.surfaces[::-1], materials=self.materials)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(thicknesses={self.thicknesses}, materials={self.materials}, surfaces={self.surfaces})'


class SingletLens(MultipletLens):
    def __init__(self, material: Material, front_surface: Surface = None, rear_surface: Surface = None,
                 thickness: Optional[Real] = None, diameter: Real = None):
        thicknesses = [thickness] if thickness is not None else None
        super().__init__(surfaces=[front_surface, rear_surface], materials=[material],
                         thicknesses=thicknesses, diameter=diameter)

    @property
    def material(self) -> Material:
        return self.materials[0]

    def __reversed__(self) -> SingletLens:
        return SingletLens(thickness=self.thickness, material=self.material,
                           front_surface=self.front_surface, rear_surface=self.rear_surface)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(thickness={self.thickness}, material={self.material}, front_surface={self.front_surface}, rear_surface={self.rear_surface})'


class CompoundOptic(Optic):
    """Represents an optical component with one or more (separated) lenses."""
    def __init__(self, optics_and_materials: Sequence[Union[Optic, Material]] = None,
                 optics: Sequence[Optic] = None, materials: Sequence[Material] = None,
                 diameter: Real = None):
        if optics_and_materials is not None:
            optics = [_ for _ in optics_and_materials if isinstance(_, Optic)]
            materials = [_ for _ in optics_and_materials if isinstance(_, Material)]
            materials += [vacuum] * (len(optics) - len(materials) - 1)
        if len(materials) >= len(optics):
            raise ValueError(f'The number of Materials ({len(materials)}) should be less than the number of Optics ({len(optics)}).')
        super().__init__(sum((_.surfaces for _ in optics), []))

        if diameter is not None:  # Enforce diameter constraint on all child-optics
            for _ in optics:
                _.diameter = diameter

        self.__optics = optics
        self.__materials = materials

    @property
    def optics(self) -> Sequence[Optic]:
        return self.__optics

    @property
    def materials(self) -> Sequence[Material]:
        return self.__materials

    @property
    def parameters(self) -> Sequence[Param]:
        return sum((_.parameters for _ in self.optics), [])

    def propagate(self, wavefront: Wavefront, exit_material: Optional[Material] = None) -> Wavefront:
        """
        Traces the wavefront to each Surface in sequence. The input argument will be destroyed by this method.

        :param: wavefront: The wavefront before reaching the front surface.
        Note that this input argument will be destroyed and should not be reused.
        :param: exit_material: (optional) The Material to refract into after the last Surface. If None is specified,
        the wavefront propagation stops at the last surface.

        :return: The wavefront at the rear surface, right after refraction when exit_material is specified, or right
        before if None is specified.
        """
        for optic, material in zip(self.optics, [*self.materials, exit_material]):
            wavefront = optic.propagate(wavefront, exit_material=material)

        return wavefront

    def plot(self, ax, color=(0, 0, 0)):
        for optic in self.optics:
            optic.plot(ax, color=color)


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class OpticalSystem:
    source: Wavefront
    optic: Optic
    detector: Surface
    detector_material: Material = vacuum
    metric: Callable[[OpticalSystem], array_type] = lambda _: asarray_c(0.0)

    def propagate(self, wavefront: Optional[Wavefront] = None) -> Wavefront:
        """
        Traces the wavefront from the source to the detector surface.
        The input argument will be destroyed by this method.

        :param wavefront: The wavefront in source space.
        Note that this input argument will be destroyed and should not be reused.

        :return: The wavefront at the detector surface (not out-refracted).
        """
        if wavefront is None:
            wavefront = self.source.copy()
        wavefront = self.optic.propagate(wavefront, exit_material=self.detector_material)
        wavefront = wavefront.propagate(self.detector)

        return wavefront

    @property
    def parameters(self) -> Sequence[Param]:
        return (*self.optic.parameters, self.detector.parameters)

    def evaluate(self) -> array_type:
        """Returns the value of the cost function, which can be an array of values."""
        return self.metric(self)

    def optimize(self) -> OpticalSystem:
        learning_rate = 0.01
        for iteration in range(10):
            # Calculate cost and update gradients for each parameter
            cost = self.evaluate()
            cost.backward()
            for p in self.parameters:
                p -= learning_rate * p.grad
        return self

    def plot(self, ax, color=(0, 0, 0)):
        self.optic.plot(ax, color=color)
        self.detector.plot(ax, color=[1, 0, 1])

