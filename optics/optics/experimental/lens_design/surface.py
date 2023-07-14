from . import array_type, array_like, asarray_r, asnumpy, einsum, norm, Param
import torch

from typing import Sequence, Optional
from numbers import Real
import numpy as np
import logging

from .geometry import Positionable

log = logging.getLogger(__name__)


class Surface(Positionable):
    def __init__(self, diameter: Real = None, position: array_like = 0.0):
        super().__init__(position=position)
        self.__diameter = None
        self.diameter = diameter

    @property
    def diameter(self) -> Real:
        """The diameter outwith no light is admitted."""
        return self.__diameter

    @diameter.setter
    def diameter(self, new_diameter: Real):
        if new_diameter is not None:  # Only set when not None
            self.__diameter = new_diameter

    @property
    def parameters(self) -> Sequence[Param]:
        raise NotImplementedError

    def normal(self, position: array_like) -> array_like:
        """
        The normal of a point on the surface in the forward direction
        :param: position: The position on the surface for which the normal is to be calculated.

        :return The normal as a normalized vector that points in the forward direction, z.
        """
        raise NotImplementedError

    def intersect(self, position: array_like, direction: array_like) -> array_like:
        """The normal of a point on the surface in the forward direction
        :param: position: The position on each ray.
        :param: direction: The direction of the ray.

        :return The position of each intersection.
        """
        raise NotImplementedError

    def plot(self, ax, color=(0, 0, 0)):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(diameter={self.diameter},position={self.position})"


class SphericalSurface(Surface):
    def __init__(self, curvature: Real = None, radius_of_curvature: Real = None, diameter: Real = None,
                 position: array_like = 0.0):
        self.__curvature = None
        self.__diameter = None
        if curvature is None:
            curvature = 1 / radius_of_curvature
        self.curvature = curvature
        self.diameter = 2.0 * np.abs(self.radius_of_curvature)
        super().__init__(diameter=diameter, position=position)

    @property
    def curvature(self) -> array_type:
        """
        The curvature, the reciprocal of the radius of curvature.
        <0: concave, the center of the sphere lies on the negative side of the axis.
         0: planar, the center of the sphere lies at infinity.
        >0: convex, the center of the sphere lies on the positive side of the axis.
        """
        return self.__curvature

    @curvature.setter
    def curvature(self, new_curvature: Real):
        self.__curvature = asarray_r(new_curvature)

    @property
    def radius_of_curvature(self) -> array_type:
        """
        The radius of curvature, the reciprocal of the curvature.
            <0: concave, the center of the sphere lies on the negative side of the axis.
        np.inf: planar, the center of the sphere lies at infinity.
            >0: convex, the center of the sphere lies on the positive side of the axis.
        """
        return 1.0 / self.curvature if self.curvature != 0.0 else np.inf

    @radius_of_curvature.setter
    def radius_of_curvature(self, new_radius_of_curvature: Real):
        self.curvature = 1 / new_radius_of_curvature

    @property
    def diameter(self) -> array_type:
        """The diameter outwith no light is admitted."""
        return self.__diameter

    @diameter.setter
    def diameter(self, new_diameter: Real):
        if new_diameter is not None:
            max_diameter = 2.0 * np.abs(self.radius_of_curvature)
            self.__diameter = asarray_r(min(new_diameter, max_diameter))

    @property
    def parameters(self) -> Sequence[Param]:
        return Param(self.curvature), Param(self.position), Param(self.diameter)

    def normal(self, position: array_like) -> array_like:
        """
        The normal of a point on the surface in the forward direction
        :param: position: The position on the surface for which the normal is to be calculated.

        :return The normal as a normalized vector that points in the forward direction, z.
        """
        if self.curvature == 0.0:
            normal = asarray_r([0.0, 0.0, 1.0])
        else:
            sphere_center = asarray_r([0.0, 0.0, self.radius_of_curvature])
            normal = sphere_center + self.position - position
            normal *= torch.sign(self.curvature) / torch.linalg.norm(normal, axis=-1, keepdims=True)
        return normal

    def intersect(self, position: array_like, direction: array_like) -> array_like:
        """The normal of a point on the surface in the forward direction
        :param: position: The position on each ray.
        :param: direction: The direction of the ray.

        :return The position of each intersection.
        """
        position = asarray_r(position)
        direction = asarray_r(direction)
        if self.curvature == 0.0:
            distance = self.position[..., 2:] - position[..., 2:]
            intersection = position + distance * direction / direction[..., 2:]
        else:
            # ||d||^2 t^2 + 2<p|d> t + ||p||^2 - r^2 == 0 with ||d||=1
            sphere_center = self.position + asarray_r([0, 0, self.radius_of_curvature])
            relative_position = position - sphere_center
            b = einsum('...i,...i->...', relative_position, direction) * 2
            c = einsum('...i,...i->...', relative_position, relative_position) - self.radius_of_curvature ** 2
            d = asarray_r(b ** 2 - 4 * c)
            non_intersecting = d <= 0
            d[non_intersecting] = 0.0
            intersection_side = -torch.sign(self.curvature) * torch.sign(direction[..., 2])
            t = 0.5 * (-asarray_r(b) + intersection_side * (d ** 0.5))
            intersection = position + t[..., np.newaxis] * direction
        return intersection

    def plot(self, ax, color=(0, 0, 0)):
        diameter = asnumpy(self.diameter) if not np.isinf(self.diameter) else 1.0
        nb_subdivs = 256
        pos = np.zeros([nb_subdivs, 3])
        pos[..., 1] = (np.arange(nb_subdivs) - (nb_subdivs - 1) / 2) * diameter / (nb_subdivs - 1)
        pos += asnumpy(self.position)
        surface_pos = asnumpy(self.intersect(position=pos, direction=[0, 0, 1]))
        lines = ax.plot(surface_pos[..., 2], surface_pos[..., 1], color=color, linewidth=2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(curvature={self.curvature}, diameter={self.diameter}, position={self.position})"


class PlanarSurface(SphericalSurface):
    def __init__(self, diameter: Real = None, position: array_like = 0.0):
        super().__init__(curvature=0.0, diameter=diameter, position=position)
