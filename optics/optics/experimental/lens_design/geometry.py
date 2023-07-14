from . import array_like, array_type, asarray_r
import logging

log = logging.getLogger(__name__)


class Positionable:
    def __init__(self, position: array_like = asarray_r([0.0])):
        self.__position = None
        self.position = position

    @property
    def position(self) -> array_type:
        """The 3D space offset of the origin of this surface."""
        return self.__position

    @position.setter
    def position(self, new_position: array_like):
        new_position = asarray_r(new_position)
        if new_position.ndim == 0:
            new_position = asarray_r([0, 0, new_position])
        elif new_position.shape[0] == 1:
            new_position = asarray_r([0, 0, *new_position])
        self.__position = new_position

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(position={self.position})"
