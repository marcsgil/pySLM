"""
A submodule to work with homogeneous (projective) coordinates and their transformations.

TODO: Test with non-default axes.
"""
from __future__ import annotations

import numpy as np
from typing import Union, Sequence, Optional, Tuple
from scipy.spatial import transform

from optics.experimental import log

__all__ = ["HomogeneousVector", "Point", "Vector", "Transformation", "Translation", "Rotation", "Scaling"]

array_like = Union[Sequence, np.ndarray, float, int]


class HomogeneousVector:
    """
    A class to represent homogeneous vectors, or sets thereof.
    """
    def __init__(self, vector: array_like, axis: int = -1, input_axis: int = -1):  # todo: test non-default axes
        self.__working_axis = -1
        if input_axis == 0:
            vector = np.stack(np.broadcast_arrays(*vector))
        else:
            vector = np.asarray(vector)
        self.__vector = vector.swapaxes(input_axis, self.__working_axis)
        self.__axis = axis

    def copy(self) -> HomogeneousVector:
        return HomogeneousVector(self.__vector, axis=self.axis)

    @property
    def axis(self) -> int:
        """The default axis to return results as."""
        return self.__axis

    @property
    def nb_space_dims(self) -> int:
        return self.__vector.shape[self.__working_axis] - 1

    @property
    def shape(self) -> Tuple[int]:
        result = list(self.__vector.shape)
        result[self.axis], result[self.__working_axis] = result[self.__working_axis], result[self.axis]
        return tuple(result)

    def __array__(self, axis: int = None) -> np.ndarray:
        if axis is None:
            axis = self.axis
        return self.__vector.swapaxes(self.__working_axis, axis)

    def project(self) -> HomogeneousVector:
        """Rescales so that the homogeneous coordinate is either 0 or 1."""
        self.__vector /= self.__vector[..., -1:0]
        return self

    def projected(self) -> HomogeneousVector:
        return self.copy().project()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__vector})"


class Vector(HomogeneousVector):
    """
    A class to represent vectors, or sets thereof. These are homogenous vectors with the homogeneous coordinate 0.
    """
    def __init__(self, vector: array_like, axis: int = -1, input_axis: int = -1):
        vector = np.asarray(vector).swapaxes(input_axis, 0)
        super().__init__([*vector, 0], input_axis=0, axis=axis)


class Point(HomogeneousVector):
    """
    A class to represent points, or sets thereof. These are homogenous vectors with the homogeneous coordinate 1.
    """
    def __init__(self, vector: array_like, axis: int = -1, input_axis: int = -1):
        vector = np.asarray(vector).swapaxes(input_axis, 0)
        super().__init__([*vector, 1], input_axis=0, axis=axis)


class Transformation:
    """
    A class to general transformations of homogeneous coordinates.
    """
    def __init__(self, matrix: Optional[array_like], matrix_inv: Optional[array_like] = None):
        self.__working_axis = -1
        if matrix is None:
            matrix = np.inv(matrix_inv)
        self.__matrix = np.asarray(matrix)
        self.__inv = Transformation(matrix_inv) if matrix_inv is not None else None

    @property
    def nb_space_dims(self) -> int:
        return self.__matrix.shape[-1] - 1

    @property
    def shape(self) -> Tuple[int]:
        return self.__matrix.shape

    @property
    def ndim(self) -> int:
        return 2

    @property
    def inv(self) -> Transformation:
        """The inverse transformation."""
        if self.__inv is None:
            self.__inv = Transformation(np.linalg.inv(self.__matrix), matrix_inv=self.__matrix)
        return self.__inv

    def __array__(self) -> np.ndarray:
        return self.__matrix

    def __matmul__(self, other: Union[Transformation, HomogeneousVector]) -> Union[Transformation, HomogeneousVector]:
        if isinstance(other, Transformation):
            return Transformation(self.__matrix @ other)
        else:
            if not isinstance(other, HomogeneousVector):
                other = np.asarray(other)
                if other.shape[-1] == self.nb_space_dims:
                    other = Point(other, axis=-1, input_axis=-1)
                else:
                    other = HomogeneousVector(other, axis=-1, input_axis=-1)

            result = (self.__matrix @ other.__array__(axis=-1)[..., np.newaxis])[..., 0]
            return HomogeneousVector(
                result,
                input_axis=-1, axis=other.axis
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__matrix})"


class Translation(Transformation):
    """
    A class to represent translations of homogeneous coordinates. This only affects `Point`s, not `Vector`s.
    """
    def __init__(self, translation: array_like):
        translation = np.asarray(translation)
        matrix = np.eye(translation.size + 1)
        matrix[:-1, -1] = translation
        matrix_inv = np.eye(translation.size + 1)
        matrix_inv[:-1, -1] = -translation
        super().__init__(matrix, matrix_inv=matrix_inv)


class Scaling(Transformation):
    """
    A class to represent scalings of homogeneous coordinates.
    """
    def __init__(self, scaling: array_like):
        matrix = np.diag([*scaling, 1])
        matrix_inv = np.diag([*(1/_ for _ in scaling), 1])
        super().__init__(matrix, matrix_inv=matrix_inv)


class Rotation(Transformation):
    """
    A class to represent rotations of homogeneous coordinates.
    """
    @classmethod
    def from_rotvec(cls, rot_vec, degrees: bool = False) -> Rotation:
        rot_mat = transform.Rotation.from_rotvec(rot_vec, degrees=degrees).as_matrix()
        return cls(rot_mat)

    @classmethod
    def from_quat(cls, quat) -> Rotation:
        rot_mat = transform.Rotation.from_quat(quat).as_matrix()
        return cls(rot_mat)

    @classmethod
    def from_euler(cls, seq, angles, degrees: bool = False) -> Rotation:
        rot_mat = transform.Rotation.from_euler(seq, angles, degrees=degrees).as_matrix()
        return cls(rot_mat)

    @classmethod
    def from_mrp(cls, mrp) -> Rotation:
        rot_mat = transform.Rotation.from_mrp(mrp).as_matrix()
        return cls(rot_mat)

    def __init__(self, rot_mat: array_like):
        rot_mat = np.asarray(rot_mat)
        matrix = np.zeros_like(rot_mat, shape=np.asarray(rot_mat.shape) + 1)
        matrix[:-1, :-1] = rot_mat
        matrix[-1, -1] = 1
        super().__init__(matrix, matrix_inv=matrix.transpose())


if __name__ == '__main__':
    scaling = Scaling(np.full(3, 0.5))
    mirroring = Scaling(np.full(3, -1))
    translation = Translation([0, 1, 2])
    rotation = Rotation.from_rotvec(np.asarray([0, 0, 1]) * 90, degrees=True)

    log.info(rotation)
    log.info(translation)
    log.info(scaling)

    origin = Point([0, 0, 0])
    point = Point([0, 1, 0])
    vector = Vector([1, 0, 0])
    points = Point([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vectors = Vector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    transformed_points = translation.inv @ rotation @ translation @ points
    transformed_vectors = translation.inv @ rotation @ translation @ vectors

    log.info(f"transformed_points = {transformed_points}")
    log.info(f"transformed_vectors = {transformed_vectors}")
