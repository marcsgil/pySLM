from __future__ import annotations
import numpy as np
import scipy.sparse.linalg as spa
import logging
from optics.utils import ft

log = logging.getLogger(__name__)


__all__ = ['GridOperator', 'Operator', 'HermitianOperator', 'SkewHermitianOperator']


class Operator(spa.LinearOperator):
    """A class to represent linear operators that keep track of the number of evaluations."""
    def __init__(self, nb_modes: int, matvec=None, rmatvec=None):
        """
        Create a new linear operator to work with gridded data.

        :param nb_modes: The number of rows and columns in the equivalent matrix.
        :param matvec: The function that performs the self @ vector operation. Default: None, identity.
        :param rmatvec: The function that performs the vector @ self operation. Default: None, same as matvec.
        """
        super().__init__(dtype=np.complex128, shape=(nb_modes, nb_modes))
        if matvec is None:
            matvec = lambda _: _
        self.__matvec_function = matvec
        self.__rmatvec_function = rmatvec if rmatvec is not None else matvec

    def _matvec(self, _):
        return self.__matvec_function(_)

    def _rmatvec(self, _):
        return self.__rmatvec_function(_)

    def _adjoint(self) -> Operator:
        return Operator(self.shape[1], self.__rmatvec_function, self.__matvec_function)

    # Diagnostic helper functions that calculate the L2 operator norm, condition number, and the infimum of the real part of Hermitian operators.
    def cond(self) -> float:
        """The condition number of this operator."""
        return self.norm() * self.norm_inv()

    def norm(self) -> float:
        """This operator's l2-norm."""
        return spa.svds(self, 1, which='LM')[1][0]

    def norm_inv(self) -> float:
        """The l2-norm of the inverse of this operator."""
        return 1.0 / spa.svds(self, 1, which='SM')[1][0]

    def real(self) -> float:
        """The infimum of Re <x, self @ x> / <x, x>"""
        return spa.eigsh(self.adjoint() @ self, 1, which='SR')[0][0] / 2


class HermitianOperator(Operator):
    """A class to allow use to represent functions as Hermitian matrices."""
    def __init__(self, nb_modes: int, matvec=None):
        """
        Create a new linear operator that is Hermitian to work with gridded data.

        :param nb_modes: The number of rows and columns in the equivalent matrix.
        :param matvec: The function that performs the self @ vector operation. Default: None, identity.
        """
        super().__init__(nb_modes=nb_modes, matvec=matvec)

    def _adjoint(self) -> HermitianOperator:
        """The adjoint operator, i.e. its Hermitian transpose."""
        return self

    def norm(self) -> float:
        """This operator's l2-norm."""
        return np.abs(spa.eigsh(self, 1, which='LM')[0][0])

    def norm_inv(self) -> float:
        """The l2-norm of the inverse of this operator."""
        return 1.0 / np.abs(spa.eigsh(self, 1, which='SM')[0][0])

    def real(self) -> float:
        """The infimum of Re <x, self @ x> / <x, x>"""
        return spa.eigsh(self, 1, which='SR')[0][0]


class SkewHermitianOperator(Operator):
    """A class to allow use to represent functions as Hermitian matrices."""
    def __init__(self, nb_modes: int, matvec=None):
        """
        Create a new linear operator that is Anti-Hermitian to work with gridded data.

        :param nb_modes: The number of rows and columns in the equivalent matrix.
        :param matvec: The function that performs the self @ vector operation. Default: None, identity.
        """
        super().__init__(nb_modes=nb_modes, matvec=matvec, rmatvec=lambda _: matvec(-_))

    def _adjoint(self) -> HermitianOperator:
        """The adjoint operator, i.e. its Hermitian transpose."""
        return -self

    def norm(self) -> float:
        """This operator's l2-norm."""
        return np.abs(spa.eigsh(1j * self, 1, which='LM')[0][0])

    def norm_inv(self) -> float:
        """The l2-norm of the inverse of this operator."""
        return 1.0 / np.abs(spa.eigsh(1j * self, 1, which='SM')[0][0])

    def real(self) -> float:
        """The infimum of Re <x, self @ x> / <x, x>"""
        return 0.0


class GridOperator:
    def __init__(self, grid: ft.Grid):
        self.__grid = grid

        # Define some useful operators
        self.__identity = self.new_Hermitian()  # The identity operator
        self.__fourier_transform = self.new(matvec=lambda _: self.a2v(ft.fftn(self.v2a(_))), rmatvec=lambda _: self.a2v(ft.ifftn(self.v2a(_))))  # Forward Discrete Fourier Transform
        self.__inverse_fourier_transform = self.__fourier_transform.adjoint()  # The inverse Fourier transform
        
    @property
    def grid(self) -> ft.Grid:
        return self.__grid.immutable

    def a2v(self, _):
        """Maps from n-dimensional 'fields' to vectors."""
        return _.ravel()

    def v2a(self, _):
        """Maps from vectors to n-dimensional 'fields'."""
        return _.reshape(self.grid.shape)

    def new(self, matvec=None, rmatvec=None) -> Operator:
        """
        Create a new linear operator to work with gridded data.

        :param matvec: The function that performs the self @ vector operation. Default: None, identity.
        :param rmatvec: The function that performs the vector @ self operation. Default: None, Hermitian assumed.
        """
        if rmatvec is None:
            return self.new_Hermitian(matvec=matvec)
        else:
            return Operator(nb_modes=self.grid.size, matvec=matvec, rmatvec=rmatvec)

    def new_Hermitian(self, matvec=None) -> HermitianOperator:
        """
        Create a new Hermitian operator for gridded data.

        :param matvec: The function that performs the self @ vector operation. Default: None, identity.
        """
        return HermitianOperator(nb_modes=self.grid.size, matvec=matvec)

    @property
    def identity(self) -> HermitianOperator:
        """The identity operator."""
        return self.__identity

    @property
    def ft(self) -> Operator:
        """The Fourier transform operator."""
        return self.__fourier_transform

    @property
    def ift(self) -> Operator:
        """The inverse Fourier transform operator."""
        return self.__inverse_fourier_transform

    @property
    def laplacian(self) -> Operator:
        """The Laplacian operator, negative sum of second derivatives, calculated in Fourier space."""
        k2 = sum(_**2 for _ in self.grid.k)
        return self.ift @ self.new_Hermitian(lambda _: self.a2v(-k2) * self.a2v(_)) @ self.ft
