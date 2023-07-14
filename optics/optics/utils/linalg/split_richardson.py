from __future__ import annotations

import numpy as np
from typing import Callable, Union, Sequence
from numbers import Real, Complex
import logging

from optics.utils import ft

log = logging.getLogger(__name__)

array_like = Union[Sequence, Complex, np.ndarray]


class Solution:
    """
    A class to represent a(n intermediate) solution to a linear problem solved with the split-Richardson method.
    """
    def __init__(self, l_plus_identity_inv: Callable = None, identity_minus_v: Callable = None, b: array_like = None,
                 x0: array_like = 0.0, callback: Callable[[Solution], bool] = None,
                 relative_tolerance: float = 1e-6, maximum_iteration: Real = np.inf, display_progress: bool = True):
        """
        Solves the (non-symmetric) linear system Hx = (L+V)x = b iteratively for x when the real part of H=L+V is
        positive definite, i.e. :code:`np.real(x'*H*x) > 0;` and V  is a contraction, i.e. norm(V) < 1. The latter can be
        ensured for any bound linear system by scaling both sides appropriately.
        Similarly, when the Rayleigh quotient of (L+V) is contained within any half of the complex plane,
        multiplication by :code:`np.exp(-1j*phi)` for some angle phi can ensure that the real part of its Rayleigh quotient is
        positive.

        This method is computationally efficient when:

            - The inverse, (L+I) \ y, can be calculated efficiently, where I is the identity and \ indicating the
                left-multiplication with the inverse.

            - The left-multiplication (V-I)*x can be calculated efficiently.

            - The eigenvalues of V-I are similar.

        Calculations are performed in constant space. During the iteration, space is allocated for two additional
        copies of the working solution.

        Input arguments:

        :param l_plus_identity_inv: The matrix inv(L+I) or a function that calculates the left-multiplication with it in a
            memory efficient way. The function should take one 2D input argument and operate on its columns.

        :param identity_minus_v: The matrix (I-V) or a function that calculates the left-multiplication with it in a
            memory efficient way. The function should take one 2D input argument and operate on its columns.

        :param b: The right-hand side of the linear set of equations.

        :param x0: The starting value of the iteration. Default: 0.

        :param callback: A function, callback(sol: Solution) -> bool, that takes this object as argument and is called
            after each iteration. If it returns True, the iteration is halted. The input arguments maxit and tol are
            ignored when a callback function is provided.

        :param relative_tolerance: The tolerance level. The iteration is terminated when the relative residue has
            decreased below this threshold. This is ignored if the user-defined callback returns a value. Default: 1e-6;

        :param maximum_iteration: The maximum number of iterations before the iteration is terminated. Each iteration
            requires to calls to identity_minus_v and one call to l_plus_identity_inv, as well as 3 additions and the
            calculation of the residual norm. This is ignored if the user-defined callback returns a value.

        :param display_progress: A boolean indicating whether the default callback displays progress every 100
            iterations. This is ignored if the user-defined callback returns a value. Default: True

        """
        x0 = np.array(x0)

         # Standardize inputs
        if not isinstance(l_plus_identity_inv, Callable):
            l_plus_identity_inv = lambda y: l_plus_identity_inv * y
        if not isinstance(identity_minus_v, Callable):
            identity_minus_v = lambda x: identity_minus_v * x

        if np.isclose(np.amin(np.linalg.norm(b, axis=0)), 0.0):
            raise ValueError('The right hand side (b) contains a vector with a norm close to zero.')

        # Store the operators
        self.__l_plus_identity_inv = l_plus_identity_inv
        self.__identity_minus_v = identity_minus_v
        del l_plus_identity_inv, identity_minus_v

        # Make sure that the right-hand side is a column vector
        b = np.asarray(b)
        while b.ndim < 2:
            b = b[..., np.newaxis]
        self.__b = b
        del b
        self.__prec_b_norm = None

        self.__x = None
        self.__dx = None
        self.__residue = np.inf
        self.__iteration = 0

        # If possible, avoid initial multiplications with 0
        if np.all(x0.flatten() == 0):
            # Short-cut the first iteration
            x = self.__identity_minus_v(self.__l_plus_identity_inv(self.b))
            self.__prec_b_norm = np.linalg.norm(x, axis=0)
            self.__residue = self.__prec_b_norm
            self.__iteration += 1
        else:
            x = x0
            del x0

        if callback is None:
            callback = Solution.default_callback

        self.__callback = callback
        self.__max_iteration_default_callback = maximum_iteration
        self.__relative_tolerance_default_callback = relative_tolerance
        self.__display_progress_default_callback = display_progress

        # Initialized first iteration
        self.x = x

    def __to_working_shape(self, vector: array_like) -> np.ndarray:
        if vector is None:
            vector = 0.0
        vector = np.asarray(vector)
        while vector.ndim < 2:
            vector = vector[..., np.newaxis]
        # Broadcast to correct size if needed
        vector = np.array(np.broadcast_to(vector, self.b.shape))
        return vector

    @property
    def x(self) -> np.ndarray:
        """The unknown variable."""
        return self.__x

    @x.setter
    def x(self, new_value: array_like):
        self.__x = self.__to_working_shape(new_value)

        # Take the first step of the first iteration
        self.__dx = self.__identity_minus_v(self.__x)

    @property
    def b(self) -> np.ndarray:
        """The right-hand side."""
        return self.__b

    @b.setter
    def b(self, new_value: array_like):
        self.__prec_b_norm = None
        self.__b = self.__to_working_shape(new_value)

    @property
    def iteration(self) -> int:
        """The iteration number. Starts with 0."""
        return self.__iteration

    @iteration.setter
    def iteration(self, new_value: int):
        self.__iteration = new_value

    @property
    def residue(self) -> np.array:
        """Returns the residue for each parallel problem."""
        return self.__residue

    @property
    def relative_residue(self) -> np.array:
        """Returns the relative residue for each parallel problem."""
        if self.__prec_b_norm is None:
            prec_b = self.__identity_minus_v(self.__l_plus_identity_inv(self.b))
            self.__prec_b_norm = np.linalg.norm(prec_b, axis=0)
        return self.residue / self.__prec_b_norm

    def __iter__(self, callback=None):
        if callback is None:
            callback = self.__callback
        # Initialize iteration
        halt_iteration = False
        while not halt_iteration:  # continue until the callback says otherwise
            # Calculate the correction dx = Gamma(V-1)x + (V-1)x - Gamma b
            self.__dx += self.b
            self.__dx = self.__l_plus_identity_inv(self.__dx)
            self.__dx -= self.__x
            self.__dx = self.__identity_minus_v(self.__dx)
            # Update solution  x = Gamma(V-1)x + Vx - Gamma b
            self.__x += self.__dx  # Do not deallocate dx and temporary variable, we will recycle the allocated memory in the next loop
            self.__iteration += 1

             # Convergence check and feedback
            previous_residue = self.residue
            self.__residue = np.linalg.norm(self.__dx, axis=0)
            self.__dx[:] = self.__identity_minus_v(self.__x)  # Do this right away for the next iteration

             # Call callback function without redundant arguments
            halt_iteration = callback(self)
            if halt_iteration is None:
                halt_iteration = Solution.default_callback(self)
            # Override callback when divergence detected
            if np.any(self.residue > previous_residue):
                log.warning('Divergence detected, either V is not a contraction, the real part of (L+V) is not positive definite, or numerical precission has been reached.')
                halt_iteration = True

            yield self

        return self

    def __call__(self, callback = None) -> Solution:
        """
        Solve iteratively.

        :param callback: (optional) A callback that takes the current solution and decides whether the convergence criterion has been met.
        If the callback returns True, the iteration is stopped.
        :return: A Solution object representing the converged solution.
        """
        for _ in self.__iter__(callback):
            pass  # self is updated in iterator
        return self

    def __str__(self) -> str:
        return f'split_richardson.Solution() at iteration {self.iteration} with residue {self.residue}.'

    @staticmethod
    def default_callback(solution: Solution):
        """The default callback function only takes two arguments"""
        res = solution.relative_residue.flatten()
        if solution.__display_progress_default_callback and solution.iteration % 100 == 0:
            log.info(f'Iteration  {solution.iteration}: error {100*np.mean(res):0.6f}%.')
        return solution.iteration >= solution.__max_iteration_default_callback or np.all(res <= solution.__relative_tolerance_default_callback)


if __name__ == '__main__':
    test_vector_length = 1000
    # Define a test diagonal to convolve with
    ld = (10 / test_vector_length) * np.arange(test_vector_length)[:, np.newaxis] * (1 + 0.5j)
    ld = (ld - ld[ld.size // 2])**2
    ld = ld / ld[0]
    l_plus_identity = lambda x: ft.ifft(ft.fft(x) * (ld + 1))
    l_plus_identity_inv = lambda y: ft.ifft(ft.fft(y) / (ld + 1))
    rng = np.random.RandomState(seed=1)
    vd = (1.0 * rng.rand(test_vector_length, 1) + 0.0) * np.exp(1j * np.pi * (rng.rand(test_vector_length, 1) - 0.5))
    identity_minus_v = lambda x: x - vd * x
    rng = np.random.RandomState(seed=2)
    b = rng.rand(test_vector_length, 2)  # Solve two problems at once

    H = lambda x: l_plus_identity(x) - identity_minus_v(x)
    prec = lambda x: identity_minus_v(l_plus_identity_inv(x))

    import time
    s = Solution(l_plus_identity_inv, identity_minus_v, b)
    start_time = time.perf_counter()
    log.info(s(lambda _: _.iteration >= 1000))
    total_time = time.perf_counter() - start_time
    log.info(f'Converged in {total_time:0.3f} at {s.iteration} ({total_time/s.iteration*1e3:0.3f}ms/iteration) with relative residues of {s.relative_residue}.')

    # x = np.empty((test_vector_length, 1), dtype=np.complex128)
    # H_mat = np.empty((test_vector_length, test_vector_length), dtype=np.complex128)
    # PrecH_mat = np.empty((test_vector_length, test_vector_length), dtype=np.complex128)
    # for _ in range(test_vector_length):
    #     x.ravel()[:] = 0.0
    #     x.ravel()[_] = 1.0
    #     H_mat[:, _] = H(x).ravel()
    #     PrecH_mat[:, _] = prec(H(x)).ravel()
    # w, v = np.linalg.eig(H_mat)
    # print(f'     H real: {np.amin(w.real)} -> {np.amax(w.real)}, imag: {np.amin(w.imag)} -> {np.amax(w.imag)}')
    # w, v = np.linalg.eig(PrecH_mat)
    # print(f'Prec H real: {np.amin(w.real)} -> {np.amax(w.real)}, imag: {np.amin(w.imag)} -> {np.amax(w.imag)}')

