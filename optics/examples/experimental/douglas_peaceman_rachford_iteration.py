import time

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

from examples.experimental import log
from optics.utils import ft
from optics.utils.display import complex2rgb, grid2extent
from optics.calc.polynomial import Polynomial


class DouglasPeacemanRachford:
    def __init__(self, resolvent_l, resolvent_r, resolvent_r_inv, y=None, x0=0.0, shape=None, dtype=np.complex128):
        self.__resolvent_l = resolvent_l
        self.__resolvent_r = resolvent_r
        self.__resolvent_r_inv = resolvent_r_inv
        self.__shape = np.asarray(shape if shape is not None else np.asarray(x0).shape).copy()
        self.__shape.flags['WRITEABLE'] = False
        self.__iteration: int = 0
        self.__y = y
        self.__dtype = dtype  # before setting x
        self.__resolvent_r_x = None
        self.x = x0

        self.__residue = None
        self.__previous_updated_norm = np.infty

    @property
    def shape(self) -> np.ndarray:
        """The shape of the solution."""
        return self.__shape

    @property
    def iteration(self) -> int:
        """
        The current iteration number.

        :return: An integer indicating how many iterations have been done.
        """
        return self.__iteration

    @iteration.setter
    def iteration(self, new_iteration: int = 0):
        """
        The current iteration number.
        Resets the iteration count or sets it to a specified integer.
        This does not affect the calculation, only (potentially) the stop criterion.

        :param new_iteration: (optional) the new iteration number
        """
        self.__iteration = new_iteration

    @property
    def x(self) -> np.ndarray:
        """Get the current solution."""
        return self.__resolvent_r_inv(self.__resolvent_r_x)

    @x.setter
    def x(self, new_x):
        """Set the current solution."""
        new_x = np.broadcast_to(new_x, self.shape).astype(dtype=self.__dtype).copy()
        self.__resolvent_r_x = self.__resolvent_r(new_x)

    def __array__(self) -> np.ndarray:
        """The current solution."""
        return self.x

    @property
    def y(self) -> np.ndarray:
        """Get the right-hand side of the problem."""
        return self.__y

    @y.setter
    def y(self, new_y):
        """Set the right-hand side of the problem."""
        self.__y = new_y

    @property
    def residue(self) -> float:
        """
        Returns the current relative residue of the inverse problem.
        """
        if self.__residue is None:
            self.__residue = float(self.__previous_updated_norm / np.linalg.norm(self.x))

        return self.__residue

    def __iter__(self):
        """
        Returns an iterator that on __next__() yields this Solution after updating it with one cycle of the algorithm.
        Obtaining this iterator resets the iteration counter.

        Usage:

        .. code:: python

            for _ in DouglasPeacemanRachford(...):
                if _.iteration > 100:
                    break
            print(solution.residue)
        """
        update_weight = 3 / 4

        self.iteration = 0  # reset iteration counter
        while True:
            self.iteration += 1
            self.__residue = None  # Invalidate residue

            # Calculate update to the field (self.__field_array, d_field, and self.__source are scaled by k0^2)
            z = 2 * self.__resolvent_r(self.__resolvent_r_x) - self.__resolvent_r_x
            dx = - self.__resolvent_r_x - z + self.__resolvent_l(self.__y + 2 * z)
            self.__resolvent_r_x += update_weight * dx
            self.__previous_updated_norm = np.linalg.norm(dx)

            yield self

    def solve(self, y=None, callback: Callable = lambda _: _.iteration < 1e4 and _.residue > 1e-4):
        """
        Runs the algorithm until the convergence criterion is met or until the maximum number of iterations is reached.

        :param y: The right-hand side.
        :param callback: optional callback function that overrides the one set for the solver.
            E.g. callback=lambda s: s.iteration < 100

        :return: This Solution object, which can be used to query e.g. the final field E using Solution.E.
        """
        if y is not None:
            self.y = y
        log.info(f'{self.iteration}: {self.residue:0.6f}')
        for sol in self:
            # sol is the current iteration result
            # now execute the user-specified callback function
            if not callback(sol):  # function may have side effects
                log.debug('Convergence target met, stopping iteration.')
                break  # stop the iteration if it returned False

        return self


def douglas_peaceman_rachford_iteration(resolvent_l, resolvent_r, resolvent_r_inv, y=None, x0=0.0, shape=None,
                                        callback: Callable = lambda _: _.iteration < 1e4 and _.residue > 1e-4):
    solution = DouglasPeacemanRachford(resolvent_l, resolvent_r, resolvent_r_inv, y=y, x0=x0, shape=shape)
    return solution.solve(callback=callback)


def calc_distance_in_bound(grid: ft.Grid, boundary_thicknesses) -> np.ndarray:
    distance_in_boundary = 0.0
    for rng, t in zip(grid, boundary_thicknesses):
        distance_in_boundary = np.maximum(distance_in_boundary,
                                          np.maximum((rng.ravel()[0] + t) - rng, rng - (rng.ravel()[-1] - t)) / (t + (t==0)))
    return distance_in_boundary


class Chi3NonLinear:
    """
    Solves y = b x + d |x|^2 x
    """
    def __init__(self, b, d):
        self.b, self.d = np.broadcast_arrays(b, d)

    def __call__(self, x):
        return (self.b + self.d * np.abs(x)**2) * x

    def inv(self, y):
        """
        Solves b x + d |x|^2 x = y, by first solving
        |b x + d |x|^2 x|^2 = |y|^2, and then matching the complex argument of b x + d |x|^2 x and y.

        :param y: The right hand side of the equation.
        :return: The unknown x
        """
        poly = Polynomial([-np.abs(y)**2, np.abs(self.b)**2, 2 * (np.conj(self.b) * self.d).real, np.abs(self.d)**2])
        abs_x = np.sqrt(np.amax(poly.roots.real, axis=0))
        phi_x = np.angle(y) - np.angle(self(abs_x))
        return abs_x * np.exp(1j * phi_x)


if __name__ == '__main__':
    test_inputs = False

    grid = ft.Grid(np.array([256, 256]) * 2, 0.1 / 4)

    refractive_index = 1.5 + 0.0j
    chi3 = 0.1 + 0.012j

    k0 = 2 * np.pi
    k2 = sum((_/k0)**2 for _ in grid.k)

    structure = (np.abs(grid[0]) < grid.extent[0] / 8)

    boundary_thicknesses = [5, 0]
    permittivity = np.ones(grid.shape, dtype=np.complex128)
    permittivity += 0.5j * calc_distance_in_bound(grid, boundary_thicknesses=boundary_thicknesses)

    source = np.exp(1j * k0 * grid[0]) * np.exp(-0.5 * ((grid[0] - (grid[0].ravel()[0] + boundary_thicknesses[0])) / 0.5) ** 2) \
        * np.exp(-0.5 * (grid[1] / (grid.extent[1] / 16)) ** 2)
    source *= 1e-6

    epsilon0 = 1.0  # Must be one to allow the calculation of resolvent_r
    scaling_factor = 1.1j * np.amax(np.abs(permittivity - epsilon0))
    log.info(f'Scaling factor = {scaling_factor}.')
    l_values = (epsilon0 - k2) / scaling_factor
    r_values = (permittivity - epsilon0) / scaling_factor
    if np.any(l_values.real < 0):
        log.error(f'Left-hand has negative real range [{np.amin(l_values.real)}, {np.amax(l_values.real)}]!')
    if np.any(r_values.real < 0):
        log.error(f'Right-hand has negative real range [{np.amin(r_values.real)}, {np.amax(r_values.real)}]!')
    log.info(f'|L|={np.amax(np.abs(l_values))}, |R|={np.amax(np.abs(r_values))}')
    resolvent_l_inv = lambda _: ft.ifftn(ft.fftn(_) * (1 + l_values))
    resolvent_l = lambda _: ft.ifftn(ft.fftn(_) / (1 + l_values))
    resolvent_r_inv = lambda _: _ * (1 + r_values)
    resolvent_r = lambda _: _ / (1 + r_values)
    cnl = Chi3NonLinear(b=1 + r_values, d=chi3 / scaling_factor)
    def resolvent_r_inv(_):  # 1 + R
        # 1 + R + chi3^2 / scaling_factor
        linear_response = _ * (1 + r_values)
        non_linear_response = cnl(_)
        return linear_response * (1 - structure) + non_linear_response * structure  # combine materials
    def resolvent_r(_):
        linear_result = _ / (1 + r_values)
        non_linear_result = cnl.inv(_)
        return linear_result * (1 - structure) + non_linear_result * structure

    y = 0.8 * source / scaling_factor

    def callback(_):
        if _.iteration % 100 == 0:
            log.info(f'{_.iteration}: {_.residue:0.6f}')
        return _.iteration < 5000 and _.residue > 1e-5

    # For checking
    A = lambda x: resolvent_l_inv(x) + resolvent_r_inv(x) - 2 * x

    if test_inputs:
        log.info('Testing inputs...')
        rng = np.random.RandomState(seed=1)
        for _ in range(100):
            if _ % 10 == 0:
                log.debug(f'Accretivity test {_}...')
            x0 = rng.randn(*grid.shape) + 1j * rng.randn(*grid.shape)
            x1 = rng.randn(*grid.shape) + 1j * rng.randn(*grid.shape)
            accretive_part = np.vdot(A(x1) - A(x0), x1 - x0).real
            if accretive_part < 0.01:
                log.error(f'Not accretive! Accretive part = {accretive_part}.')
        for _ in range(10):
            if _ % 1 == 0:
                log.debug(f'Resolvent test {_}...')
            test = rng.randn(*grid.shape) + 1j * rng.randn(*grid.shape)
            result = resolvent_r(resolvent_r_inv(test))
            rel_error = np.linalg.norm(result - test) / np.linalg.norm(test)
            if rel_error > 1e-8:
                log.error(f'resolvent_r_inv is not the inverse of resolvent_r. Relative error: {rel_error}')

    log.info('Solving...')
    start_time = time.perf_counter()
    solution = douglas_peaceman_rachford_iteration(resolvent_l, resolvent_r, resolvent_r_inv,
                                                   y, shape=y.shape, callback=callback)
    log.info(f'Solved with {solution.iteration} iterations in {time.perf_counter() - start_time:0.3f}s.')
    log.info(f'Residue = {solution.residue} or {np.linalg.norm(A(solution.x) - y) / np.linalg.norm(y)}')

    log.info('Displaying...')
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(complex2rgb(source, normalization=1.0), extent=grid2extent(grid))
    axs[1].imshow(complex2rgb(solution.x, normalization=4.0), extent=grid2extent(grid))

    log.info('Done!')
    plt.show()
