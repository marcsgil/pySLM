"""
Code to check how finite difference derivatives compare to Fourier transform based ones when sampling close to Nyquist.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from scipy.interpolate import interp1d

from optics.utils import ft
from optics.utils.display import complex2rgb

from optics.experimental import log


k0 = 2 * np.pi / 1
dtype = np.complex128


def create_matrix(grid: ft.Grid, f: Callable):
    result = np.zeros((grid.shape[0], grid.shape[0]), dtype=dtype)
    for col, x in zip(result, np.eye(grid.shape[0])):
        col[:] = f(x) * grid.step[0]  # scale by the area represented by a single element
    return result


def get_bound(grid, boundary_thickness, max_attenuation: float = 0.25):
    distance_in_bound = np.maximum(0, (grid[0][0] + boundary_thickness) - grid[0]) + \
                        np.maximum(0, grid[0] - (grid[0][0] + grid.extent[0] - boundary_thickness))
    return max_attenuation * 1j * distance_in_bound / boundary_thickness


def gauss(g, mean: float = 0.0, stddev: float = 1.0):
    return np.exp(-0.5 * (((g[0] - mean) / stddev) ** 2))


def get_epsilon(grid, boundary_thickness):
    thickness = grid.extent[0] / 8
    position = grid[0][0] + grid.extent[0] - boundary_thickness - thickness
    return 1 + get_bound(grid, boundary_thickness) + (np.abs(grid[0] - position) < thickness / 2) * 0  # refractive plate


def get_source(grid, position):
    return gauss(grid, position, 0.000001) * np.exp(1j * k0) * 1j * k0 * 2  # to get a unit amplitude solution in free space


def ft_helmholtz(grid, epsilon):
    laplacian_ft = -grid.k[0]**2
    return lambda x: ft.ifftn(laplacian_ft * ft.fftn(x)) + k0**2 * epsilon * x


def fd_helmholtz(grid, epsilon):
    def diff2(x):
        return np.diff(x, n=2, append=0, prepend=0) / grid.step[0]**2
    return lambda x: diff2(x) + k0**2 * epsilon * x


def calc_error(step, display=True):
    grid_extent = 32
    reference_subsampling = 1024
    boundary_thickness = grid_extent / 4
    grid = ft.Grid(first=-boundary_thickness, extent=[grid_extent], step=step)
    grid_hr = ft.Grid(first=-boundary_thickness, extent=grid.extent, step=1/reference_subsampling)

    source_position = grid[0][0] + boundary_thickness

    epsilon = get_epsilon(grid, boundary_thickness)
    source = get_source(grid, source_position)
    epsilon_hr = get_epsilon(grid_hr, boundary_thickness)
    source_hr = get_source(grid_hr, source_position)

    max_matrix_size = np.maximum(grid.shape[0], grid.shape[0])  # Add _hr if not using analytical
    log.info(f'Creating matrices up to size {max_matrix_size}x{max_matrix_size}...')
    ft_matrix = create_matrix(grid, ft_helmholtz(grid, epsilon))
    fd_matrix = create_matrix(grid, fd_helmholtz(grid, epsilon))
    # matrix_hr = create_matrix(grid_hr, fd_helmholtz(grid_hr, epsilon_hr))  # analytical solution for free space

    log.info(f'Inverting matrices up to size {max_matrix_size}x{max_matrix_size}...')
    ft_solution = np.linalg.inv(ft_matrix) @ source
    fd_solution = np.linalg.inv(fd_matrix) @ source
    # solution_hr = np.linalg.inv(matrix_hr) @ source_hr  # use this when not in free space
    solution_hr = np.exp(1j * k0 * np.abs(grid_hr[0]))  # analytical solution for free space

    log.info('Calculating error...')
    inside_domain = np.logical_and(
        grid_hr[0][0] + boundary_thickness < grid_hr[0],
        grid_hr[0] < grid_hr[0][0] + grid_hr.extent[0] - boundary_thickness,
    )
    int_args = dict(kind='cubic', fill_value=0, bounds_error=0)
    ft_solution_int = interp1d(grid[0], ft_solution, **int_args)
    fd_solution_int = interp1d(grid[0], fd_solution, **int_args)
    ft_solution_err_hr = (ft_solution_int(grid_hr[0]) - solution_hr) * inside_domain
    fd_solution_err_hr = (fd_solution_int(grid_hr[0]) - solution_hr) * inside_domain

    if display:
        log.info('Displaying...')
        fig, axs = plt.subplots(4, 1, sharex='all')
        solution_style = dict(color='#000000ff', linewidth=1)
        fd_style = dict(marker='o', markersize=3, color='#c0000080', linewidth=2)
        ft_style = dict(marker='o', markersize=3, color='#00800080', linewidth=2)
        axs[0].plot(grid_hr[0], epsilon_hr.real, **solution_style, label='hr real')
        axs[0].plot(grid_hr[0], epsilon_hr.imag, linestyle=':', **solution_style, label='hr imag')
        axs[0].plot(grid[0], epsilon.real, **fd_style, label='lr real')
        axs[0].plot(grid[0], epsilon.imag, linestyle=':', **fd_style, label='lr imag')
        axs[0].set_title('epsilon')
        axs[0].legend()

        axs[1].plot(grid_hr[0], source_hr.real, **solution_style, label='hr real')
        axs[1].plot(grid_hr[0], source_hr.imag, ':', **solution_style, label='hr imag')
        axs[1].plot(grid[0], source.real, **fd_style, label='lr real')
        axs[1].plot(grid[0], source.imag, ':', **fd_style, label='lr imag')
        axs[1].set_title('source')
        axs[1].legend()

        axs[2].plot(grid_hr[0], solution_hr.real, **solution_style, label='hr real')
        axs[2].plot(grid_hr[0], solution_hr.imag, ':', **solution_style, label='hr imag')
        axs[2].plot(grid[0], fd_solution.real, **fd_style, label='fd real')
        axs[2].plot(grid[0], fd_solution.imag, ':', **fd_style, label='fd imag')
        axs[2].plot(grid[0], ft_solution.real, **ft_style, label='ft real')
        axs[2].plot(grid[0], ft_solution.imag, ':', **ft_style, label='ft imag')
        axs[2].set_title('solution')
        axs[2].legend()

        axs[3].plot(grid_hr[0], fd_solution_err_hr.real, **fd_style, label='fd real')
        axs[3].plot(grid_hr[0], fd_solution_err_hr.imag, ':', **fd_style, label='fd imag')
        axs[3].plot(grid_hr[0], ft_solution_err_hr.real, **ft_style, label='ft real')
        axs[3].plot(grid_hr[0], ft_solution_err_hr.imag, ':', **ft_style, label='ft imag')
        axs[3].set_title('error')
        axs[3].legend()

        plt.show(block=False)
        plt.pause(0.001)

    # crop domain for further analysis
    ft_solution_err_hr = ft_solution_err_hr[inside_domain]
    fd_solution_err_hr = fd_solution_err_hr[inside_domain]

    return ft_solution_err_hr, fd_solution_err_hr


if __name__ == '__main__':
    # steps = np.array([1/16, 1/32])
    # steps = 1 / 2**np.arange(1, 6+1)
    steps = 1 / np.arange(2, 8.25, 0.25)
    mean_error_list = []
    median_error_list = []
    rms_error_list = []
    max_error_list = []
    log.info(f'Analyzing subdivisions by {1 / steps}...')
    for step in steps:
        log.info(f'Analyzing {1/step:0.3f} subdivisions...')
        error_list = calc_error(step, display=steps.size <= 3)
        mean_errors = [np.mean(np.abs(_)) for _ in error_list]
        median_errors = [np.median(np.abs(_)) for _ in error_list]
        rms_errors = [np.sqrt(np.mean(np.abs(_)**2)) for _ in error_list]
        max_errors = [np.amax(np.abs(_)) for _ in error_list]
        for label, mean_error, median_error, rms_error, max_error in zip(['Fourier space', 'Finite Diff. '], mean_errors, median_errors, rms_errors, max_errors):
            log.info(f'{label}: mean={mean_error:0.6f}, median={median_error:0.6f}, rms=mean={rms_error:0.6f}, max={max_error:0.6f}')
        mean_error_list.append(mean_errors)
        median_error_list.append(median_errors)
        rms_error_list.append(rms_errors)
        max_error_list.append(max_errors)
    mean_error_list = np.array(mean_error_list).transpose()
    median_error_list = np.array(median_error_list).transpose()
    rms_error_list = np.array(rms_error_list).transpose()
    max_error_list = np.array(max_error_list).transpose()

    fig, ax = plt.subplots(1, 1)
    ax.plot(1 / steps, mean_error_list[0], ':', color='#008000ff', label='FT mean')
    ax.plot(1 / steps, mean_error_list[1], ':', color='#c00000ff', label='FD mean')
    ax.plot(1 / steps, median_error_list[0], '--', color='#008000ff', label='FT median')
    ax.plot(1 / steps, median_error_list[1], '--', color='#c00000ff', label='FD median')
    ax.plot(1 / steps, rms_error_list[0], '-', color='#008000ff', linewidth=3, label='FT rms')
    ax.plot(1 / steps, rms_error_list[1], '-', color='#c00000ff', linewidth=3, label='FD rms')
    ax.plot(1 / steps, max_error_list[0], '-o', color='#008000ff', label='FT max')
    ax.plot(1 / steps, max_error_list[1], '-o', color='#c00000ff', label='FD max')
    ax.plot(1 / steps, max_error_list[1] * 0, '-', color='#000000ff', linewidth=0.5)  # a visiual guide
    ax.set(xlabel='fraction of wavelength', ylabel='error', xlim=1/steps[[0, -1]])
    ax.legend()

    plt.show()
