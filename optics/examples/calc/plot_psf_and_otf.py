#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from examples.calc import log
from optics.utils import ft
from optics.calc import zernike
from optics.utils.ft import Grid
from optics.utils.display import complex2rgb, grid2extent
from optics.instruments import objective
from optics.calc.psf import PSF, RichardsWolfPSF

if __name__ == '__main__':
    show_intensity_analysis = False

    log.info('Defining the optical system...')
    obj = objective.InfinityCorrected(40, 0.80, refractive_index=1.33)
    wavelength = 500e-9
    polarized = True
    polarization = np.array([1.0, 1.0j]) / np.sqrt(2)
    polarization = np.array([0.0, 1.0])
    # polarization = np.array([1.0, 0.0])

    def tophat(nu_y, nu_x):
        return 1.0


    def cubic(nu_y, nu_x):
        return np.exp(2j * np.pi * 5 * (nu_y**3 + nu_x**3))


    def aberrated_pupil_function(nu_y, nu_x):
        aberration = zernike.Polynomial([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])  # primary spherical
        return np.exp(2j * np.pi * aberration.cartesian(nu_y, nu_x))


    # The function that is actually displayed:
    scalar_pupil_function = tophat
    if polarized:
        def pupil_function(nu_y, nu_x):
            return scalar_pupil_function(nu_y, nu_x) * polarization[:, np.newaxis, np.newaxis]
    else:
        pupil_function = scalar_pupil_function

    # Define the point-spread function sampling
    grid = Grid(3, 1.0e-6, first=0) @ Grid(np.full(2, 64), 0.1e-6)

    # Calculate the PSF
    log.info(f'Calculating PSF between {np.round(grid.first / 1e-9)} nm and '
             + f'{np.round((grid.first+grid.extent-grid.step) / 1e-9)} nm in steps of {np.round(grid.step / 1e-9)} nm...')
    psf = PSF(objective=obj, vacuum_wavelength=wavelength, pupil_function=pupil_function)
    # psf = RichardsWolfPSF(objective=obj, vacuum_wavelength=wavelength, pupil_function=1.0)

    log.info('Calculating PSF...')
    field_array = psf(*grid)  # the actual calculations happens here
    intensity_array = np.abs(field_array) ** 2
    intensity_array = np.sum(intensity_array, axis=-4)

    # Calculate the OTF
    log.info('Calculating OTF...')
    otf_array = ft.fftshift(ft.fft2(ft.ifftshift(intensity_array)))
    k_grid = grid.k.as_origin_at_center.project([-2, -1])

    #
    # Display
    #
    log.info('Displaying...')
    fig_E, axs_E = plt.subplots(1 + 2*polarized, field_array.shape[-3], sharex='all', sharey='all', figsize=(10, 8))
    plt.get_current_fig_manager().set_window_title('Field Point Spread Function')
    axs_E = np.atleast_1d(axs_E)
    while axs_E.ndim < 2:
        axs_E = axs_E[np.newaxis, ...]
    normalization = np.amax(np.abs(field_array)) / 2
    for z_idx in range(axs_E.shape[1]):
        for p_idx in range(1 + 2 * polarized):
            axs_E[p_idx, z_idx].imshow(complex2rgb(field_array[p_idx, z_idx, :, :] / normalization),
                                       extent=grid2extent(grid.project([-2, -1]) * 1e6))
            axs_E[p_idx, z_idx].set(xlabel='$x [\mu m]$', ylabel='$y [\mu m]$',
                                    title=f"$E_{'zyx'[p_idx]}$ at {grid[0].ravel()[z_idx]*1e6:0.1f} $\mu m$")
            # axs_E[p_idx, z_idx].set(xlim=[-25, 25], ylim=[-25, 25])

    if show_intensity_analysis:
        fig_I, axs_I = plt.subplots(2, field_array.shape[-3], sharex='row', sharey='row', figsize=(10, 8))
        try:
            fig_I.canvas.set_window_title('Intensity Point Spread Function and Optical Transfer Function')
        except AttributeError:
            fig_I.canvas.setWindowTitle('Intensity Point Spread Function and Optical Transfer Function')
        for z_idx in range(axs_I.shape[1]):
            axs_I[0, z_idx].imshow(intensity_array[z_idx, :, :],
                                   extent=grid2extent(grid.project([-2, -1]) * 1e6))
            axs_I[0, z_idx].set(xlabel='$x [\mu m]$', ylabel='$y [\mu m]$',
                                title=f'I at {grid[0].ravel()[z_idx]*1e6:0.1f} $\mu m$')
            # axs_I[0, z_idx].set(xlim=[-25, 25], ylim=[-25, 25])
            axs_I[1, z_idx].imshow(complex2rgb(otf_array[z_idx, :, :], normalization=1), extent=grid2extent(k_grid * 1e-6))
            axs_I[1, z_idx].set(xlabel='$k_x [rad/\mu m]$', ylabel='$k_y [rad/\mu m]$',
                                title=f'OTF at {grid[0].ravel()[z_idx]*1e6:0.0f} $\mu m$')

        fig_OTF = plt.figure(figsize=(10, 4))
        try:
            fig_OTF.canvas.set_window_title('Modulation Transfer Function')
        except AttributeError:
            fig_I.canvas.setWindowTitle('Modulation Transfer Function')
        for z_idx in range(field_array.shape[0]):
            ax = fig_OTF.add_subplot(1, field_array.shape[0], 1 + z_idx, projection="3d")
            ax.plot_surface(k_grid[-1] * 1e-6, k_grid[-2] * 1e-6, np.abs(otf_array[z_idx, :, :]),
                            cmap=cm.viridis, linewidth=1, antialiased=True)
            ax.set(xlabel='$k_x [rad/\mu m]$', ylabel='$k_y [rad/\mu m]$',
                   title=f"MTF at {grid[0].ravel()[z_idx]*1e6:0.1f} $\mu m$")

    plt.show()
