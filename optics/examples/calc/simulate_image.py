#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
import matplotlib.pyplot as plt

from examples.calc import log
from optics.utils import ft, reference_object
from optics.calc import zernike
from optics.utils.display import grid2extent
from optics.instruments import objective
from optics.calc.psf import PSF, RichardsWolfPSF


if __name__ == '__main__':
    log.info('Defining the optical system...')
    obj = objective.InfinityCorrected(40, 0.80, refractive_index=1.33)
    wavelength = 500e-9

    max_number_of_photons = 100  # dynamic range of camera that is used

    rng = np.random.Generator(np.random.PCG64(seed=1))


    def tophat(nu_y, nu_x):
        return 1.0


    def cubic(nu_y, nu_x):
        return np.exp(2j * np.pi * 5 * (nu_y**3 + nu_x**3))


    def aberrated_pupil_function(nu_y, nu_x):
        aberration = zernike.Polynomial([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])  # primary spherical
        return np.exp(2j * np.pi * aberration.cartesian(nu_y, nu_x))


    # The function that is actually displayed:
    pupil_function = cubic

    # Define the point-spread function sampling
    grid = ft.Grid(3, 1.0e-6, first=0) @ ft.Grid(np.full(2, 512), 0.1e-6)

    # Calculate the PSF
    log.info(f'Calculating PSF between {np.round(grid.first / 1e-9)} nm and '
             + f'{np.round((grid.first+grid.extent-grid.step) / 1e-9)} nm in steps of {np.round(grid.step / 1e-9)} nm...')
    psf = PSF(objective=obj, vacuum_wavelength=wavelength, pupil_function=pupil_function)
    # psf = RichardsWolfPSF(objective=obj, wavelength=wavelength, pupil_function=1.0)

    log.info('Calculating PSF...')
    psf_field_array = psf(*grid)  # the actual calculations happen here
    psf_intensity_array = np.abs(psf_field_array[0]) ** 2
    psf_intensity_array /= np.amax(np.sum(psf_intensity_array, axis=(-2, -1)))  # TODO: Can this be done right from the start?

    log.info('Calculating OTF...')
    # Calculate the OTF
    otf_array = ft.fft2(ft.ifftshift(psf_intensity_array, axes=(-2, -1)))  # Maximum value = 1
    k_grid = grid.k.as_origin_at_center.project([-2, -1])

    log.info('Loading reference...')
    original_object = np.asarray(reference_object.usaf1951(grid.shape[-2:], scale=1.0)) / 255.0  # Maximum value = 1

    log.info('Simulating light propagation...')
    blurred_images = np.maximum(0.0, ft.ifft2(ft.fft2(original_object) * otf_array).real)  # Non-negative, maximum value ~1

    log.info('Simulating Poisson photon noise...')
    detected_images = rng.poisson(blurred_images * max_number_of_photons).astype(np.float32) / max_number_of_photons

    #
    # Display
    #
    log.info('Displaying...')
    fig_E, axs = plt.subplots(1, detected_images.shape[-3], sharex='all', sharey='all', figsize=(10, 8))
    try:
        fig_E.canvas.set_window_title('Blur as a function of distance.')
    except AttributeError:
        fig_E.canvas.setWindowTitle('Blur as a function of distance.')
    for z_idx, ax in enumerate(axs):
        ax.imshow(blurred_images[z_idx],
                  extent=grid2extent(grid.project([-2, -1]) * 1e6))
        ax.set(xlabel=r'$x [\mu m]$', ylabel=r'$y [\mu m]$',
               title=f"$E_{'xyz'[0]}$ at {grid[0].ravel()[z_idx]*1e6:0.1f} $\mu m$")

    plt.show()
