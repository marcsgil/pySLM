#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pathlib
from PIL import Image

from examples.calc import log
from optics.utils import ft
from optics.utils.display import complex2rgb, grid2extent, colormap
from optics.calc.beam import Beam
from optics.utils import reference_object
from optics.experimental import sphere_packing


if __name__ == '__main__':
    use_spheres = False

    k_0 = 1.0
    numerical_aperture = 0.75
    k_max = k_0 * numerical_aperture

    output_path = pathlib.Path('output').absolute()
    output_filepath = pathlib.PurePath(output_path, pathlib.PurePath(__file__).name[:-3])

    cmap = colormap.InterpolatedColorMap('rainbow', [(0, 0, 0), (1, 0.25, 0.25), (1, 1, 0.25), (0.25, 1, 0.25),
                                                     (0.25, 1, 1), (0.25, 0.25, 1), (1, 0.25, 1), (1, 1, 1)],
                                         points=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1])

    cubic1d = lambda nu_x: np.exp(2j * np.pi * 10 * nu_x ** 3) * (nu_x ** 2 < 1.0)

    # Define the PSF sampling
    image_width = 1024
    grid = ft.Grid([image_width, 1, image_width], first=[-image_width // 3, 0, -image_width * 9 // 16])

    log.info('Defining the scattering object.')
    cloud = np.array(reference_object.cloud(grid.shape[[0, 2]], scale=0.5))[:, np.newaxis, :, 2] > 0
    cloud = np.roll(cloud, -image_width // 20, axis=-3)
    n = 1.0 + 0.33 * cloud
    if use_spheres:
        log.info('Packing spheres...')
        spheres = sphere_packing.pack_and_rasterize(grid.project(axes_to_keep=[-3, -1]), radius_mean=10.0)[:, np.newaxis, :]
        n = 1.0 + 0.33 * cloud * spheres

    log.info('Calculating the transmitted field...')
    beam = Beam(grid=grid, field_ft=lambda k_z, k_y, k_x: cubic1d(k_x / k_max - 0.25))  # Add a bit of tilt
    forward_field_array = beam.field(refractive_index=n[::-1])[:, ::-1]
    forward_intensity_array = np.sum(np.abs(forward_field_array) ** 2, axis=-4)

    log.info('Calculating phase conjugation...')
    phase_conjugate_grid = ft.Grid(grid.shape, first=[0, 0, grid.first[-1]])
    phase_conjugated_beam = Beam(grid=phase_conjugate_grid, field=np.conj(forward_field_array[:, 0]))
    phase_conjugated_field_array = phase_conjugated_beam.field(refractive_index=n)
    phase_conjugated_intensity_array = np.sum(np.abs(phase_conjugated_field_array)**2, axis=-4)

    #
    # Display
    #
    log.info('Done! Displaying...')
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(16, 10))
    axs[0, 0].imshow(complex2rgb(forward_field_array[0, :, 0, :], normalization=2),
                     extent=grid2extent(grid.project([-3, -1])))
    axs[0, 1].imshow(forward_intensity_array[:, 0, :], cmap=cmap,
                     extent=grid2extent(grid.project([-3, -1])))
    axs[1, 0].imshow(complex2rgb(phase_conjugated_field_array[0, :, 0, :], normalization=2),
                     extent=grid2extent(grid.project([-3, -1])))
    axs[1, 1].imshow(phase_conjugated_intensity_array[:, 0, :], cmap=cmap,
                     extent=grid2extent(grid.project([-3, -1])))
    for _ in range(2):
        axs[_, 0].set(xticks=[], yticks=[], title=f"E")
        mask = n[:, 0, :] > 1.0
        mask = np.stack((0.9 * mask, 0.9 * mask, mask, 0.25 * mask), axis=-1).astype(np.float32)
        axs[_, 0].imshow(mask, extent=grid2extent(grid.project([-3, -1])))
        axs[_, 1].set(xticks=[], yticks=[], title=f"I")
        axs[_, 1].imshow(mask, extent=grid2extent(grid.project([-3, -1])))
        fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), ax=axs[_, 1])

    plt.show(block=False)

    log.info('Saving results to %s...' % output_filepath.as_posix())
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.ioff()
    fig.savefig(output_filepath.as_posix() + '.pdf', bbox_inches='tight', format='pdf')
    fig.savefig(output_filepath.as_posix() + '.svgz', bbox_inches='tight', format='svgz')
    plt.ion()

    log.info('Saving the calculated data...')
    np.savez(output_filepath.as_posix() + '_data', k_0=k_0, refractive_index=n,
             forward_field_array=forward_field_array,
             phase_conjugated_field_array=phase_conjugated_field_array)
    field_rgb = complex2rgb(phase_conjugated_field_array[0, :, 0, :], normalization=2)
    intensity_rgb = cmap(phase_conjugated_intensity_array[:, 0, :] / np.amax(phase_conjugated_intensity_array[:, 0, :]),
                         bytes=True)
    Image.fromarray((mask * 255.5).astype(np.uint8)).save(output_filepath.as_posix() + '_mask.png')
    Image.fromarray((field_rgb * 255.5).astype(np.uint8)).save(output_filepath.as_posix() + '_field.png')
    Image.fromarray(intensity_rgb).save(output_filepath.as_posix() + '_intensity.png')
    Image.fromarray(cmap(np.tile(1 - np.arange(1024)[:, np.newaxis] / (1024 - 1), 32), bytes=True)).save(output_filepath.as_posix() + '_colormap.png')

    log.info('Done saving!')
    plt.show(block=True)
