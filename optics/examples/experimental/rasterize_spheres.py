#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Code to rasterize spheres from a csv file generated using the Rust package spherical_cow and custom Rust code.
# The resulting permittivity distribution is written as a npz archive.
#

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pathlib

from optics.utils.ft import Grid, MutableGrid
from optics.utils.display import grid2extent

from optics import log

if __name__ == '__main__':
    wavelength = 500e-9
    sphere_diameter = 1e-6  # convert from unit (in the file) to meters
    sampling = wavelength / 4
    side_px = int(sphere_diameter * 120 / sampling)  # Side that will use-up all the generated spheres

    grid = Grid(shape=np.ones(3) * 1200, step=wavelength/4)  # z, y, x
    oversampling_factor = 1  # integer number of subdivisions for anti-aliasing
    background_permittivity = 1.0
    sphere_permittivity = 1.5 ** 2

    sphere_pos_r_filepath = pathlib.Path(__file__).absolute().parent.parent.parent.parent.parent / 'rust/sphere_packing/output/sphere_positions_120.csv'
    output_filepath = pathlib.Path(__file__).absolute().parent / 'output' \
        / f'rasterized_spheres_d{sphere_diameter/1e-6:0.1f}um_n{np.sqrt(sphere_permittivity):0.3f}_step{np.amax(grid.step)/1e-9:0.0f}nm_aa{oversampling_factor}.npz'
    log.info(f'The output file will be {output_filepath}.')
    pathlib.Path.mkdir(output_filepath.parent, parents=True, exist_ok=True)

    log.info(f'Loading sphere positions from {sphere_pos_r_filepath}...')
    positions_radii = np.loadtxt(sphere_pos_r_filepath, delimiter=',', skiprows=1, dtype=np.float32)
    positions_radii = positions_radii[:, [2, 1, 0, 3]]  # reorder as z, y, x, r
    positions_radii = positions_radii[np.argsort(positions_radii[:, 0], axis=0), :]  # sort on z-value
    positions_radii *= sphere_diameter  # convert from unit (in the file) to meters

    # Statistics
    sphere_volume = (4/3) * np.pi * np.sum(positions_radii[:, -1]**3)
    sphere_fraction = sphere_volume / np.prod(
        np.amax(positions_radii[:, :3] + positions_radii[:, -1:], axis=0)
        - np.amin(positions_radii[:, :3] - positions_radii[:, -1:], axis=0)
    )

    log.info(f'Sphere packing fraction = {sphere_fraction*100:0.3f}%.')

    log.info(f'Checking which of the {positions_radii.shape[0]} spheres are within the volume of interest.')

    positions_radii = positions_radii[np.all(
                       np.logical_and(grid.first <= positions_radii[:, :3] - positions_radii[:, -1:],
                                      positions_radii[:, :3] + positions_radii[:, -1:] < grid.first + grid.extent
                                      ), axis=-1
    ), :]

    # # Removing sphere near boundaries in z-direction to allow freespace for the source
    # remove_extent = 15 * 633e-9
    # positions_radii = positions_radii[np.abs(positions_radii[:, 0]) <
    #                                   np.amax(np.abs(positions_radii[:, 0])) - remove_extent]
    # # Removing spheres in the centre
    # cutout_radius = 4e-6
    # distance_from_centre = np.sqrt(np.sum([_**2 for _ in grid]))
    # cutout_bool = distance_from_centre > cutout_radius
    # positions_radii = positions_radii[np.sqrt(np.sum(positions_radii[:, :3] ** 2, axis=1)) > cutout_radius]

    # Save sphere coordinates
    np.savetxt('output/modified_positions_radii.csv', positions_radii, delimiter=',')

    log.info(f'Rasterizing {positions_radii.shape[0]} spheres on {grid.shape}-grid with oversampling factor {oversampling_factor}.')

    grid_hr = Grid(extent=grid.extent, step=grid.step / oversampling_factor)
    sub_grid = Grid(shape=oversampling_factor, step=grid_hr.step)

    permittivity = np.zeros(grid.shape, dtype=np.float16)  # start with susceptibility
    for z_idx, z in enumerate(grid[0].flatten()):
        spheres_in_slice = positions_radii[
                           np.logical_and(z <= positions_radii[:, 0]+positions_radii[:, -1],
                                          positions_radii[:, 0]-positions_radii[:, -1] < z + grid.step[0]), :]
        # Display progress
        if z_idx % max(1, grid.shape[0] // 100) == 0:
            log.info(f'{z_idx/grid.shape[0] * 100:0.1f}%: {z/1e-6:0.1f}um, {spheres_in_slice.shape[0]} spheres in layer.')

        for y_idx, y in enumerate(grid[1].flatten()):
            spheres_in_column = spheres_in_slice[
                                np.logical_and(y <= spheres_in_slice[:, 1]+spheres_in_slice[:, -1],
                                               spheres_in_slice[:, 1]-spheres_in_slice[:, -1] < y + grid.step[1]), :]
            for sphere in spheres_in_column:
                position = sphere[:3]
                radius = sphere[-1]
                r2 = (position[0] - z - sub_grid[0]) ** 2 + (position[1] - y - sub_grid[1]) ** 2 + (position[2] - grid_hr[2]) ** 2
                inside = (r2 < radius ** 2).astype(permittivity.dtype)
                if oversampling_factor > 1:
                    inside = np.sum(inside, axis=(0, 1))
                    inside = np.sum(inside.reshape((-1, oversampling_factor)), axis=-1)
                else:
                    inside = inside[0, 0]
                permittivity[z_idx, y_idx] += inside  # [0, 0]
    # Convert rasterized spheres to permittivity
    permittivity /= oversampling_factor ** 3  # Convert to inside-fraction
    permittivity *= sphere_permittivity - background_permittivity  # Convert inside fraction to susceptibility
    permittivity += background_permittivity  # Convert susceptibility to permittivity

    log.info('Checking for sphere overlap...')
    if np.amax(permittivity) > sphere_permittivity + 1e-3:
        log.warning(f'Overlap in {np.sum(permittivity > sphere_permittivity)}/{permittivity.size} points!')
    else:
        log.info('No sphere overlap detected.')
    log.info(f'Permittivity between {np.amin(permittivity):0.3f} and {np.amax(permittivity):0.3f} (n={np.sqrt(np.amax(permittivity)):0.3f}).')

    log.info(f'Saving to {output_filepath}...')
    np.savez(output_filepath, permittivity=permittivity, shape=grid.shape, step=grid.step, oversampling_factor=oversampling_factor,
             background_permittivity=background_permittivity, sphere_permittivity=sphere_permittivity)
    log.info(f'Saved results to {output_filepath}...')

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(permittivity[permittivity.shape[0]//2, :, :], extent=grid2extent(grid.project(axes_to_remove=0))/1e-6)
    axs[0].set(xlabel=r'y [$\mu$m]', ylabel=r'x [$\mu$m]')
    axs[1].imshow(permittivity[:, permittivity.shape[1]//2, :], extent=grid2extent(grid.project(axes_to_remove=1))/1e-6)
    axs[1].set(xlabel=r'z [$\mu$m]', ylabel=r'x [$\mu$m]')
    axs[2].imshow(permittivity[:, :, permittivity.shape[2]//2], extent=grid2extent(grid.project(axes_to_remove=2))/1e-6)
    axs[2].set(xlabel=r'z [$\mu$m]', ylabel=r'y [$\mu$m]')

    log.info('Done.')
    plt.show()
