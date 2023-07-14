import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from optics.utils.ft import Grid
from optics.utils.display import complex2rgb
from projects.airy_lightsheet import log


if __name__ == '__main__':
    mask_width = 15e-3 / 2.0
    grid = Grid(extent=[mask_width, mask_width], step=10e-6)
    beam_radius = 5e-3 / 2.0
    alpha = 3.0
    wavelength = 488e-9
    refractive_index = 1.4630
    used_fraction = 5e-3 / mask_width

    waves_to_meter = wavelength / (refractive_index - 1)

    grid_norm = grid / beam_radius

    height_in_waves = alpha * (grid_norm[0] ** 3 + grid_norm[1] ** 3 - (grid_norm[0] + grid_norm[1]) * 3 / 5)
    height_in_m = height_in_waves * waves_to_meter

    height_diff = np.amax(height_in_m) - np.amin(height_in_m)
    log.info(f'Maximum height difference = {height_diff / 1e-6:0.3f}um.')
    inside_beam_square = np.logical_and(np.abs(grid[0]) <= beam_radius, np.abs(grid[1]) <= beam_radius)
    height_diff_in_beam_square = np.amax(height_in_m * inside_beam_square) - np.amin(height_in_m * inside_beam_square)
    log.info(f'Maximum height difference in beam square = {height_diff_in_beam_square / 1e-6:0.3f}um.')

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(complex2rgb(np.exp(2j * np.pi * height_in_waves)))
    axs[1].imshow(height_in_m)

    x_mesh, y_mesh = np.meshgrid(grid[0].ravel(), grid[1].ravel())
    fig_3d, axs_3d = plt.subplots(subplot_kw=dict(projection='3d'))
    axs_3d.plot_surface(x_mesh, y_mesh, height_in_m*inside_beam_square, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    log.info('Done.')
    plt.show()
