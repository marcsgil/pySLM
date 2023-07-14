"""
Code to demonstrate the complex, fraction, zooming feature in :mod:``optics.ft``.
This essentially uses a Chirp-z interpolation to sinc-interpolate complex values. More details in :mod:``optics.ft.czt``.
"""
import numpy as np
import matplotlib.pyplot as plt

from optics.utils.ft.grid import Grid
from optics.utils.ft import interp, SincInterpolator
from optics.utils.display import complex2rgb, grid2extent

from examples.calc import log


if __name__ == '__main__':
    grid = Grid(shape=[200, 300], first=0)
    zoom_grid = Grid(center=[100, 150], extent=[20, 30], step=0.1)

    log.info(f'Preparing a complex field of shape {grid.shape} for testing...')
    fld = np.exp(2j * np.pi * 1/3 * grid[0]) \
          * (np.abs(grid[0] - zoom_grid.center[0]) < zoom_grid.extent[0] / 2) \
          * (np.abs(grid[1] - (zoom_grid.center[1] - zoom_grid.extent[1] / 2)) < zoom_grid.extent[1] / 4)
    fld[grid.shape[0] // 2, grid.shape[1] // 2] = 1
    fld += np.cos(2 * np.pi * 0.5 * grid[0]) \
           * (np.abs(grid[0] - zoom_grid.center[0]) < zoom_grid.extent[0] / 2) \
           * (np.abs(grid[1] - (zoom_grid.center[1] + zoom_grid.extent[1] / 2)) < zoom_grid.extent[1] / 4)

    log.info(f'Zooming around {zoom_grid.center}...')
    zoom_fld = interp(fld, to_grid=zoom_grid, from_grid=grid)  # The from_grid argument is redundant here because it is the default
    # Equivalent formulations:
    # zoom_fld = SincInterpolator(fld, grid=grid)[90:110:0.1, 135:165:0.1]  # The grid argument is redundant here because it is the default
    # zoom_fld = SincInterpolator(fld, grid=grid)(*zoom_grid)  # The grid argument is redundant here because it is the default

    log.info('Displaying...')
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(complex2rgb(fld), extent=grid2extent(grid))
    axs[0].set_title('full')
    axs[1].imshow(complex2rgb(zoom_fld), extent=grid2extent(zoom_grid))
    axs[1].set_title('zoom')
    log.info('Done.')
    plt.show()

