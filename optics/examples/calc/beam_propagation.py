import numpy as np
import matplotlib.pyplot as plt

from examples.calc import log

from optics.calc.beam import Beam
from optics.utils.ft.grid import Grid
from optics.utils.display import complex2rgb, grid2extent


if __name__ == '__main__':
    wavelength = 500e-9
    k0 = 2 * np.pi / wavelength
    propagation_axis = 0
    beam_std = 5 * wavelength
    polarization = np.array([0, 1, 1j])

    grid = Grid([1024, 512, 1], step=200e-9)
    grid_slice = grid.project(axes_to_remove=propagation_axis)
    r = np.sqrt(sum(_ ** 2 for _ in grid_slice))

    beam = Beam(grid=grid, vacuum_wavelength=wavelength,
                field=polarization[:, np.newaxis, np.newaxis, np.newaxis] * np.exp(-0.5 * (r / beam_std) ** 2))
    fld = beam.field()

    log.info('Displaying results')
    fig, axs = plt.subplots(1, 1 + 2 * beam.vectorial)
    axs = np.atleast_1d(axs)
    for pol_axis, ax in enumerate(axs):
        ax.imshow(complex2rgb(fld[pol_axis, ..., 0], normalization=2.0),
                  extent=grid2extent(grid.project(axes_to_keep=[0, 1])) * 1e6)
        ax.set(xlabel='z  [$\mu$m]', ylabel='y  [$\mu$m]')

    plt.show()
