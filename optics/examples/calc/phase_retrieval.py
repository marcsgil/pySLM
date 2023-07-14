import numpy as np
import matplotlib.pyplot as plt

from examples.calc import log
from optics.utils import ft
from optics.utils.ft import Grid
from optics.utils.reference_object import usaf1951
from optics.utils.display import complex2rgb, grid2extent
from optics.calc import retrieve_phase

if __name__ == '__main__':
    #
    # A phase retrieval example to replicate a USAF target in a disk, ignoring everything outside of it.
    #
    grid = Grid(np.ones(2) * 64)
    f2 = sum(_**2 for _ in grid.f)
    target_area = sum(_ ** 2 for _ in grid) < (np.amin(grid.shape) / 3) ** 2
    target_intensity_in = np.exp(-0.5 * f2 / 0.1**2)
    target_in = np.sqrt(target_intensity_in)
    target_intensity_out = np.array(usaf1951(grid.shape)) / 255.0
    log.info('Converting target intensity to field.')
    target_out = np.sqrt(target_intensity_out) * np.exp(1j * np.arctan2(grid.f[1], grid.f[0]))
    target_out *= target_area
    target_out /= np.amax(np.abs(target_out))

    log.info('Rescaling object amplitude target.')
    target_out_norm = np.linalg.norm(target_out)
    scaling_factor_in = np.linalg.norm(target_in) / target_out_norm

    log.info('Determining upper and lower bounds for the magnitudes.')
    target_magnitude_min_in = np.abs(target_in) / scaling_factor_in
    target_magnitude_max_in = np.abs(target_in) / scaling_factor_in
    target_magnitude_min_out = np.abs(target_out) * target_area
    target_magnitude_max_out = np.abs(target_out) + (1 - target_area) * 1e16

    log.info('Marking output area that is not of interest with NaN.')
    target_out[np.logical_not(target_area)] = np.NaN

    log.info('Retrieving phase...')
    estimate_in = retrieve_phase(target_out, max_iter=100, max_rel_residual=1e-3)
    estimate_out = ft.fftn(estimate_in)
    residual_out = (np.abs(estimate_out) - np.abs(target_out)) * target_area
    rel_residual_out_norm = np.linalg.norm(residual_out) / target_out_norm

    log.info('Scaling the source field back to original for display.')
    estimate_in *= scaling_factor_in

    log.info('Displaying...')
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(complex2rgb(ft.fftshift(target_in)), extent=grid2extent(grid.f.as_origin_at_center))
    axs[0, 0].set_title(f'hologram before [{np.amin(np.abs(target_in)):0.3f}, {np.amax(np.abs(target_in)):0.3f}]')
    axs[0, 1].imshow(complex2rgb(target_out), extent=grid2extent(grid))
    axs[0, 1].set_title('target')
    axs[1, 0].imshow(complex2rgb(ft.fftshift(estimate_in)), extent=grid2extent(grid.f.as_origin_at_center))
    axs[1, 0].set_title(f'estimated hologram [{np.amin(np.abs(estimate_in)):0.3f}, {np.amax(np.abs(estimate_in)):0.3f}]')
    axs[1, 1].imshow(complex2rgb(estimate_out), extent=grid2extent(grid))
    axs[1, 1].set_title(f'estimated [$\Delta$={rel_residual_out_norm:0.3e}]')

    log.info('Done.')
    plt.show(block=True)

