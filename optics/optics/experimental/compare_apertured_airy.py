"""
Calculating an apertured Airy beam in two ways.
"""
import time
import logging

import numpy as np
from scipy.special import jv
from scipy.signal import convolve
import matplotlib.pyplot as plt
from optics.utils.ft import Grid
from optics.utils.display import complex2rgb, grid2extent
from optics.utils import ft
from optics.calc import special_beams

log = logging.getLogger(__name__)


def fourier_airy(grid: Grid, alpha: float) -> np.ndarray:
    """
    Calculate the Airy beam for a circular aperture using the Fast Fourier Transform.

    :param grid: A Grid object, representing the spatial sampling grid (see example below).
    :param alpha: The alpha value in wavelengths.
    :return: An np.ndarray with the complex field at the sample points.
    """
    aperture = sum(_**2 for _ in grid.f) < 1
    aperture = aperture * (np.prod(grid.f.step) / np.pi) ** (1/2)

    cubic = np.exp(2j * np.pi * alpha * sum(_**3 for _ in grid.f))
    cubic /= grid.size ** (1/2)  # Normalize intensity after fft

    return ft.fftshift(ft.fftn(aperture * cubic))


def spatial_airy(grid: Grid, alpha: float) -> np.ndarray:
    """
    Calculate the Airy beam for a circular aperture as the convolution of the 2D Airy function and the Airy disk,
    i.e. without using a Fourier transform.

    :param grid: A Grid object, representing the spatial sampling grid (see example below).
    :param alpha: The alpha value in wavelengths.
    :return: An np.ndarray with the complex field at the sample points.
    """
    airy_field = special_beams.Airy(alpha)(*grid)

    r = np.sqrt(sum(_**2 for _ in grid))
    airy_disk = 2 * jv(1, 2 * np.pi * r) / (2 * np.pi * (r + (r==0)))
    airy_disk *= (np.prod(grid.step) * np.pi) ** (1/2)

    blurred_airy_field = convolve(airy_field, airy_disk, mode='same')  # Here is still an integral

    # log.debug(f'disk: {np.linalg.norm(airy_disk)}, field: {np.linalg.norm(airy_field)}, blurred: {np.linalg.norm(blurred_airy_field)}')

    return blurred_airy_field


if __name__ == '__main__':
    from optics.experimental import log

    alpha = 5  # The alpha value in units of wavelength
    grid = Grid([1024, 1024], 0.05)  # The grid object (see online documentation for more details)

    start_time = time.perf_counter()
    fld_s = spatial_airy(grid, alpha=alpha)
    log.info(f'The spatial calculation took {time.perf_counter() - start_time:0.3f}s.')

    start_time = time.perf_counter()
    fld_f = fourier_airy(grid, alpha=alpha)
    log.info(f'The Fourier transform took {time.perf_counter() - start_time:0.3f}s.')

    log.info(f'Norm of spatial calculation result is {np.linalg.norm(fld_s):0.3f}, while that of the Fourier transform is {np.linalg.norm(fld_f):0.3f}.')
    fld_s /= np.amax(np.abs(fld_s))
    fld_f /= np.amax(np.abs(fld_f))

    log.info('Displaying...')
    fig, axs = plt.subplots(1, 3, sharex='all', sharey='all')
    axs[0].imshow(complex2rgb(fld_s), extent=grid2extent(grid))
    axs[0].set(xlabel=r'$x / \lambda$', ylabel=r'$y / \lambda$', title='Spatial Airy')
    axs[1].imshow(complex2rgb(fld_f), extent=grid2extent(grid))
    axs[1].set(xlabel=r'$x / \lambda$', ylabel=r'$y / \lambda$', title='Fourier Airy')
    axs[2].imshow(complex2rgb(fld_f - fld_s), extent=grid2extent(grid))
    axs[2].set(xlabel=r'$x / \lambda$', ylabel=r'$y / \lambda$', title='Difference')

    log.info('Done! Close window to exit.')
    plt.show()

    # TODO: Ensuring that both beams have the same intensity will require going back to the analytical expression for the spatial version.
