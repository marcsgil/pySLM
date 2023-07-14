import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from optics import log
from optics.utils.display import complex2rgb, grid2extent
from optics.utils.ft.grid import Grid
from optics.utils import ft
from projects.confocal_interference_contrast.complex_phase_image import Interferogram

simulate = True

pixel_pitch = 5.3e-6
approximate_period_px = 3
approximate_period = pixel_pitch * approximate_period_px
if not simulate:
    interference_image = Image.open(r'C:\Users\Laurynas\OneDrive - University of Dundee\Work\Pictures\Phase Contrast Era\michealson setup - 2 telescopes\White_tape_sample_2020-10-26_17-21-04.389.png')
    interference_image = np.array(interference_image, dtype=float)[:, :, 0] / 256.0
else:
    test_grid = Grid(shape=(234, 345), step=pixel_pitch)
    r = np.sqrt(test_grid[0]**2 + test_grid[1]**2)
    interference_field = (np.exp(2j*np.pi / (approximate_period * np.sqrt(2)) * (test_grid[0] + test_grid[1])) +
                          np.exp(5j * np.arctan2(test_grid[1], test_grid[0]) * (r < np.amax(r)/4))) / 2 \
        * np.exp(-0.5 * (r / np.amax(r))**2)
    interference_image = np.abs(interference_field) ** 2

complex_phase_image = Interferogram(interference_image, approximate_period=approximate_period_px)

grid = Grid(shape=interference_image.shape, step=pixel_pitch)

frequency_separation = np.linalg.norm(complex_phase_image.registration.shift * grid.f.step)
actual_period = 1 / frequency_separation
log.info(f'Approximate period {approximate_period*1e6:0.3f}um = {approximate_period/pixel_pitch:0.3f}px. ' +
         f'Actual period {actual_period*1e6:0.3f}um = {actual_period/pixel_pitch:0.3f}px')

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(complex_phase_image.raw_interference_image, extent=grid2extent(grid) / 1e-6)
axs[0, 0].set(xlabel='x [$\mu m$]', ylabel='y [$\mu m$]', title='recorded')
axs[1, 0].imshow(complex2rgb(ft.fftshift(ft.fft2(complex_phase_image.measured)), 10), extent=grid2extent(grid.f) * 1e-3)
axs[1, 0].set(xlabel='$f_x$ [cycles/mm]', ylabel='$f_y$ [cycles/mm]', title='spectrum')
axs[0, 1].imshow(complex2rgb(complex_phase_image.measured, 1), extent=grid2extent(grid) / 1e-6)
axs[0, 1].set(xlabel='x [$\mu m$]', ylabel='y [$\mu m]$', title='complex')
axs[1, 1].imshow(complex2rgb(np.exp(1j * np.angle(complex_phase_image.measured)), 1), extent=grid2extent(grid) / 1e-6)
axs[1, 1].set(xlabel='x [$\mu m$]', ylabel='y [$\mu m]$', title='phase only')

plt.show()

