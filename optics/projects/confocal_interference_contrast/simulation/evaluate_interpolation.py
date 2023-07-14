import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from projects.confocal_interference_contrast.simulation import log
from optics.utils import ft
from optics.utils.display import complex2rgb, grid2extent


if __name__ == '__main__':
    interpolate_complex = True

    input_path = r'c:\Users\lab\OneDrive - University of Dundee\Documents\lab\code\python\optics\projects\confocal_interference_contrast\results\intensities_64x200_2023-02-21_17-21-28.npz'
    log.info(f'Reading data from file {input_path}...')

    recorded_data = np.load(input_path)
    step_size = recorded_data['step_size']
    intensities_hr = recorded_data['intensities']
    complex_amplitude_hr = recorded_data['complex_amplitude']  # The high resolution data

    grid_hr = ft.Grid(complex_amplitude_hr.shape, step_size)

    log.info(f'Read file {input_path} with a {grid_hr.shape} image and step size of {grid_hr.step[0] / 1e-6:0.3f}um x {grid_hr.step[1] / 1e-6:0.3f}um')

    grid_lr = ft.Grid(grid_hr.shape // 2, grid_hr.step * 2, first=grid_hr.first)
    bandpass = sum((_ / np.amin(_))**2 for _ in grid_hr.k) < 0.5**2  # circular symmetric blur
    # bandpass = sum((_ / np.amin(_)) for _ in grid_hr.k) < 0.5  # square blur
    complex_amplitude_hr_smooth = ft.ifft2(ft.fft2(complex_amplitude_hr) * bandpass)
    complex_amplitude_lr = complex_amplitude_hr_smooth[::2, ::2]
    # complex_amplitude_lr = ft.interp(complex_amplitude_hr_smooth, from_grid=grid_hr, to_grid=grid_lr)
    intensities_hr_smooth = ft.ifft2(ft.fft2(intensities_hr) * bandpass)
    intensities_lr = intensities_hr_smooth[..., ::2, ::2]

    if interpolate_complex:
        log.info('Interpolating the complex differential phase contrast map.')
        complex_amplitude_lr_int = ft.interp(complex_amplitude_lr, factor=2.0)
        # complex_amplitude_lr_int = ft.interp(complex_amplitude_lr, from_grid=grid_lr, to_grid=grid)
    else:
        log.info('Interpolating the measured intensities.')
        intensities_int = ft.interp(intensities_lr, factor=[1, 2, 2])
        log.info('Calculating the complex differential phase contrast map from the interpolated intensities.')
        intensities_int_ft = ft.fft(intensities_int, axis=0)
        visibility = intensities_int_ft[0].real
        noise_to_signal_ratio = 1e-3
        wiener_filter = visibility / (np.abs(visibility) ** 2 + noise_to_signal_ratio ** 2)
        complex_amplitude_lr_int = intensities_int_ft[-1] * wiener_filter  # The relative contrast and phase
        # complex_amplitude_lr_int[intensities_int_ft[0].real <= 1e-3] = 0.0

    int_error = complex_amplitude_lr_int - complex_amplitude_hr
    int_error_rel_norm = np.linalg.norm(int_error) / np.linalg.norm(complex_amplitude_hr)
    log.info(f'Interpolation error is {int_error_rel_norm * 100:0.3f}%.')

    normalization = 1.0 * np.amax(np.abs(complex_amplitude_hr))
    log.info(f'Displaying with maximum value of {normalization:0.3f}...')

    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=[16, 9])
    axs[0, 0].imshow(complex2rgb(complex_amplitude_hr / normalization), extent=grid2extent(grid_hr) / 1e-6)
    axs[0, 0].set_title(f'high-res step {grid_hr.step[0] / 1e-6:0.3f}um')
    axs[0, 1].imshow(complex2rgb(complex_amplitude_lr / normalization), extent=grid2extent(grid_lr) / 1e-6)
    axs[0, 1].set_title(f'low-res step {grid_lr.step[0] / 1e-6:0.3f}um')
    axs[1, 0].imshow(complex2rgb(complex_amplitude_lr_int / normalization), extent=grid2extent(grid_hr) / 1e-6)
    axs[1, 0].set_title(f'interpolation step {grid_hr.step[0] / 1e-6:0.3f}um')
    axs[1, 1].imshow(complex2rgb(int_error / normalization), extent=grid2extent(grid_hr) / 1e-6)
    axs[1, 1].set_title(f'error={int_error_rel_norm * 100:0.1f}% step {grid_hr.step[0] / 1e-6:0.3f}um')
    for ax in axs.ravel():
        ax.set(xlabel='y [$\mu$m]', ylabel='x [$\mu$m]')

    log.info('Done!')
    plt.show()
