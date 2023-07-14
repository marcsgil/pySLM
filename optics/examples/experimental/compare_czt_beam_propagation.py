import numpy as np
import matplotlib.pyplot as plt
import time

from optics.utils import ft, rms
from optics.utils.display import complex2rgb, grid2extent

from examples.calc import log


if __name__ == '__main__':
    # grid = ft.Grid(8, step=1)
    # grid_czt_f = ft.Grid(shape=4, step=1 / grid.shape, origin_at_center=False)
    # fld = np.zeros(grid.shape)
    # fld[:3] = [1, -1, 2]
    # fld_ft = ft.fftn(ft.ifftshift(fld))
    # fld_czt = ft.zoomftn(fld, x=grid, f=grid_czt_f)
    # fld_ft_crop = ft.ifftshift(ft.fftshift(fld_ft)[grid.shape[-1]//2 - grid_czt_f.shape[-1]//2 + np.arange(grid_czt_f.shape[-1])])
    #
    # log.info(fld_ft_crop)
    # log.info(fld_czt)
    # log.info(f'rms = {rms(fld_ft_crop, fld_czt):0.3f}')
    #
    # exit()

    pupil_grid_shape = 128
    numerical_aperture = 0.80
    wavelength = 1e-6

    k_max = 2 * np.pi * numerical_aperture / wavelength

    grid = ft.Grid(shape=np.full(2, 512), step=wavelength/4)
    grid_z = ft.Grid(step=grid.step[-1], extent=20e-6)
    grid_k = grid.k
    k0 = 2 * np.pi / wavelength
    k2 = sum(_ ** 2 for _ in grid_k)
    kz = np.sqrt(np.maximum(0.0, k0**2 - k2))
    sub_grid = ft.Grid(step=grid.step, shape=grid.shape // 4)
    pupil_grid_k = ft.Grid(step=sub_grid.k.step, extent=2 * k0)
    sub_k2 = sum(_ ** 2 for _ in pupil_grid_k)
    sub_kz = np.sqrt(np.maximum(0.0, k0**2 - sub_k2))

    grid_display_ref = grid_z @ grid.project(axes_to_keep=-1)
    grid_display = grid_z @ sub_grid.project(axes_to_keep=-1)

    def get_n(g):
        stuff = (sum((_ - 1e-6) ** 2 for _ in g) < 3e-6 ** 2)
        return 1.0 + 0.0 * stuff
    refractive_index_ref = get_n(grid_display_ref)
    refractive_index = get_n(grid_display)

    def pupil(u, v) -> np.ndarray:
        return u**2 + v**2 < 1

    log.info('Propagating...')
    fld_ref_ft = pupil(*(grid_k / k_max)).astype(np.complex64)
    fld_ft = pupil(*(pupil_grid_k / k_max)).astype(np.complex64)
    fld_ref = []
    fld = []
    timings = []
    fld_ref_iter_ft = fld_ref_ft * np.exp(1j * grid_z.first * kz)
    fld_iter_ft = fld_ft * np.exp(1j * grid_z.first * sub_kz)
    for n_ref, n in zip(refractive_index_ref, refractive_index):
        start_time = time.perf_counter()
        fld_ref_iter_ft *= np.exp(1j * grid_z.step * kz) * pupil(*(grid_k / k0))
        fld_ref_iter = ft.fftshift(ft.ifftn(fld_ref_iter_ft))  # from grid_k to grid
        dn_ref = n_ref - 1
        if np.any(dn_ref != 0):
            fld_ref_iter *= np.exp(2j * np.pi * dn_ref)
            fld_ref_iter_ft = ft.fftn(ft.ifftshift(fld_ref_iter))  # from grid to grid_k
        fld_ref.append(fld_ref_iter[grid.shape[0] // 2])
        timings.append([time.perf_counter() - start_time])

        start_time = time.perf_counter()
        frac_outside = 1 - np.linalg.norm(fld_iter_ft * pupil(*(pupil_grid_k / k_max))) / np.linalg.norm(fld_iter_ft)
        if frac_outside > 1e-3:
            log.info(f'fraction outside NA: {frac_outside:0.6f}')
        fld_iter_ft *= np.exp(1j * grid_z.step * sub_kz) * pupil(*(pupil_grid_k / k0))
        fld_iter = ft.izoomftn(fld_iter_ft, k=pupil_grid_k, x=sub_grid)
        dn = n - 1
        if np.any(dn != 0):
            fld_iter *= np.exp(2j * np.pi * dn)
            fld_iter_ft = ft.zoomftn(fld_iter, x=sub_grid, k=pupil_grid_k)
        fld.append(fld_iter[sub_grid.shape[0] // 2])
        timings[-1].append(time.perf_counter() - start_time)

    timings = np.asarray(timings)
    fld_diff = np.asarray(fld_ref)[:,
               grid.shape[-1]//2 - sub_grid.shape[-1]//2 + np.arange(sub_grid.shape[-1])] - np.asarray(fld)
    difference = np.linalg.norm(fld_diff) / np.linalg.norm(fld)
    log.info(f'FFT & CZT took {np.mean(timings, axis=0)}+-{np.std(timings, axis=0)}. Difference = {difference}')

    log.info('Displaying...')
    normalization = np.amax(fld_ref) / 2
    fig, axs = plt.subplots(2, 2, figsize=[16, 10], sharex='all', sharey='all')
    axs = axs.ravel()
    axs[0].imshow(complex2rgb(ft.fftshift(fld_ref_ft), 1), extent=grid2extent(grid_k) * 1e-3)
    axs[0].set(xlabel='$k_x$  [rad/mm]', ylabel='$k_y$  [rad/mm]')
    axs[0].set_title('pupil')
    axs[1].imshow(complex2rgb(fld_ref / normalization), extent=grid2extent(grid_display_ref) / 1e-6)
    axs[1].set(xlabel=r'x  [$\mu$m]', ylabel=r'z  [$\mu$m]')
    axs[1].set_title('fft')
    axs[2].imshow(complex2rgb(fld / normalization), extent=grid2extent(grid_display) / 1e-6)
    axs[2].set(xlabel=r'x  [$\mu$m]', ylabel=r'z  [$\mu$m]')
    axs[2].set_title('czt')
    axs[3].imshow(complex2rgb(fld_diff / normalization * 10), extent=grid2extent(grid_display) / 1e-6)
    axs[3].set(xlabel=r'x  [$\mu$m]', ylabel=r'z  [$\mu$m]')
    axs[3].set_title(f'diff {difference:0.3f}')

    log.info('Done.')
    plt.show()

