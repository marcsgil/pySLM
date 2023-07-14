import numpy as np
import matplotlib.pyplot as plt

from projects.single_beam_interferometer import log
from optics.utils import ft
from optics.utils.ft import Grid
from optics.calc.beam import BeamSection, Beam
from optics.utils.display import complex2rgb, grid2extent


if __name__ == '__main__':
    log.info('Lets calculate...')
    focal_length = 100e-3
    wavelength = 633e-9
    beam_waist = 0.5e-3  # In the focal plane
    blade_size = [5e-3, 100e-3, 0.1e-3]  # z, y, x
    refractive_index_blade = 2.0 + 1.0j
    object_size = [1e-3, 100e-3, 2e-3]  # z, y, x
    object_position = [0, 0, -(blade_size[2] + object_size[2]) / 2]
    refractive_index_object = 1.3312
    reference_position = [0, 0, (blade_size[2] + object_size[2]) / 2]
    refractive_index_reference = 1.331

    cmap = 'hot'

    #s_Rayleigh_length = 2 / np.pi * np.arctan2(z, rayleigh_distance)
    grid_zyx = Grid([128, 256, 256], step=[0.050e-3, 0.050e-3, 0.050e-3])  # propagation (z), projection (y), vertical (x)
    grid_yx = grid_zyx.project(axes_to_remove=0)

    log.info('Defining the object.')
    refractive_index = 1.0 + (refractive_index_blade - 1.0) * np.prod(np.array(
        [np.abs(grid_zyx[_]) <= blade_size[_] / 2 for _ in range(3)],
        dtype=object))  # blade
    refractive_index += (refractive_index_object - 1.0) * np.prod(np.array(
        [np.abs(grid_zyx[_] - object_position[_]) <= object_size[_] / 2 for _ in range(3)],
        dtype=object))  # object
    refractive_index += (refractive_index_reference - 1.0) * np.prod(np.array(
        [np.abs(grid_zyx[_] - reference_position[_]) <= object_size[_] / 2 for _ in range(3)],
        dtype=object))  # reference

    inside_blade = np.amax(refractive_index.imag, axis=1) > 0.5
    maximum_refractive_index = np.amax(refractive_index.real, axis=1)
    inside_object = maximum_refractive_index == refractive_index_object
    inside_reference = maximum_refractive_index == refractive_index_reference
    overlay = np.stack([inside_blade,
                        inside_blade + inside_object,
                        inside_blade + inside_reference,
                        inside_blade + 0.25 * (inside_reference + inside_object)], axis=-1)

    def get_hermite_gaussian(m: int = 0, n: int = 0):
        normalization_factor = np.sqrt(2**(1 - (n + m))) / (np.sqrt(np.pi) * np.math.factorial(m) * np.math.factorial(n))
        return normalization_factor / beam_waist \
            * np.polynomial.hermite.Hermite.basis(m)(np.sqrt(2) * grid_yx[0] / beam_waist) \
            * np.polynomial.hermite.Hermite.basis(n)(np.sqrt(2) * grid_yx[1] / beam_waist) \
            * np.exp(-sum(_ ** 2 for _ in grid_yx) / beam_waist ** 2)

    hermite_gaussian = get_hermite_gaussian(0, 2)
    gaussian = get_hermite_gaussian(0, 0)
    superposition = 2 * hermite_gaussian + gaussian

    far_field_before = BeamSection(grid_yx, vacuum_wavelength=wavelength, field=superposition)
    combined_beam = Beam(grid_zyx, vacuum_wavelength=wavelength, field=superposition)

    log.info('Start of calculations...')
    field_slice = np.zeros(grid_zyx.shape[[0, 2]], dtype=np.complex128)
    projected_intensity = np.zeros(grid_zyx.shape[[0, 2]])
    for z_idx, beam_section in enumerate(combined_beam.__iter__(refractive_index=refractive_index)):
        if z_idx % 100 == 0:
            log.info(f'Calculating slice {z_idx} / {grid_zyx.shape[0]} ...')
        field_slice[z_idx, :] = beam_section.field[0, 0, grid_zyx.shape[1] // 2]
        projected_intensity[z_idx, :] = np.mean(np.abs(beam_section.field[0, 0]) ** 2, axis=0)

    log.info('Displaying...')
    display_top_view_extent = grid2extent(grid_zyx.project(axes_to_remove=1))
    display_top_view_extent = (*display_top_view_extent[2:] / 1e-3, *display_top_view_extent[:2] / 1e-3)  # Transpose and scale

    display_k_space_extent = grid2extent(grid_yx.k.as_origin_at_center * 1e-3)

    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex='col', sharey='col')

    axs[0, 0].imshow(complex2rgb(ft.fftshift(far_field_before.field_ft[0, 0]), normalization=1.25), extent=display_k_space_extent)
    axs[0, 0].set(title='far field before', xlabel='$k_y$ [rad/mm]', ylabel='$k_x$ [rad/mm]')

    axs[0, 1].imshow(complex2rgb(field_slice.transpose(), normalization=1.25), extent=display_top_view_extent)  # saturate a little
    axs[0, 1].set(title='field section', xlabel='z [mm]', ylabel='x [mm]')

    axs[0, 2].imshow(complex2rgb(ft.fftshift(beam_section.field_ft[0, 0]), normalization=1.25), extent=display_k_space_extent)
    axs[0, 2].set(title='far field after', xlabel='$k_y$ [rad/mm]', ylabel='$k_x$ [rad/mm]')

    # Intensity
    axs[1, 0].imshow(np.abs(ft.fftshift(far_field_before.field_ft[0, 0]))**2, cmap=cmap, extent=display_k_space_extent)
    axs[1, 0].set(title='far field before', xlabel='$k_y$ [rad/mm]', ylabel='$k_x$ [rad/mm]')

    axs[1, 1].imshow(projected_intensity.transpose(), cmap=cmap, extent=display_top_view_extent)
    axs[1, 1].set(title='intensity mean', xlabel='z [mm]', ylabel='x [mm]')

    axs[1, 2].imshow(np.abs(ft.fftshift(beam_section.field_ft[0, 0]))**2, cmap=cmap, extent=display_k_space_extent)
    axs[1, 2].set(title='far field after', xlabel='$k_y$ [rad/mm]', ylabel='$k_x$ [rad/mm]')

    for _ in axs[:, 1]:
        _.imshow(overlay.transpose((1, 0, 2)), extent=display_top_view_extent)
        _.set_aspect(abs((display_top_view_extent[1] - display_top_view_extent[0]) / (display_top_view_extent[3] - display_top_view_extent[2])) * 9 / 16)

    log.info('Done!')
    plt.show(block=True)
