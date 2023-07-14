import numpy as np
import scipy
import scipy.constants as const
import logging

from optics.utils.ft import Grid
from optics.utils.display import complex2rgb, grid2extent

log = logging.getLogger(__name__)


def discrete_dipole_approximation(positions, wavenumber, epsilons, cross_sections, current_density, source_position):
    """
    TODO: Finish this implementation before using it!

    :param positions:
    :param wavenumber:
    :param epsilons:
    :param cross_sections:
    :param current_density:
    :param source_position:
    :return:
    """
    positions = np.asarray(positions)
    epsilons = np.asarray(epsilons)

    nb_dipoles = len(positions)
    ndim = positions[0].size

    background_epsilon = 1
    distances = np.zeros((nb_dipoles, nb_dipoles))
    for idx_a, pos_a in enumerate(positions):
        for idx_b, pos_b in enumerate(positions):
            distances[idx_a, idx_b] = np.linalg.norm(pos_a - pos_b)
    # Define Green's function
    background_n = np.lib.scimath.sqrt(background_epsilon)

    def greens_function_r_1d(r):  # plane
        if np.isclose(wavenumber, 0):
            return np.zeros_like(r)
        else:
            return np.exp(1j * r * wavenumber * background_n) / (wavenumber * background_n)

    def greens_function_r_2d(r):  # line
        if np.isclose(wavenumber, 0):
            return np.zeros_like(r)
        else:
            return scipy.special.kv(0, 1j * wavenumber * background_n * r)

    def greens_function_r_3d(r):  # point
        if np.isclose(wavenumber, 0):
            return np.zeros_like(r)
        else:
            return np.exp(1j * wavenumber * background_n * r) / (4 * np.pi * np.maximum(r, np.finfo(r.dtype).eps))
    greens_functions_r = [greens_function_r_1d, greens_function_r_2d, greens_function_r_3d]

    def greens_function(x):
        r = np.sqrt(np.sum(np.abs(x)**2, axis=-1))
        return greens_functions_r[ndim](r)

    # l_p1 = F_inv * (k x k - |k|^2 + background_epsilon) * F
    # v_m1 = np.diag(epsilons) - background_epsilon
    # h =  l_p1 + v_m1  # h = l+v
    b = -1j * wavenumber * const.c * const.mu_0 * current_density

    # precondition with the Green's function for the background permittivity
    # => the field of the background does not affect the field at the dipoles => only consider the dipole positions
    susceptibility = epsilons - background_epsilon  # V / m^2
    potentials = susceptibility * cross_sections  # V
    pre_h = np.empty((nb_dipoles, nb_dipoles), dtype=np.complex)
    for idx in range(nb_dipoles):
        pre_h[idx, :] = 0 * greens_function(positions - positions[idx, :]) * potentials[idx]
        pre_h[idx, idx] = 1 - potentials[idx]  # todo: This is probably incorrect

    # But the source is now convolved to cover the dipole positions:
    def greens_source(pos):
        return b * greens_function_r_1d(pos[..., 0] - source_position[0])  # traveling along x-axis
    pre_b = greens_source(positions)
    log.debug(f'Solving {pre_h}x = {pre_b} for x...')

    # Solve the preconditioned system gamma h x = gamma b for x
    field_at_dipoles = np.linalg.solve(pre_h, pre_b)
    log.debug(f'Amplitude at dipoles: {np.abs(field_at_dipoles)}, phase: {np.angle(field_at_dipoles)}')
    relative_error = np.linalg.norm(pre_h @ field_at_dipoles - pre_b) / np.linalg.norm(pre_b)
    log.debug(f'Relative error = {relative_error}.')
    if relative_error > 1e-6:
        log.warning(f'Did not converge! Relative error = {relative_error}.')

    #
    # Return result
    #
    # Return a function that evaluates the result field on an arbitrary grid
    def field(*ranges):
        grid = Grid.from_ranges(*ranges)
        # f = np.zeros(grid.shape, dtype=np.complex)
        f = greens_source(np.stack(np.broadcast_arrays(*grid), axis=-1))
        f = np.array(np.broadcast_to(f, grid.shape))
        for pos, dipole_field in zip(positions, field_at_dipoles):
            relative_grid = grid - pos
            relative_positions = np.stack(np.broadcast_arrays(*relative_grid), axis=-1)
            f += dipole_field * greens_function(relative_positions)
        return f
    # Store the raw information too
    field.position = positions
    field.permittivity = epsilons
    field.cross_section = cross_sections
    field.angular_frequency = wavenumber * const.c
    field.wavenumber = wavenumber
    field.field = field_at_dipoles

    return field


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from optics.experimental import log

    unit_distance = 1e-6 / 4
    center_wavelength = 500e-9
    grid = Grid([128, 128], unit_distance)
    center_wavenumber = 2 * np.pi / center_wavelength
    travel_distances = Grid(grid.shape[0], unit_distance)  # Grid(64, extent=grid.extent[0]*2)
    wavenumbers = travel_distances.k.as_origin_at_center  # + center_wavenumber
    log.info(f'{2*np.pi / np.asarray(wavenumbers) * 1e9} nm')

    positions = np.array([(0, 0.5), (2, -1), (2, 2)]) * 5e-6
    # positions = np.random.rand(20, 2) * 5 * 1e-6
    epsilons = np.ones(len(positions)) * 1.5**2
    cross_sections = np.prod(grid.step[:2]) * 1e15
    current_density = 1e15  # source
    source_position = [travel_distances[0] * 0, 0]  # source before the sample
    log.info(f'Source at {source_position[0]*1e6:0.1f} um.')

    log.info('Starting discrete dipole approximation...')
    fields = []
    for idx, k in enumerate(wavenumbers):
        log.info(f'Wavelength {2*np.pi / k * 1e9:0.1f} nm...')
        field = discrete_dipole_approximation(positions, k, epsilons, cross_sections, current_density, source_position)
        fields.append(field)
    log.info('Discrete dipole approximation finished.')
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    def calc_field_at_time(fields, t):
        f = np.zeros(grid.shape, dtype=np.complex)
        for field in fields:
            f += np.exp(-1j * field.angular_frequency * t) * field(*grid)
        return f
    log.info('Determining image normalization...')
    t_center = (travel_distances[int(travel_distances.size/2)] - travel_distances[0]) / const.c
    f = calc_field_at_time(fields, t_center)
    normalization = 10 * np.nanmedian(np.abs(f))
    for travel_distance in travel_distances[::8]:
        t = (travel_distance - travel_distances[0]) / const.c
        log.info(f'Calculating field at t = {t*1e15:0.3f} fs, x = {travel_distance*1e6:0.1f} um...')
        f = calc_field_at_time(fields, t)
        f /= normalization
        axs[0].clear()
        axs[0].imshow(complex2rgb(f), extent=grid2extent(grid) * 1e6)
        axs[0].set(xlabel='x [$\mu m$]', ylabel='y [$\mu m$]',
                   title=f't = {t*1e15:0.0f} fs, x = {travel_distance*1e6:0.1f} um')
        axs[1].clear()
        axs[1].plot(grid[0] * 1e6, np.abs(f[:, int(f.shape[1] / 2)]))
        axs[1].plot(grid[0] * 1e6, np.real(f[:, int(f.shape[1] / 2)]))
        axs[1].plot(grid[0] * 1e6, np.imag(f[:, int(f.shape[1] / 2)]))
        axs[1].set(xlabel='[x $\mu m$]', ylabel='E [$V/m^2$]')
        plt.show(block=False)
        plt.pause(0.01)

    plt.show(block=True)
