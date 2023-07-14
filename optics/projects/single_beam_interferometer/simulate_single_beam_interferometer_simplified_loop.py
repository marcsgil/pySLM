import numpy as np
import matplotlib.pyplot as plt

from projects.single_beam_interferometer import log
from optics.utils import ft
from optics.utils.ft import Grid


beam_waist = 0.5e-3  # In the focal plane
grid_zyx = Grid([128, 3*256, 3*256], step=[0.050e-3, 0.050e-3, 0.050e-3])  # propagation (z), projection (y), vertical (x)
grid_yx = grid_zyx.project(axes_to_remove=0)


def get_hermite_gaussian(m: int = 0, n: int = 0):
    normalization_factor = np.sqrt(2**(1 - (n + m))) / (np.sqrt(np.pi) * np.math.factorial(m) * np.math.factorial(n))
    return normalization_factor / beam_waist \
           * np.polynomial.hermite.Hermite.basis(m)(np.sqrt(2) * grid_yx[0] / beam_waist) \
           * np.polynomial.hermite.Hermite.basis(n)(np.sqrt(2) * grid_yx[1] / beam_waist) \
           * np.exp(-sum(_ ** 2 for _ in grid_yx) / beam_waist ** 2)


hermite_gaussian = get_hermite_gaussian(0, 2)
gaussian = get_hermite_gaussian(0, 0)
superposition = 2.75 * hermite_gaussian + gaussian


def propagate_light_field(field, phase=0.0):
    superposition = field
    phase_mask = np.exp(1j * (grid_zyx[2] > 0) * phase)[0]
    beam_section_field_ft = ft.fft2(superposition * phase_mask)
    far_field_after = ft.fftshift(beam_section_field_ft)
    far_field_after_intensity = np.abs(far_field_after)**2
    trace = np.mean(far_field_after_intensity, axis=0)
    return far_field_after_intensity, trace


if __name__ == '__main__':
    log.info('Lets calculate...')
    control = 3
    if control == 1:
        nb_traces = 720
        traces = []
        for _ in np.arange(nb_traces) / nb_traces * 2 * np.pi - np.pi:
            int_far, trace = propagate_light_field(superposition, phase=_)
            traces.append(trace)
        traces = np.asarray(traces)
    elif control == 2:
        gauss_int, trace_gauss = propagate_light_field(gaussian, phase=0.0)
        hg_int, trace_hg = propagate_light_field(hermite_gaussian, phase=0.0)
        sup_int, trace_sup = propagate_light_field(superposition, phase=0.0)
        fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharex='col', sharey='col')
        cmap = 'hot'
        axs[0].imshow(gauss_int, cmap=cmap)
        axs[1].imshow(hg_int, cmap=cmap)
        axs[2].imshow(sup_int, cmap=cmap)
        plt.show()
    elif control == 3:
        gap = np.ones(superposition.shape[1])
        gap[379:390] = 0
        phase = np.pi/2
        gauss_int, trace_gauss = propagate_light_field(gaussian*gap, phase=phase)
        hg_int, trace_hg = propagate_light_field(hermite_gaussian*gap, phase=phase)
        sup_int, trace_sup = propagate_light_field(superposition*gap, phase=phase)
        fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharex='col', sharey='col')
        cmap = 'hot'
        axs[0].imshow(gauss_int/np.amax(sup_int), cmap=cmap)
        axs[1].imshow(hg_int/np.amax(sup_int), cmap=cmap)
        axs[2].imshow(sup_int/np.amax(sup_int), cmap=cmap)
        plt.show()

    # cmap = 'hot'
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex='col', sharey='col')
    # ax.plot(trace/np.amax(trace))
    # ax.plot(gap)
    # plt.show()