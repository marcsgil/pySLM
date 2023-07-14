import numpy as np
import matplotlib.pylab as plt
from pathlib import Path

from optics.calc.interferogram import Interferogram, InterferogramOld
from optics.utils.array import add_dims_on_right
from optics.utils.display import complex2rgb, grid2extent, format_image_axes, complex_color_legend
from optics.utils import ft
from optics.utils.ft.subpixel import Reference, roll, roll_ft

from projects.confocal_interference_contrast.polarization_memory_effect import log

from tkinter import filedialog as fd
from tqdm import tqdm


def display_polarization_memory_effect(interference_file_path: Path,
                                       initial_reference_file_path: Path,
                                       display: bool = True,
                                       display_diagnostics: bool = False, use_separate_initial_reference: bool = True,
                                       spatially_correct_using_wiener_filter: bool = True,
                                       remove_tip_tilt_and_piston: bool = True
                                       ):
    """
    This function analyses the files produced by polarization_memory_effect_image_aquisition and displays the results.

    :param interference_file_path: The file with the scattering medium interferograms for each input and output polarization.
    :param initial_reference_file_path: The file with the reference experiment, typically pure agarose (default, the same as above).
    :param display: If True: display the amplitude and angle image for linear and circular polarizations.
    :param display_diagnostics: If True: show the 4x4 matrix with all DLAR input and output polarizations.
    :param use_separate_initial_reference: If set to False, use the reference contained in the same file as the scattering experiment. Leave True unless you know what you are doing.
    :param spatially_correct_using_wiener_filter: Filters the amplitude of the back aperture interferogram. Leave True unless you know what you are doing.
    :param remove_tip_tilt_and_piston: Removes a shift in the interference pattern. Leave True unless you know what you are doing. A warning will be issued if it too large.

    :return:
    """
    output_file_path = interference_file_path.with_suffix('.png')

    interference_file_path = interference_file_path.resolve()

    if initial_reference_file_path is None or not use_separate_initial_reference:
        initial_reference_file_path = interference_file_path

    log.info(f'Loading initial reference data from {initial_reference_file_path}...')
    initial_reference_data = np.load(initial_reference_file_path)

    log.info(f'Loading experiment data from {interference_file_path}...')
    experiment_data = np.load(interference_file_path)

    # Extract each matrix
    initial_exposure_times = initial_reference_data['initial_exposure_times']
    initial_interference_images = initial_reference_data['initial_interference_images']  # Actual a diagonal polarization input
    calibration_amplitude = experiment_data['calibration_amplitude']
    exposure_times = experiment_data['exposure_times']
    polarization_amplitudes = experiment_data['polarization_amplitudes']
    interference_image_pairs = experiment_data['interference_image_pairs']
    log.info(f'Spot amp. corr. {calibration_amplitude}. Image: {initial_interference_images[0].shape}, exp. time {initial_exposure_times/1e-3}ms -> {exposure_times/1e-3}ms.')

    log.info('Analyzing reference interferograms. This interprets the intensity interferograms as complex images.')
    reference_interferograms = [InterferogramOld(_) for _ in initial_interference_images]
    grid = reference_interferograms[0].grid
    log.info('Detecting image translation between the two output polarization channels based on the amplitude only.')
    registration_of_second = Reference(np.abs(reference_interferograms[0])).register(np.abs(reference_interferograms[1]))
    inter_channel_shift = registration_of_second.shift
    inter_channel_factor = registration_of_second.factor.real # using real values because our measurement are intensity images
    log.info(f'Found a shift of {inter_channel_shift} px between the two channels, and the second is brighter by a factor of {inter_channel_factor:0.3f}.')

    log.info('Calculating filter for per-pixel phase and amplitude correction filters.')
    initial_interferogram_images = np.array([np.asarray(_) for _ in reference_interferograms])
    log.info(f'Max contrast of initial interferograms: {np.amax(np.abs(initial_interferogram_images), axis=(-2, -1))}')
    noise_level = 0.30 * np.amax(np.abs(initial_interferogram_images))  # TODO: Check if this is a good value
    if spatially_correct_using_wiener_filter:
        wiener_filters = np.conj(initial_interferogram_images) / (np.abs(initial_interferogram_images) ** 2 + noise_level ** 2)
    else:
        wiener_filters = 1

    log.info('Selecting only the input polarizations D,L,A,R for processing.')
    polarization_names = 'DLAR'
    polarization_phasors = [1, 1j, -1, -1j]  # == np.exp(2j * np.pi * np.arange(len(polarization_names)) / len(polarization_names))
    # polarization_names = 'HLVR'
    # polarization_phasors = [0, 1j, np.inf, -1j]  # == np.exp(2j * np.pi * np.arange(len(polarization_names)) / len(polarization_names))
    measurement_indexes = []
    for polarization_phasor in polarization_phasors:
        measurement_indexes.append(np.where(polarization_phasor == polarization_amplitudes)[0][0])
    interference_image_pairs = interference_image_pairs[measurement_indexes]
    polarization_amplitudes = polarization_amplitudes[measurement_indexes]

    log.info('Extracting, correcting, and aligning fields for each polarization...')
    hv_field_pairs = []
    for img_pair in interference_image_pairs:  # loop over all measurements
        #     interferograms = [Interferogram(img) for img, ref in zip(img_pair, reference_interferograms)]
        #     for img_int, ref in zip(interferograms, reference_interferograms):
        #         if np.linalg.norm(img_int.fringe_frequency - ref.fringe_frequency) > 1e-2:
        #             log.info(f'Ignoring a tip-tilt of {img_int.fringe_frequency - ref.fringe_frequency} periods.\n' +
        #                      f'The spatial frequency spectrum has shifted from {ref.fringe_frequency} to {img_int.fringe_frequency}.')
        #         rel_amplitude = img_int.fringe_amplitude / ref.fringe_amplitude
        #         log.info(f'Detected a phase change of {np.angle(rel_amplitude) * 180 / np.pi:0.1f} degrees and an amplitude change of {np.abs(rel_amplitude)}.')
        # else:
        interferograms = [InterferogramOld(img,
                                           fringe_frequency=ref.fringe_frequency, fringe_amplitude=ref.fringe_amplitude,
                                           ) for img, ref in zip(img_pair, reference_interferograms)]
        # convert to numpy array for the next steps
        hv_field_pair = wiener_filters * np.stack([_.__array__() for _ in interferograms])  # A pair of images
        # Correct inter-image shift based on the initial image pair
        hv_field_pair[1] = roll(hv_field_pair[1], -inter_channel_shift) / inter_channel_factor
        hv_field_pairs.append(hv_field_pair)
    hv_field_pairs = np.asarray(hv_field_pairs)  # Dimensions: input_pol, output_pol_pair, vertical, horizontal
    if display_diagnostics:
        display_normalization = np.amax(np.abs(hv_field_pairs))
        fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex='all', sharey='all')
        fig.canvas.manager.set_window_title(str(interference_file_path))
        for fld_row, ax_row, measurement_name in zip(hv_field_pairs.transpose([1, 0, 2, 3]), axs, ['trans', 'refl']):
            for fld, ax, input_pol in zip(fld_row, ax_row, polarization_names):
                ax.imshow(complex2rgb(fld / display_normalization), extent=grid2extent(grid))
                format_image_axes(ax, scale=grid.extent[0] / 8, unit='px',
                                  title=f'{input_pol}$\\rightarrow${measurement_name}')
        fig.tight_layout()
        plt.show(block=False)

    log.info(f'Intensities per input polarization: {np.sqrt(np.mean(np.abs(hv_field_pairs) ** 2, axis=(-3, -2, -1)))}')

    log.info(f'Simulating measurements for all polarizations {polarization_names} -> {polarization_names}.')
    # hv_field_pairs dimensions: input_pol, output_pol_pair, vertical, horizontal
    measurement_matrix = (  # 4x4 complex images
            hv_field_pairs[:, 0, :, :] +
            hv_field_pairs[:, 1, :, :] * add_dims_on_right(polarization_amplitudes, 3).conj()
    )  # Dimensions output_pol, input_pol, vertical, horizontal

    if display_diagnostics:
        display_normalization = np.amax(np.abs(measurement_matrix))
        fig, axs = plt.subplots(4, 4, figsize=(16, 12), sharex='all', sharey='all')
        fig.canvas.manager.set_window_title(str(interference_file_path))
        for im_row, ax_row, input_pol in zip(measurement_matrix, axs, polarization_names):
            for im, ax, output_pol in zip(im_row, ax_row, polarization_names):
                ax.imshow(complex2rgb(im / display_normalization), extent=grid2extent(grid))
                format_image_axes(ax, scale=grid.extent[0] / 8, unit='px',
                                  title=f'{input_pol}$\\rightarrow${output_pol}')
        fig.tight_layout()
        plt.show(block=False)

    log.info('Simulating interference of two orthogonal inputs (D-A or L-R) at 4 different phase delays.')
    # Consider linear and circular separately
    linear_matrix = measurement_matrix[0, :] + measurement_matrix[2, [2, 3, 0, 1]]  # pol_change, vertical, horizontal
    circular_matrix = measurement_matrix[1, [1, 2, 3, 0]] + measurement_matrix[3, [3, 0, 1, 2]]  # pol_change, vertical, horizontal
    linear_circular_matrix = np.stack((linear_matrix, circular_matrix))  # 2 complex images: circular, pol_change, vertical, horizontal
    linear_circular_matrix_intensity = np.abs(linear_circular_matrix) ** 2  # Interfere
    linear_circular_matrix_intensity_ft = ft.fft(linear_circular_matrix_intensity, axis=1)  # Fourier transform over the 4 types of detector polarization/phase delays: DLAR
    linear_circular_background = linear_circular_matrix_intensity_ft[:, 0]  # circular or not, vertical, horizontal
    linear_circular_phasor = linear_circular_matrix_intensity_ft[:, 1]  # circular or not, vertical, horizontal
    if remove_tip_tilt_and_piston:
        log.info('Removing tip and tilt...')
        linear_circular_phasor_regs = [Reference(ndim=2).register(subject_ft=ft.ifftshift(_)) for _ in linear_circular_phasor]
        linear_circular_phasor_shifts = [_.shift for _ in linear_circular_phasor_regs]
        linear_circular_phasor_factors = [_.factor for _ in linear_circular_phasor_regs]
        log.info(f'Canceling tip/tilt of {linear_circular_phasor_shifts} waves...')
        max_periods = np.amax(np.linalg.norm(linear_circular_phasor_shifts, axis=-1))
        if max_periods > 0.1:
            log.warning(f'Detected a large tip/tilt of {max_periods:0.1f} waves!')
        linear_circular_phasor = [ft.fftshift(roll_ft(ft.ifftshift(ph), shift=-reg.shift)) for reg, ph in zip(linear_circular_phasor_regs, linear_circular_phasor)]
    linear_circular_phasor_mean = np.mean(linear_circular_phasor, axis=(-2, -1))
    linear_circular_background_mean = np.mean(linear_circular_background, axis=(-2, -1))
    linear_circular_phasor_rel = linear_circular_phasor_mean / linear_circular_background_mean * 2  # Times 2 because the maximum is 0.50
    log.info(f'Linear|Circular: phase={np.angle(linear_circular_phasor_rel)*180/np.pi} degrees, contrast={np.abs(linear_circular_phasor_rel)}')

    if display:
        log.info('Displaying interference of the two channels...')
        display_normalization = 1/4 * np.amax(np.abs(linear_circular_phasor))
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex='all', sharey='all')
        fig.canvas.manager.set_window_title(str(interference_file_path))
        amp_angle = np.zeros((2, 2))
        count = 0
        for linear_or_circular_image, ax, pol_name, phasor in zip(linear_circular_phasor, axs, ['linear', 'circular'], linear_circular_phasor_rel):
            ax.imshow(complex2rgb(linear_or_circular_image / display_normalization), extent=grid2extent(grid))
            format_image_axes(ax, scale=grid.extent[0] / 8, unit='px',
                              title=f'{pol_name}: |A| = {np.abs(phasor)*100:0.1f}%, $\\angle A={np.angle(phasor)*180/np.pi:0.1f}^\\circ$')
            amp_angle[count, 0] = np.abs(phasor)*100
            amp_angle[count, 1] = np.angle(phasor)*180/np.pi
            count += 1
        # fig.tight_layout()
        #
        # # Add a color legend to the potential plot
        # subplot_shape = [1, 2]
        # legend_size = 1/3 / np.array(subplot_shape)
        # ax_legend = fig.add_axes([1/subplot_shape[1] - legend_size[1], 1 - legend_size[0], legend_size[1], legend_size[0]])
        # complex_color_legend.draw(ax_legend, foreground_color='white', saturation=1.50)
        # plt.show(block=False)
        # plt.pause(0.01)
        log.debug(f'Saving to {output_file_path}...')
        fig.savefig(output_file_path)
        log.info(f'Saved to {output_file_path}.')
        # fig2, (axt,axr) = plt.subplots(1, 2)
        # axt.imshow(interference_image_pairs[2,0])
        # axt.axis('off')
        # axt.set_title('Transmitted camera')
        # axr.imshow(interference_image_pairs[2,1])
        # axr.axis('off')
        # axr.set_title('Reflected camera')
        # plt.show(block=True)
    return np.concatenate(amp_angle, axis=None)


if __name__ == '__main__':
    input_directories = [r'C:\Users\lab\OneDrive - University of Dundee\Documents\lab\code\python\optics\projects\confocal_interference_contrast\polarization_memory_effect\results',
                         r'C:\Users\tvettenburg\Downloads\polarization_memory_effect']

    ref_status = input('1 -> None; 2 -> New reference')
    if ref_status == '1':
        initial_reference_file_path = None
    elif ref_status == '2':
        initial_reference_file_path = Path(fd.askopenfilename(initialdir=input_directories[0], title='Choose the reference file')) #input('What reference do you want to use?')

    interference_files_paths = fd.askopenfilenames(initialdir=input_directories[0])
    number_concentrations = 20
    amps_angles = np.zeros((number_concentrations, 20, 4))
    concentrations_count = np.zeros(number_concentrations, dtype='int')
    for _ in tqdm(range(len(interference_files_paths))):
        concentration = int(interference_files_paths[_].split('polarization_memory_effect_')[1].split('c')[0])
        interference_file_path = Path(interference_files_paths[_])
        amp_angle = display_polarization_memory_effect(interference_file_path=interference_file_path,
                                                       initial_reference_file_path=initial_reference_file_path)
        amps_angles[concentration, concentrations_count[concentration], :] = amp_angle[:]
        concentrations_count[concentration] += 1
    linear_mean_amp = np.zeros(number_concentrations)
    linear_mean_angle = np.zeros(number_concentrations)
    circular_mean_amp = np.zeros(number_concentrations)
    circular_mean_angle = np.zeros(number_concentrations)

    linear_std_amp = np.zeros(number_concentrations)
    linear_std_angle = np.zeros(number_concentrations)
    circular_std_amp = np.zeros(number_concentrations)
    circular_std_angle = np.zeros(number_concentrations)

    for _ in range(number_concentrations):
        linear_mean_amp[_] = np.mean(amps_angles[_, 0:concentrations_count[_], 0])
        linear_mean_angle[_] = np.mean(amps_angles[_, 0:concentrations_count[_], 1])
        circular_mean_amp[_] = np.mean(amps_angles[_, 0:concentrations_count[_], 2])
        circular_mean_angle[_] = np.mean(amps_angles[_, 0:concentrations_count[_], 3])

        linear_std_amp[_] = np.std(amps_angles[_, 0:concentrations_count[_], 0])
        linear_std_angle[_] = np.std(amps_angles[_, 0:concentrations_count[_], 1])
        circular_std_amp[_] = np.std(amps_angles[_, 0:concentrations_count[_], 2])
        circular_std_angle[_] = np.std(amps_angles[_, 0:concentrations_count[_], 3])

    # print(concentrations_count)
    # print(linear_mean_amp)
    # print(circular_mean_amp)
    #
    # fig = plt.figure()
    # x = np.linspace(0, 10, number_concentrations)
    # plt.errorbar(x, linear_mean_amp, linear_std_amp, label='Linear polarization')
    # plt.errorbar(x, circular_mean_amp, circular_std_amp, label='Circular polarization')
    # plt.legend()
    # plt.show()

    if ref_status == '2':
        ref_name = initial_reference_file_path.name.removesuffix('.npz')
        output_file_path = Path(f'../../confocal_interference_contrast/polarization_memory_effect/results/06-09-2022/results_for_ref_{ref_name}').resolve()
        np.savez(output_file_path,
                linear_mean_amp = linear_mean_amp , linear_std_amp = linear_std_amp,
                linear_mean_angle = linear_mean_angle , linear_std_angle = linear_std_angle,
                circular_mean_amp = circular_mean_amp , circular_std_amp = circular_std_amp,
                circular_mean_angle = circular_mean_angle , circular_std_angle = circular_std_angle)

    log.info('Done.')
