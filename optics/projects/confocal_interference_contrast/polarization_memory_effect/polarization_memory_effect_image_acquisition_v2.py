# code to measure the intensity of the spots used at the optical memory effect project
# test code - Braian

import numpy as np
import time
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from projects.confocal_interference_contrast.polarization_memory_effect import log
from optics.instruments.cam.ids_cam import IDSCam, IDSCamError
from optics.instruments.slm import PhaseSLM
from optics.utils.display import  grid2extent
from optics.calc import correction_from_pupil
from optics.utils.ft import Grid
from process_polarization_memory_effect import display_polarization_memory_effect
from optics.instruments.stage.nanostage import NanostageLT3

input_config_file_path = Path('../../../results/aberration_correction_2022-05-12_15-54-37.439768.npz').resolve()

# todo: scan in 6 polarizations (HVDARL)
#

def display(cam_ax_ims, interference_images):
    for cam_ax_im, interference_image in zip(cam_ax_ims, interference_images):
        saturated = interference_image >= 1.0
        marked_image = np.repeat(interference_image[..., np.newaxis], repeats=3, axis=-1)
        marked_image[:, :, 1:] *= 1 - saturated[..., np.newaxis]
        cam_ax_im.set_data(marked_image)
        cam_ax_im.set_clim(0, np.amax(marked_image)/2)


def create_spots(relative_amplitude: complex = 1.0):
    """Function that creates two focal spots with a relative complex amplitude of the second spot
    :param relative_amplitude: The amplitude of the second spot with respect to the first.
    """
    if not np.isinf(relative_amplitude):
        amplitudes = np.array([1.0, relative_amplitude])
    else:
        amplitudes = np.array([0.0, 1.0])
    amplitudes /= np.linalg.norm(amplitudes) * np.sqrt(2)
    reference_beam_amplitude_fraction = 1/3
    amplitudes = [*(amplitudes * (1 - reference_beam_amplitude_fraction)), reference_beam_amplitude_fraction]
    offsets_x = np.array([-0.2585, 0.2574, 0.3874])
    offsets_y = np.array([0, 0, 0.155])
    slm.modulate(lambda x, y: (np.sqrt(x**2 + y**2) < 100) *
                              sum(a * np.exp(2j*np.pi * (x*ox + y*oy))
                                  for a, ox, oy in zip(amplitudes, offsets_x, offsets_y))
                 )


if __name__ == '__main__':
    available_cams = IDSCam.list()
    log.info(available_cams)
    cam_exposures = np.array([10e-3, 10e-3])
    cam_rois = [None, None]
    log.info(f'Loading aberration correction from {input_config_file_path}...')
    aberration_correction_settings = np.load(input_config_file_path.as_posix())
    pupil_aberration = aberration_correction_settings['pupil_aberration']
    log.info('Opening the spatial light modulator...')

    with PhaseSLM(display_target=0, deflection_frequency=(0, -1/4), two_pi_equivalent=0.3) as slm:
        slm.correction = correction_from_pupil(pupil_aberration, attenuation_limit=4)
        slm.post_modulation_callback = lambda _: time.sleep(100e-3)
        log.info('Opening the cameras...')
        create_spots(relative_amplitude=1.0)

        with IDSCam(serial=4103697390, exposure_time=cam_exposures[0], gain=0.0) as cam_transmitted, \
                IDSCam(serial=4103198121, exposure_time=cam_exposures[1], gain=0.0) as cam_reflected:
            cams = [cam_transmitted, cam_reflected]  # horizontal, vertical polarization
            initial_black_levels = [cam.black_level for cam in cams]
            for cam in cams:
                 log.info(f'cam {cam} black_level before = {cam.black_level}')
                 try:
                     cam.black_level = 200
                 except IDSCamError as ce:
                     log.warning(f'Could not set black level on camera {cam}: {ce}')
                 log.info(f'cam {cam} black_level after = {cam.black_level}')
            log.info(f'Transmitted camera exposure: {cam_transmitted.exposure_time * 1e3:0.3f}ms and region of interest: {cam_transmitted.roi}.')
            log.info(f'Reflected camera exposure: {cam_reflected.exposure_time * 1e3:0.3f}ms and region of interest: {cam_reflected.roi}.')

            #display
            fig171, axs = plt.subplots(1, 2, figsize=(12, 8), sharex='all', sharey='all')
            cam_axs_0 = axs[:]
            cam_ax_ims_0 = [cam_axs_0[_].imshow(np.zeros(cam.roi.shape), extent=grid2extent(cam.roi.grid),
                                                cmap='jet', clim=[0, 1])
                            for _, cam in enumerate([cam_transmitted, cam_reflected])]
            cam_axs_0[0].set_title('Reference transmitted')
            cam_axs_0[1].set_title('Reference reflected')

            def creates_display():
                fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharex='all', sharey='all')
                cam_axs_0 = axs[:]
                cam_ax_ims_0 = [cam_axs_0[_].imshow(np.zeros(cam.roi.shape), extent=grid2extent(cam.roi.grid),
                                                    cmap='jet', clim=[0, 1])
                                for _, cam in enumerate([cam_transmitted, cam_reflected])]
                cam_axs_0[0].set_title('Reference transmitted')
                cam_axs_0[1].set_title('Reference reflected')


            def acquire_interference_images(number_of_measurements: int = 1, relative_amplitude: complex = 1.0):
                cams = (cam_transmitted, cam_reflected)
                log.debug(f'Create two spots with relative amplitude {relative_amplitude}.')
                create_spots(relative_amplitude)
                log.debug('Acquiring interference images...')
                interference_images = []
                for cam in cams:
                    avg_img = 0.0
                    for _ in range(number_of_measurements):
                        avg_img = avg_img + cam.acquire()
                    interference_images.append(avg_img / number_of_measurements)
                log.debug('Mirroring the second image left-right.')
                interference_images[1] = interference_images[1][:, ::-1]
                return interference_images

            def measure_and_display(relative_amplitude: complex = 1.0):
                interference_images = acquire_interference_images(1, relative_amplitude)
                display(cam_ax_ims_0, interference_images)
                plt.show(block=False)
                plt.pause(0.1)
                return interference_images

            def get_top_quantile(_: np.ndarray) -> float:
                return np.quantile(_, 0.90)

            def exposure_time_calibration(diagonal_polarizer: bool = True):
                target_range = np.array([0.9, 0.99])
                if diagonal_polarizer:
                    target_range *= 0.8

                nb_attempts = 10
                horizontal_max = get_top_quantile(measure_and_display(0)[0])
                attempt = 0
                while attempt < nb_attempts and (horizontal_max > target_range[1] or horizontal_max < target_range[0]):
                    if horizontal_max > target_range[1]:
                        cam_transmitted.exposure_time /= 2.0
                    else:
                        cam_transmitted.exposure_time *= target_range[0] / horizontal_max
                    horizontal_max = get_top_quantile(measure_and_display(0)[0])
                    attempt += 1

                vertical_max = get_top_quantile(measure_and_display(np.inf)[1])
                attempt = 0
                while attempt < nb_attempts and (vertical_max > target_range[1] or vertical_max < target_range[0]):
                    if vertical_max > target_range[1]:
                        cam_reflected.exposure_time /= 2.0
                    else:
                        cam_reflected.exposure_time *= target_range[0] / vertical_max
                    vertical_max = get_top_quantile(measure_and_display(np.inf)[1])
                    attempt += 1
                log.info(f'Set exposure times of {cam_transmitted} to {cam_transmitted.exposure_time*1e3:0.3f}ms and that of {cam_reflected} to {cam_reflected.exposure_time*1e3:0.3f}ms.')

            def measure_phase_and_amplitude(previous_correction = 1.0):
                """
                Measures the amplitude fraction vertical / horizontal of the optical set-up. This can be used to divide out the effect.

                :return: The measured value (divide by this to correct)
                """
                number_samples = 10
                phase_grid = Grid([number_samples*2], extent=3*np.pi)
                amplitudes = np.linspace(0.1, 1.4, num=number_samples*4)
                phasors = np.exp(1j * phase_grid[0])
                total_intensity_phase = np.zeros(len(phasors))
                total_intensity_amplitude = np.zeros(len(amplitudes))
                number_measurements = 1
                for _ in tqdm(range(len(phasors)), desc='Relative phase calibration progress'):
                    measurement = acquire_interference_images(number_measurements, previous_correction * phasors[_])
                    total_intensity_phase[_] = (np.mean(measurement[0]) + np.mean(measurement[1]))/2
                measured_phase = np.angle(np.vdot(phasors, total_intensity_phase))
                measured_phasor = np.exp(1j * measured_phase)
                log.info(f'Phase difference between horizontal and vertical: {measured_phase:0.3f} rad = {measured_phase * 180 / np.pi:0.1f} degrees.')
                for _ in tqdm(range(len(amplitudes)), desc='Relative amplitude calibration progress'):
                    measurement = acquire_interference_images(number_measurements, previous_correction / measured_phasor * -amplitudes[_])   # anti-diagonal
                    total_intensity_amplitude[_] = (np.mean(measurement[0]) + np.mean(measurement[1])) / 2

                # Correct for the normalization during the measurement of create_spots
                total_intensity_amplitude = (1 + amplitudes ** 2) * total_intensity_amplitude
                coefficients = np.polyfit(amplitudes, total_intensity_amplitude, 2)
                fitted_polynomial = np.poly1d(coefficients)
                correction_amplitude = -coefficients[1] / (2 * coefficients[0])
                if correction_amplitude < np.amin(amplitudes) or correction_amplitude > np.amax(amplitudes):
                    correction_amplitude = np.clip(correction_amplitude, np.amin(amplitudes), np.amax(amplitudes))
                    log.warning(f'Clipped correction amplitude to {correction_amplitude:0.3f}!')
                measured_amplitude = 1 / np.abs(correction_amplitude)  # so that it works with diagonal and anti-diagonal.
                log.info(f'1/measured_amplitude = {1/measured_amplitude:0.3f}')

                # Display

                # fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
                # ax1.plot(phase_grid[0], total_intensity_phase, 'r')
                # ax1.plot(phase_grid[0], np.mean(total_intensity_phase) + (np.amax(total_intensity_phase) - np.mean(total_intensity_phase)) * np.cos(phase_grid[0] + measured_phase))
                # ax1.set_title('Relative phase [rad]')
                # ax2.plot(amplitudes, total_intensity_amplitude, '.', amplitudes, fitted_polynomial(amplitudes), 'g')
                # #ax2.set(ylim=[0, 1.1 * np.amax(total_intensity_amplitude)])
                # ax2.set_title('Relative amplitude')
                # plt.show()

                return -measured_phasor * measured_amplitude / previous_correction

            def sample_adjustment(measured_amp_phase: complex = 1.0):

                while plt.fignum_exists(fig171.number):
                    interference_images = acquire_interference_images(1, 1.0 / measured_amp_phase)
                    display(cam_ax_ims_0, interference_images)
                    plt.show(block=False)
                    plt.pause(0.01)


            def polarizations_measurement(number_of_measurements: int = 1, measured_amp_phase: complex = 1.0):
                interference_image_pairs = []
                polarization_amplitudes = (0.0, np.inf, 1.0, -1.0, 1j, -1j)
                polarization_names = ['horizontal', 'vertical', 'diagonal', 'anti-diagonal', 'right circular', 'left circular']

                log.info(f'measured amplitude = {np.abs(measured_amp_phase):0.3f} and phase {np.angle(measured_amp_phase):0.3f}rad or {np.angle(measured_amp_phase) * 180 / np.pi:0.1f} degrees.')
                #initial_interference_images = acquire_interference_images(number_of_measurements, 1 / measured_amp_phase)
                for pol_amplitude, pol_name in zip(polarization_amplitudes, polarization_names):
                    log.info(f'Recording the {pol_name} polarization.')
                    interference_image_pair = acquire_interference_images(number_of_measurements, pol_amplitude / measured_amp_phase)
                    interference_image_pairs.append(interference_image_pair)
                initial_interference_images = interference_image_pairs[2]

                # display
                fig, axs = plt.subplots(2, 6)
                for cam_ind in range(2):
                    for _ in range(6):
                        axs[cam_ind,_].imshow(interference_image_pairs[_][cam_ind])
                        axs[cam_ind, _].axis('off')
                        axs[0, _].set_title(f'{polarization_names[_]}')
                axs[0,0].set_ylabel('Transmitted')
                axs[1,0].set_ylabel('Reflected')
                plt.show()
                save_the_results = input('Enter a sample name to save')
                output_file_path = None
                if save_the_results != '':
                    sample_name = save_the_results
                    output_file_path = Path(f'../../confocal_interference_contrast/polarization_memory_effect/results/polarization_memory_effect_{sample_name}_'
                                        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")).resolve()
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(output_file_path,
                             polarization_amplitudes=polarization_amplitudes,
                             exposure_times=[_.exposure_time for _ in cams],
                             initial_exposure_times=cam_exposures,
                             initial_interference_images=initial_interference_images,
                             interference_image_pairs=interference_image_pairs,
                             calibration_amplitude=measured_amp_phase)
                    log.info(f'Saved to {output_file_path}.')
                return output_file_path

            def scan(nanostage, positions):
                positions = np.asarray(positions)
                cams = (cam_transmitted, cam_reflected)
                nanostage.center_all()
                x0 = nanostage.stage_range[0, 1] / 2 * 1e-9
                scan_images = []
                for _ in tqdm(range(number_of_points)):
                    nanostage.move(axis=1, value=int((x0 + positions[_, 0]) * 1e9))
                    time.sleep(0.5)
                    interference_images = []
                    for cam in cams:
                        interference_images.append(cam.acquire())
                    interference_images[1] = interference_images[1][:, ::-1]
                    scan_images.append(interference_images)
                return scan_images

            control = "1"
            measured_amp_phase = 1.0
            interference_file_path = None
            initial_reference_file_path = None
            nano = NanostageLT3(port='COM3')
            nano.center_all()


            measured_phasor = np.exp(1j * 3.067)

            measured_amp_phase = -measured_phasor * (1/0.758)

            while control != "exit":
                control = input("Options: 0 -> Exit; 1 -> initial calibration; 2 -> sample position adjustment ; 3 -> Polarization measurement; 4-> exposure time adjust; 5-> Process polarization; 6-> scan")
                if control == "1":
                    log.info('Calibrating phase...')
                    exposure_time_calibration(diagonal_polarizer=True)
                    measured_amp_phase = measure_phase_and_amplitude()
                elif control == '2':
                    log.info('Refocusing objective...')
                    fig_status = plt.fignum_exists(fig171.number)
                    if fig_status == False:
                        fig171, axs = plt.subplots(1, 2, figsize=(12, 8), sharex='all', sharey='all')
                        cam_axs_0 = axs[:]
                        cam_ax_ims_0 = [cam_axs_0[_].imshow(np.zeros(cam.roi.shape), extent=grid2extent(cam.roi.grid),
                                                            cmap='jet', clim=[0, 1])
                                        for _, cam in enumerate([cam_transmitted, cam_reflected])]
                        cam_axs_0[0].set_title('Reference transmitted')
                        cam_axs_0[1].set_title('Reference reflected')
                    sample_adjustment(measured_amp_phase)
                    #measured_amp_phase = measure_phase_and_amplitude(1 / measured_amp_phase)
                elif control == '3':
                    log.info('Adjusting exposure time and measuring for all polarizations...')
                    exposure_time_calibration(diagonal_polarizer=False)
                    mean_exposure = (cam_transmitted.exposure_time + cam_reflected.exposure_time)/2
                    cam_reflected.exposure_time = mean_exposure
                    cam_transmitted.exposure_time = mean_exposure
                    log.info(f'Set exposure times of {cam_transmitted} to {cam_transmitted.exposure_time*1e3:0.3f}ms and that of {cam_reflected} to {cam_reflected.exposure_time*1e3:0.3f}ms.')
                    #log.info(f'Exposure times on cameras before the calibration: {[_.exposure_time for _ in cams]}s.')
                    interference_file_path = polarizations_measurement(1, measured_amp_phase)
                elif control == '4':
                    control_2 = input('1 -> exposure time calibration; 2-> set new exposure time')
                    if control_2 == '1':
                        exposure_time_calibration(diagonal_polarizer=False)
                        mean_exposure = (cam_transmitted.exposure_time + cam_reflected.exposure_time)/2
                        cam_reflected.exposure_time=mean_exposure
                        cam_transmitted.exposure_time=mean_exposure
                        log.info(f'Set exposure times of {cam_transmitted} to {cam_transmitted.exposure_time*1e3:0.3f}ms and that of {cam_reflected} to {cam_reflected.exposure_time*1e3:0.3f}ms.')
                    elif control_2 == '2':
                        cam_reflected.exposure_time = float(input('set cam reflected exposure time (ms):')) * 1e-3
                        cam_transmitted.exposure_time = float(input('set cam transmitted exposure time (ms):')) * 1e-3
                        log.info(f'Set exposure times of {cam_transmitted} to {cam_transmitted.exposure_time*1e3:0.3f}ms and that of {cam_reflected} to {cam_reflected.exposure_time*1e3:0.3f}ms.')

                elif control == '5':
                    if interference_file_path is None:
                        log.info('No interference recorded, please type 3 first.')
                    else:
                        ref_status = input('Enter -> Keep last reference; 1 -> None; 2 -> New reference')
                        if ref_status == '1':
                            initial_reference_file_path = None
                        elif ref_status == '2':
                            initial_reference_file_path = input('What reference do you want to use?')
                        if initial_reference_file_path is None:
                            log.info('Using itself as reference!')
                        else:
                            initial_reference_file_path = Path(initial_reference_file_path).with_suffix('.npz')
                            if not initial_reference_file_path.is_file():
                                initial_reference_file_path = interference_file_path.parent / initial_reference_file_path.name
                            log.info(f'Using {initial_reference_file_path} as reference...')
                        display_polarization_memory_effect(interference_file_path, initial_reference_file_path)
                        plt.show()

                elif control == '6':
                    polarization_names = ['horizontal', 'vertical', 'diagonal', 'anti_diagonal', 'right_circular', 'left_circular']
                    polarization_amplitudes = (0, np.infty, 1.0, -1.0, 1j, -1j)
                    sample_name = input('sample name:')
                    for _ in range(6): # todo 6
                        create_spots(polarization_amplitudes[_] / measured_amp_phase) #diagonal polarization
                        time.sleep(0.5)
                        scans = []
                        number_of_points = 100
                        dx = 25e-9
                        positions = [(_*dx, 0.0, 0.0) for _ in np.arange(number_of_points) - number_of_points // 2]
                        scans.append(scan(nano, positions))
                        output_file_path = Path(f'../../confocal_interference_contrast/polarization_memory_effect/results/scan_'+ sample_name
                                                +'_'+polarization_names[_]+'_'+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")).resolve()
                        output_file_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(output_file_path, scan_images=scans, positions=positions)

                elif control == '8':
                    # precision test scan, diagonal pol only
                    create_spots(1.0 / measured_amp_phase)
                    sample_name = input('sample name:')
                    for _ in range(20): #todo 20
                        time.sleep(0.5)
                        scans = []
                        number_of_points = 100
                        dx = 25e-9
                        positions = [(_*dx, 0.0, 0.0) for _ in np.arange(number_of_points) - number_of_points // 2]
                        scans.append(scan(nano, positions))
                        output_file_path = Path(f'../../confocal_interference_contrast/polarization_memory_effect/results/scan_precision_test_'+ sample_name
                                                +'_'+str(_)+'_'+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")).resolve()
                        output_file_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(output_file_path, scan_images=scans, positions=positions)

                elif control == '9':
                    # stability test, diagonal pol only
                    #keep sample at same position and capture a image every second along 10 minutes
                    create_spots(1.0 / measured_amp_phase)
                    time.sleep(0.5)
                    sample_name = input('sample name:')
                    nano.center_all()
                    stability_image_pairs = []
                    for _ in tqdm(range(600)): #todo 600
                        interference_images = []
                        for cam in cams:
                            interference_images.append(cam.acquire())
                        interference_images[1] = interference_images[1][:, ::-1]
                        stability_image_pairs.append(interference_images)

                    output_file_path = Path(f'../../confocal_interference_contrast/polarization_memory_effect/results/stability_test_'+ sample_name
                                            +'_'+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")).resolve()
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(output_file_path, stability_image_pairs=stability_image_pairs)

                elif control == '7':
                    #  scan the reference
                    create_spots(1.0 / measured_amp_phase)
                    sample_name = input('reference name:')
                    scans = []
                    number_of_points = 10
                    dx = 1e-6
                    positions = [(_*dx, 0.0, 0.0) for _ in np.arange(number_of_points) - number_of_points // 2]
                    for _ in range(1):
                        scans.append(scan(nano, positions))
                    output_file_path = Path(f'../../confocal_interference_contrast/polarization_memory_effect/results/reference_scan_'+ sample_name
                                            +'_'+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")).resolve()
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(output_file_path, scan_images=scans, positions=positions)

                elif control == 'exit':
                    #control = "exit"
                    log.info('Done.')
