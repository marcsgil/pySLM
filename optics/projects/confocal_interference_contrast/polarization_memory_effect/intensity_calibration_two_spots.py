# code to measure the intensity of the spots used at the optical memory effect project
# test code - Braian
from typing import List, Optional, Tuple
import numpy as np
import time
import matplotlib.pylab as plt
from pathlib import Path

from projects.confocal_interference_contrast.polarization_memory_effect import log
from optics.instruments.cam.ids_cam import IDSCam
from optics.instruments.slm import PhaseSLM
from optics.utils.display import complex2rgb, grid2extent
from optics.calc import correction_from_pupil
from projects.confocal_interference_contrast.complex_phase_image import Interferogram
from datetime import datetime

input_config_file_path = Path('../../../results/aberration_correction_2022-05-12_15-54-37.439768.npz').resolve()

if __name__ == '__main__':
    cam_list = IDSCam.list()
    log.info(cam_list)
    # ml_nir_cam = cam_list['IDSCam2_4103697390_UI324xML-NIR']['index'] - 1
    # ml_m_cam = cam_list['IDSCam1_4103198121_UI324xML-M']['index'] - 1

    cam_exposures = np.array([20e-3, 15e-3])
    cam_rois = [None, None]

    log.info(f'Loading aberration correction from {input_config_file_path}...')
    aberration_correction_settings = np.load(input_config_file_path.as_posix())
    pupil_aberration = aberration_correction_settings['pupil_aberration']

    log.info('Opening the spatial light modulator...')
    with PhaseSLM(display_target=0, deflection_frequency=(0, -1/4), two_pi_equivalent=0.3) as slm:
        slm.correction = correction_from_pupil(pupil_aberration, attenuation_limit=4)
        slm.post_modulation_callback = lambda _: time.sleep(100e-3)  # Set a function to be called every time we modulate the slm

        log.info('Opening the cameras...')
        with IDSCam(serial=4103697390, exposure_time=cam_exposures[0], gain=0.0) as cam_transmitted, \
                IDSCam(serial=4103198121, exposure_time=cam_exposures[0], gain=0.0) as cam_reflected:
            # cam_transmitted.roi = cam_rois[0]
            # cam_reflected.roi = cam_rois[1]
            log.info(f'Transmitted camera exposure: {cam_transmitted.exposure_time * 1e3:0.3f}ms and region of interest: {cam_transmitted.roi}.')
            log.info(f'Reflected camera exposure: {cam_reflected.exposure_time * 1e3:0.3f}ms and region of interest: {cam_reflected.roi}.')

            # todo remove this part
            log.info('Preparing the display...')
            fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex='all', sharey='all')
            cam_axs = axs[:, 0]
            cam_ax_ims = [cam_axs[_].imshow(np.zeros(cam.roi.shape), extent=grid2extent(cam.roi.grid),
                                            cmap='gray', clim=[0, 1])
                          for _, cam in enumerate([cam_transmitted, cam_reflected])]
            cam_axs[0].set_title('Cam transmitted')
            cam_axs[1].set_title('Cam reflected')

            fld_axs = axs[:, 1]
            fld_ax_ims = [fld_axs[_].imshow(np.zeros(cam.roi.shape), extent=grid2extent(cam.roi.grid),
                                            cmap='gray', clim=[0, 1])
                          for _, cam in enumerate([cam_transmitted, cam_reflected])]
            fld_axs[0].set_title('Cam transmitted')
            fld_axs[1].set_title('Cam reflected')

            registrations = [None, None]

            def create_spots(relative_amplitude: complex = 1.0, separation: float = 2*0.2574):
                """Funtion that creates two focal spots with a relative complex amplitude of the second spot
                :param relative_amplitude: The amplitude of the second spot with respect to the first.
                :param separation: The spot separation.
                """
                offsets = np.array([-1, 1]) * separation / 2
                if not np.isinf(relative_amplitude):
                    amplitudes = np.array([1.0, relative_amplitude])
                else:
                    amplitudes = np.array([0.0, 1.0])
                amplitudes /= np.sum(np.abs(amplitudes))
                log.info(f'Sending two spots to first order with amplitudes {amplitudes}.')
                slm.modulate(lambda x, y: (np.sqrt(x**2 + y**2) < 100) *
                             sum(a*np.exp(2j*np.pi*x*o) for a, o in zip(amplitudes, offsets))
                             )

            def measure_interferograms(relative_amplitude: complex = 1.0,
                                       initial_interferograms: Optional[List[Interferogram]] = None,
                                       ) -> List[np.ndarray]:
                """
                Function to produce two spots with a given relative amplitude using the SLM and record the complex
                amplitude on the two cameras.

                :param relative_amplitude: The amplitude of the second spot with respect to the first.
                :param initial_interferograms: An optional list of interferogram objects to reference against.
                :return: A list of interferograms, one for each camera.
                """
                log.debug(f'Create two spots with relative amplitude {relative_amplitude}.')
                create_spots(relative_amplitude)
                log.debug('Acquiring interference images...')
                interference_images = [cam.acquire() for cam in (cam_transmitted, cam_reflected)]
                log.debug('Mirroring the second image left-right.')
                interference_images[1] = interference_images[1][:, ::-1]

                if initial_interferograms is None:
                    initial_registrations = [None, None]
                else:
                    initial_registrations = [_.registration for _ in initial_interferograms]

                interferograms = [Interferogram(img, reg, maximum_fringe_period=10)
                                  for img, reg in zip(interference_images, initial_registrations)]
                return interference_images, interferograms

            def display(interference_fields, interference_images):
                for cam_ax_im, interference_image in zip(cam_ax_ims, interference_images):
                    saturated = interference_image >= 1.0
                    marked_image = np.repeat(interference_image[..., np.newaxis], repeats=3, axis=-1)
                    marked_image[:, :, 1:] *= 1 - saturated[..., np.newaxis]
                    cam_ax_im.set_data(marked_image)
                for fld_ax_im, interference_field in zip(fld_ax_ims, interference_fields):
                    fld_ax_im.set_data(complex2rgb(interference_field, normalization=1.0))

            def measure_complex_amplitude():
                nb_phase_steps = 1000
                nb_amplitude_steps = 100
                phases = np.arange(nb_phase_steps) / nb_phase_steps * 2 * np.pi - np.pi
                intensities_for_phase = []
                for phase in phases:
                    log.info(f'Measuring for phase {phase}...')
                    create_spots(np.exp(1j * phase))
                    intensity = np.mean(cam_transmitted.acquire())
                    intensities_for_phase.append(intensity)
                diagonal_phase = phases[np.argmin(intensities_for_phase)] % (2 * np.pi) - np.pi
                log.info(f'Diagonal phase = {diagonal_phase:0.3f}rad or {diagonal_phase*180/np.pi:0.1f} degree.')

                amplitudes = np.arange(nb_amplitude_steps) / nb_amplitude_steps + 0.5
                intensities_for_amplitude = []
                for amplitude in amplitudes:
                    log.info(f'Measuring for amplitude {amplitude:0.3f}...')
                    create_spots(np.exp(1j * diagonal_phase) * amplitude)
                    intensity = np.mean(cam_transmitted.acquire())
                    intensities_for_amplitude.append(intensity)
                optimal_amplitude = amplitudes[np.argmax(intensities_for_amplitude)]
                log.info(f'Optimal amplitude = {optimal_amplitude}.')

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(phases, intensities_for_phase)
                axs[0].set(xlabel='phase [rad]', ylabel='I')
                axs[1].plot(amplitudes, intensities_for_amplitude)
                axs[1].set(xlabel='amplitude', ylabel='I')
                plt.show(block=False)

                return optimal_amplitude * np.exp(1j * diagonal_phase)

            # optimal_amplitude = 0.97
            # optimal_phase = -1.90
            # calibration_amplitude = optimal_amplitude * np.exp(1j * optimal_phase)
            calibration_amplitude = measure_complex_amplitude()

            log.info('Recording diagonal reference before sample...')
            initial_interferograms, initial_interference_images = measure_interferograms(calibration_amplitude)
            for _ in initial_interferograms:
                log.info(f'Locked in fringe frequency {_.registration.shift}')

            def measure_and_display(amplitude):
                interference_images, _ = measure_interferograms(calibration_amplitude * amplitude,
                                                                             initial_interferograms)
                # The Wiener filter is used to divide the new measurements by the initial interferograms without
                # causing a division by zero
                NSR = 0.01  # noise-to-signal ratio
                wiener_filter = lambda _: _.conj() / (np.abs(_) ** 2 + NSR ** 2)
                interference_fields = [wiener_filter(init / np.amax(np.abs(init))) * new.__array__()
                                       for new, init in zip(interferograms, initial_interferograms)]

                interference_fields = np.array(interference_fields, dtype=np.complex64)

                # Display
                display(interference_fields, interference_images)

                plt.show(block=False)
                plt.pause(0.1)

                return interference_images

            log.info('Recording with sample...')
            interferograms_for_all_experiments = []
            interference_images_for_all_experiments = []
            amplitudes = (0.0, np.inf, 1.0, -1.0, 1j, -1j)
            for amplitude in amplitudes:
                interferograms, interference_images = measure_and_display(amplitude)
                interferograms_for_all_experiments.append(interferograms)
                interference_images_for_all_experiments.append(interference_images)

            sample_name = 'just_super_glue'
            output_file_path = Path(f'../../results/polarization_memory_effect_{sample_name}_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f%z.npz")).resolve()
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            log.info(f'Saving to {output_file_path}...')
            np.savez(output_file_path,
                     amplitudes=amplitudes,
                     exposure_times=cam_exposures,
                     interference_images_for_all_experiments=interference_images_for_all_experiments,
                     calibration_amplitude=calibration_amplitude)
            log.info(f'Saved to {output_file_path}.')

            log.info('Done.')
            plt.show()
