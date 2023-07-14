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
from datetime import datetime

input_config_file_path = Path('../../../results/aberration_correction_2022-05-12_15-54-37.439768.npz').resolve()
cam_list = IDSCam.list()
log.info(cam_list)
with IDSCam(serial=4103697390) as cam:
    img = cam.acquire()

exit()


def display(cam_ax_ims, interference_images):
    for cam_ax_im, interference_image in zip(cam_ax_ims, interference_images):
        saturated = interference_image >= 1.0
        marked_image = np.repeat(interference_image[..., np.newaxis], repeats=3, axis=-1)
        marked_image[:, :, 1:] *= 1 - saturated[..., np.newaxis]
        cam_ax_im.set_data(marked_image)


if __name__ == '__main__':
    cam_list = IDSCam.list()
    log.info(cam_list)
    ml_nir_cam = cam_list['IDSCam1_4103697390_UI324xML-NIR']['index']
    ml_m_cam = cam_list['IDSCam2_4103198121_UI324xML-M']['index']

    cam_exposures = np.array([10e-3, 10e-3])
    cam_rois = [None, None]

    log.info(f'Loading aberration correction from {input_config_file_path}...')
    aberration_correction_settings = np.load(input_config_file_path.as_posix())
    pupil_aberration = aberration_correction_settings['pupil_aberration']
    log.info('Opening the spatial light modulator...')

    with PhaseSLM(display_target=0, deflection_frequency=(0, -1/4), two_pi_equivalent=0.3) as slm:
        slm.correction = correction_from_pupil(pupil_aberration, attenuation_limit=4)
        slm.post_modulation_callback = lambda _: time.sleep(100e-3)  # Set a function to be called every time we modulate the slm

        log.info('Opening the cameras...')
        with IDSCam(index=ml_nir_cam, exposure_time=cam_exposures[0], gain=0.0) as cam_transmitted, \
                IDSCam(index=ml_m_cam, exposure_time=cam_exposures[0], gain=0.0) as cam_reflected:
            cams = [cam_transmitted, cam_reflected]  # horizontal, vertical polarization
            log.info(f'Transmitted camera exposure: {cam_transmitted.exposure_time * 1e3:0.3f}ms and region of interest: {cam_transmitted.roi}.')
            log.info(f'Reflected camera exposure: {cam_reflected.exposure_time * 1e3:0.3f}ms and region of interest: {cam_reflected.roi}.')

            def create_spots(relative_amplitude: complex = 1.0, separation: float = 2*0.2574):
                """Function that creates two focal spots with a relative complex amplitude of the second spot
                :param relative_amplitude: The amplitude of the second spot with respect to the first.
                :param separation: The spot separation.
                """
                if not np.isinf(relative_amplitude):
                    amplitudes = np.array([1.0, relative_amplitude])
                else:
                    amplitudes = np.array([0.0, 1.0])
                    # normalize
                amplitudes /= np.linalg.norm(amplitudes) * np.sqrt(2)
                #  log.info(f'Sending two spots to first order with amplitudes {amplitudes}.')
                # Add third beam
                # third_spot_offset = [separation / 2 + 0.13, 0.155]  #0.15, 0.19
                reference_beam_amplitude_fraction = 1/3
                # offsets_x = [*offsets_x, third_spot_offset[0]]
                # offsets_y = [*offsets_y, third_spot_offset[1]]
                amplitudes = [*(amplitudes * (1 - reference_beam_amplitude_fraction)), reference_beam_amplitude_fraction]
                # log.info(f'Sending third spot of amplitude {third_spot_amplitude} to {third_spot_offset}.')

                offsets_x = np.array([-0.2585, 0.2574, 0.3874])
                offsets_y = np.array([0, 0, 0.155])
                slm.modulate(lambda x, y: (np.sqrt(x**2 + y**2) < 100) *
                                          sum(a * np.exp(2j*np.pi * (x*ox + y*oy))
                                              for a, ox, oy in zip(amplitudes, offsets_x, offsets_y))
                             )

            optimal_amplitude = 1
            optimal_phase = -80 * np.pi / 180 #-80
            calibration_amplitude = optimal_amplitude * np.exp(1j * optimal_phase)

            def acquire_interference_images(relative_amplitude: complex = 1.0) -> List[np.ndarray]:
                cams = (cam_transmitted, cam_reflected)
                log.debug(f'Create two spots with relative amplitude {relative_amplitude}.')
                create_spots(relative_amplitude)
                log.debug('Acquiring interference images...')
                interference_images = [cam.acquire().astype(np.float32) for cam in cams]
                log.debug('Mirroring the second image left-right.')
                interference_images[1] = interference_images[1][:, ::-1]

                return interference_images

            def measure_and_display(amplitude: complex = 1.0):
                interference_images = acquire_interference_images(amplitude*calibration_amplitude)
                display(cam_ax_ims_0, interference_images)
                plt.show(block=False)
                plt.pause(0.1)
                return interference_images

            # todo remove this part
            log.info('Preparing the display...')
            fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharex='all', sharey='all')
            cam_axs_0 = axs[:]
            cam_ax_ims_0 = [cam_axs_0[_].imshow(np.zeros(cam.roi.shape), extent=grid2extent(cam.roi.grid),
                                            cmap='gray', clim=[0, 1])
                          for _, cam in enumerate([cam_transmitted, cam_reflected])]
            cam_axs_0[0].set_title('Reference transmitted')
            cam_axs_0[1].set_title('Reference reflected')

            calibration_position = 'n'
            while calibration_position in ('n', ''):
                log.info('Adjusting sample position...')
                measure_and_display(1.0)
                calibration_position = input('Finished? (y/n)')

            max_intensities = np.ones([2, 2])
            while np.any(max_intensities >= 0.99) or np.any(np.diag(max_intensities) < 0.8):
                interference_images_horizontal_pol = measure_and_display(1.0)
                interference_images_vertical_pol = measure_and_display(1.0)
                max_intensities = np.array([[np.amax(_) for _ in interference_images_horizontal_pol],
                                            [np.amax(_) for _ in interference_images_vertical_pol]
                                           ])
                for _, cam in enumerate(cams):
                    if max_intensities[_, _] >= 0.99:
                        cam.exposure_time /= 2.0
                    else:
                        cam.exposure_time *= 0.9 / max_intensities[_, _]
                log.info(f'Exposure times on cameras: {[_.exposure_time for _ in cams]}s resulted in maximum intensities of {np.diag(max_intensities)}.')
            if max_intensities[0, 0] < 10 * max_intensities[0, 1]:
                log.error(f'Horizontal polarized light did not go predominantly to the horizontal (transmitted) camera ({max_intensities}).')
            if max_intensities[1, 1] < 10 * max_intensities[1, 0]:
                log.error(f'Vertical polarized light did not go predominantly to the vertical (reflected) camera ({max_intensities}).')

            save_initial_measurement = 'n'
            while save_initial_measurement in ('n', ''):
                log.info('Recording diagonal reference before sample...')
                initial_interference_images = measure_and_display(1.0)
                save_initial_measurement = input('Save initial reference before sample? (y/n)')

            log.info('Recording with sample...')
            save_the_results = 'n'
            while save_the_results in ('n', ''):
                interference_image_pairs = []
                polarization_amplitudes = (0.0, np.inf, 1.0, -1.0, 1j, -1j)
                for pol_amplitude in polarization_amplitudes:
                    log.info(f'Recording the polarization:({pol_amplitude}).')
                    interference_image_pair = measure_and_display(pol_amplitude)
                    display(cam_ax_ims_0, interference_image_pair)
                    interference_image_pairs.append(interference_image_pair)
                save_the_results = input('Enter a sample name to save, or just enter to continue measuring')

            sample_name = save_the_results
            output_file_path = Path(f'../../confocal_interference_contrast/polarization_memory_effect/results/polarization_memory_effect_{sample_name}_'
                                    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f%z.npz")).resolve()
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            log.info(f'Saving to {output_file_path}...')
            np.savez(output_file_path,
                     polarization_amplitudes=polarization_amplitudes,
                     exposure_times=[_.exposure_time for _ in cams],
                     initial_exposure_times=cam_exposures,
                     initial_interference_images=initial_interference_images,
                     interference_image_pairs=interference_image_pairs,
                     calibration_amplitude=calibration_amplitude)
            log.info(f'Saved to {output_file_path}.')

            log.info('Done.')
            # plt.show()
