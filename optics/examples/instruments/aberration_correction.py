#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Demonstrates the use of the spatial light modulator to measure and correct optical aberrations.
# ? Why the correction phase does not update
# ? how to turn on and off the correction phase on SLM
# ? why this not work?
# def display(fraction=None, pupil=None):
#     slm.modulate(1.0)

import numpy as np
import time
import matplotlib.pylab as plt
from pathlib import Path
from scipy.io import savemat
from datetime import datetime

from examples.instruments import log
from optics.utils import Roi
from optics.instruments.cam.simulated_cam import SimulatedCam
from optics.instruments.cam.ids_cam import IDSCam
from optics.instruments.slm import PhaseSLM
from optics.instruments.slm.meadowlark_pci_slm import MeadowlarkPCISLM
from optics.instruments.slm import measure_aberration
from optics.utils.display import complex2rgb, grid2extent
from optics.calc import correction_from_pupil

output_file_path = Path(
    '../../results/aberration_correction_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f%z")).resolve()
output_file_path.parent.mkdir(parents=True, exist_ok=True)

slm_display = 0
slm_wait_time = 100e-3
measurement_grid_shape = (7, 7, 3)  # Increase to (25, 25, 3) for higher accuracy
avg_shape = np.array([1, 1])
attenuation_limit = 4
deflection_frequency = (0, 1 / 6)
two_pi_equivalent = 1.0

if __name__ == "__main__":
    # with WebCam(color=False, normalize=True) as cam:
    # with SimulatedCam(normalize=True, get_frame_callback=lambda _: np.random.randn(*_.roi.shape)) as cam:
    with IDSCam(index=1, normalize=True, exposure_time=50e-3, gain=1, black_level=150) as cam:
        # Define how to measure the intensity at a specific point on the camera
        def measurement_function() -> float:
            img = cam.acquire()  # skip one
            img = cam.acquire()
            if np.amax(img) >= 1.0:
                log.warning(f'{np.sum(img.ravel() >= 1.0 - 1e-3)} pixels saturated!')
            log.debug(f'Image values between {np.amin(img)} -> {np.amax(img)}')
            img = img[img.shape[0] // 2 + np.arange(avg_shape[0])[:, np.newaxis] - avg_shape[0] // 2,
                      img.shape[1] // 2 + np.arange(avg_shape[1]) - avg_shape[
                          1] // 2]  # pick the center of the region of interest
            return float(np.mean(img.ravel()))

        # Define the figure windows
        slm_fig, slm_axs = plt.subplots(1, 2, figsize=(12, 8), sharex='all', sharey='all')
        aberration_ax = slm_axs[0]
        aberration_ax.set_title('aberration')
        correction_ax = slm_axs[1]
        correction_ax.set_title('correction')

        cam_fig, cam_axs = plt.subplots(1, 2, figsize=(12, 8), sharex='all', sharey='all')
        cam_before_ax = cam_axs[0]
        cam_before_ax.set_title('Camera Before')
        cam_after_ax = cam_axs[1]
        cam_after_ax.set_title('Camera After')

        # with PhaseSLM(display_target=slm_ax, deflection_frequency=deflection_frequency) as slm:   #  Simulated
        # with PhaseSLM(display_target=slm_display, deflection_frequency=deflection_frequency, two_pi_equivalent=two_pi_equivalent) as slm:
        with MeadowlarkPCISLM(deflection_frequency=deflection_frequency, wavelength=532e-9) as slm:
            slm.roi = Roi(center=slm.shape // 2, shape=[512, 512])  # todo: Uncomment this to use a region of interest on the SLM. It doesn't crash, but that doesn't mean it already works!

            slm.post_modulation_callback = lambda _: time.sleep(slm_wait_time)  # Forces SLM to wait before returning from slm.modulate(...)

            last_update_time = [-np.infty]  # Use a list so that it is not a number but an object

            # Function to display the progress
            def display_progress(fraction=None, pupil=None):
                current_time = time.perf_counter()
                if current_time >= last_update_time[0] + 5.0:  # Update only every 5 seconds
                    if pupil is not None:
                        slm.correction = correction_from_pupil(pupil, attenuation_limit=1)
                    slm.modulate(1.0)
                    if pupil is not None:
                        slm.correction = 1.0
                    display(fraction, pupil)
                    log.info(f'{100 * fraction:0.1f}%: {measurement_function():0.3f}.')


            def display(fraction=None, pupil=None):
                # slm.modulate(1.0)
                img = cam.acquire()

                # cam_after_ax.cla()
                cam_after_ax.imshow(img, extent=grid2extent(*cam.roi.grid), cmap=plt.get_cmap('gray'))
                current_value = img[img.shape[0] // 2 + np.arange(avg_shape[0]) - avg_shape[0] // 2,
                                    img.shape[1] // 2 + np.arange(avg_shape[1]) - avg_shape[1] // 2]  # pick the center of the region of interest
                current_value = float(np.mean(current_value.ravel()))

                if fraction is not None:
                    log.info(f'Completed {fraction * 100:0.1f}%. Intensity: {current_value}.')

                if pupil is not None:
                    # aberration_ax.cla()
                    # correction_ax.cla()
                    aberration_ax.imshow(complex2rgb(pupil, normalization=1.0))
                    correction = correction_from_pupil(pupil)
                    correction_ax.imshow(complex2rgb(correction, normalization=1.0))

                plt.show(block=False)
                plt.pause(0.1)

                return True


            log.info('Focusing laser to first order spot.')
            slm.correction = 1.0
            slm.modulate(1.0)  # Focus to first order (without correction).
            log.info('Centering area of interest around brightest spot (also reduced integration time if needed)...')
            cam = cam.center_roi_around_peak_intensity(shape=(1500, 1500))
            log.info(
                f"The camera's region of interest is set to {cam.roi} with integration time {cam.exposure_time * 1000:0.3f} ms.")

            log.info('Maximizing intensity in first order spot.')

            # Display without correction
            slm.modulate(1)
            img_before = cam.acquire()
            img_before = cam.acquire()
            img_before = cam.acquire()
            cam_before_ax.imshow(img_before, extent=grid2extent(cam.roi.grid), cmap=plt.get_cmap('gray'))
            plt.draw()
            plt.pause(0.01)

            # From here the for loop for phase error correction search starts
            pupil_aberration = measure_aberration(slm, measurement_function,
                                                  measurement_grid_shape=measurement_grid_shape,
                                                  progress_callback=display_progress)

            log.info('Calculating correction pattern...')
            slm.correction = slm.correction * correction_from_pupil(pupil_aberration, attenuation_limit=attenuation_limit)

            # The following is for compatibility with our Matlab code
            log.info(f'Saving to {output_file_path}.mat...')
            savemat(output_file_path.as_posix() + '.mat', {'measuredPupilFunction': pupil_aberration,
                                                           'amplificationLimit': attenuation_limit,
                                                           'centerPos': [*cam.roi.top_left, *cam.roi.shape],
                                                           'initialCorrection': np.ones([1, 1], dtype=complex),
                                                           'pupilFunctionCorrection': slm.correction,
                                                           'referenceDeflectionFrequency': slm.deflection_frequency,
                                                           'slmRegionOfInterest': [*slm.roi.top_left, *slm.roi.shape],
                                                           'twoPiEquivalent': slm.two_pi_equivalent})
            log.info(f'Saving to {output_file_path}.npz...')
            np.savez(output_file_path.as_posix() + '.npz', pupil_aberration=pupil_aberration,
                     two_pi_equivalent=slm.two_pi_equivalent, deflection_frequency=slm.deflection_frequency,
                     attenuation_limit=attenuation_limit, cam_roi=cam.roi, slm_roi=slm.roi, correction=slm.correction,
                     slm_display=slm_display)

            # log.info('Sending Airy beam to first order.')
            # slm.modulate(lambda x, y: (np.sqrt(x**2 + y**2) < 300) * np.exp(2j * np.pi * 10.0 * ((x/300)**3 + (y/300)**3)))  # Airy beam

            # log.info('Sending L=1 vortex beam to first order.')
            # slm.modulate(lambda x, y: (np.sqrt(x**2 + y**2) < 3000) * np.exp(1j * np.arctan2(x, y)))  # Vortex beam, is the correction phase been added to the SLM?

            # slm.modulate(1.0)  # Send (corrected) plane wave to lens
            # img_after = cam.acquire()
            # img_after = cam.acquire()
            # img_after = cam.acquire()
            # cam_after_ax.imshow(img_after, extent=grid2extent(cam.roi.grid), cmap=plt.get_cmap('gray'))
            #
            # # log.info('Switch off correction and record again')
            # # slm.correction = 1
            # # slm.modulate(1)  # Send a plane wave without correction to lens
            # # img_before = cam.acquire()
            # # cam_before_ax.imshow(img_before, extent=grid2extent(cam.roi.grid), cmap=plt.get_cmap('gray'))
            # # plt.show(block=False)

            log.info('Close window to exit.')
            plt.show(block=True)
            log.info('Done.')
