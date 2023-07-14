#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This script loads an aberration from a file produced by aberration_correction.py,
# applies it to the SLM and shows the beam on the camera.
#

import numpy as np
import time
import matplotlib.pylab as plt
from pathlib import Path
from scipy.io import savemat
from datetime import datetime

from projects.transverse_structures import log
from optics.instruments.cam import SimulatedCam
from optics.instruments.cam.ids_cam import IDSCam
from optics.instruments.slm import PhaseSLM
from optics.utils.display import complex2rgb, grid2extent
from optics.calc import correction_from_pupil
# from optics.calc.special_beams import hermite_gaussian
from optics.calc.gaussian import Hermite
from optics.utils import Roi, ft

input_config_file_path = Path('../../results/aberration_correction_2022-04-26_16-28-41.300745.npz').resolve()
nb_phases = 8
output_file_path = input_config_file_path.parent / f'photonic_island_{nb_phases}_phase_images.npz'
slm_wait_time = 100e-3

grid_yx = ft.Grid([256, 256], step=[0.050e-3, 0.050e-3])  # propagation (z), projection (y), vertical (x)


if __name__ == "__main__":
    # beam = Hermite(0, 0) / np.sqrt(np.math.factorial(0) * np.math.factorial(0)) - 2.0 * Hermite(2, 0) / np.sqrt(np.math.factorial(2) * np.math.factorial(0))  # The scaling factors with respect to the previous implementation
    beam = Hermite(0, 0) - np.sqrt(2) * Hermite(2, 0)  # simplified the above

    log.info(f'Loading aberration correction from {input_config_file_path}...')
    aberration_correction_settings = np.load(input_config_file_path.as_posix())
    pupil_aberration = aberration_correction_settings['pupil_aberration']
    attenuation_limit = aberration_correction_settings['attenuation_limit']
    attenuation_limit = 4
    slm_display = aberration_correction_settings.get('slm_display')
    if slm_display is None:
        slm_display = 0

    log.info('Opening the camera...')
    # with SimulatedCam(normalize=True, exposure_time=50e-3) as cam:
    with IDSCam(normalize=True, exposure_time=50e-3) as cam:
        # Define how to measure the intensity at a specific point on the camera

        cam.roi = Roi(*aberration_correction_settings['cam_roi'])
        log.info(f"The camera's region of interest is set to {cam.roi} with integration time {cam.exposure_time * 1000:0.3f} ms.")

        log.info('Opening the spatial light modulator...')
        with PhaseSLM(display_target=slm_display, deflection_frequency=(1 / 6, 1 / 6), two_pi_equivalent=0.45) as slm:
            # Only use the low spatial frequencies of the aberration correction. The higher ones seem too noisy:
            f_abs = sum((rng * sh)**2 for rng, sh in zip(slm.roi.grid.f, slm.roi.grid.shape)) ** 0.5
            pupil_aberration = ft.ifftn(ft.fftn(pupil_aberration) * (f_abs < 5))  # only take a disk of 10 pixels diameter, the rest is noise.

            slm.two_pi_equivalent = aberration_correction_settings['two_pi_equivalent']
            slm.deflection_frequency = aberration_correction_settings['deflection_frequency']
            slm.roi = Roi(*aberration_correction_settings['slm_roi'])
            # slm.correction = aberration_correction_settings['correction']
            slm.correction *= correction_from_pupil(pupil_aberration, attenuation_limit=attenuation_limit)

            slm.post_modulation_callback = lambda _: time.sleep(slm_wait_time)  # Forces SLM to wait before returning from slm.modulate(...)

            log.info('Preparing the display...')
            slm_fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            slm_ax = axs.ravel()[0]
            slm_ax_im = slm_ax.imshow(complex2rgb(np.zeros(cam.roi.shape)), extent=grid2extent(*slm.roi.grid))
            slm_ax.set_title('SLM')
            ab_ax = axs.ravel()[1]
            ab_ax_im = ab_ax.imshow(complex2rgb(np.zeros(cam.roi.shape), normalization=1), extent=grid2extent(*slm.roi.grid))
            ab_ax.set_title('aberration')
            cor_ax = axs.ravel()[2]
            cor_ax_im = cor_ax.imshow(complex2rgb(np.zeros(cam.roi.shape)), extent=grid2extent(*slm.roi.grid))
            cor_ax.set_title('corrected')
            cam_ax = axs.ravel()[3]
            cam_ax_im = cam_ax.imshow(np.zeros(cam.roi.shape), extent=grid2extent(*cam.roi.grid), cmap=plt.get_cmap('jet'))
            cam_ax.set_title('Cam``era')

            # slm.modulate(1)
            grid = ft.Grid(shape=slm.roi.grid.shape, step=1 / 4.0)  # Step = waist size as fraction of SLM
            calibration_phases = 2 * np.pi * np.arange(nb_phases) / nb_phases
            calibration_images = np.empty([nb_phases, *cam.roi.shape])
            for _, phase in enumerate(calibration_phases):
                log.info(f'Create Hermite-Gaussian super position for phase {phase:0.1f} rad.')
                super_position_ft = beam(*grid)
                # super_position_ft2 = Hermite(0, 0)(*grid) - Hermite(2, 0)(*grid)
                # print(np.linalg.norm(super_position_ft2 - super_position_ft) / np.linalg.norm(super_position_ft2))
                super_position = ft.fftshift(ft.ifftn(ft.ifftshift(super_position_ft)))
                super_position /= np.amax(np.abs(super_position))
                log.info(f'Changing phase to {phase:0.1f} rad.')
                super_position *= np.exp(1j * phase * (grid[0] > 0))
                log.info('Sending beam to first order...')
                slm.modulate(super_position)
                # log.info('Sending Airy beam to first order.')
                # slm.modulate(lambda x, y: (np.sqrt(x**2 + y**2) < radius_on_slm) * np.exp(2j * np.pi * 10.0 * ((x/radius_on_slm)**3 + (y/radius_on_slm)**3)))
                log.info('Capturing image.')
                img = cam.acquire()
                calibration_images[_] = img

                # log.info('Displaying...')
                slm_ax_im.set_data(complex2rgb(slm.complex_field))
                ab_ax_im.set_data(complex2rgb(pupil_aberration, normalization=1))
                cor_ax_im.set_data(complex2rgb(slm.complex_field * slm.correction))
                cam_ax_im.set_data(img)
                cam_ax_im.set_clim(0, np.amax(img))
                plt.pause(0.01)
                plt.show(block=False)

    log.info(f'Saving to {output_file_path}...')
    np.savez(output_file_path, images=calibration_images, phases=calibration_phases)
    log.info('Done. Close window to exit.')
    plt.show(block=True)

    log.info('Done.')


