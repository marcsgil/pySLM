#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Demonstrates the use of the spatial light modulator to measure and correct optical aberations.
#

import numpy as np
import time
import matplotlib.pylab as plt
from pathlib import Path
from examples.instruments.slm import log
from optics.instruments.cam.ids_cam import IDSCam
from optics.instruments.slm import PhaseSLM
from optics.utils.display import complex2rgb, grid2extent
from optics.calc import correction_from_pupil
from optics.utils import Roi

input_file_path = Path('../../results/aberration_correction2022-02-15_18-35-02.062751').resolve()

if __name__ == "__main__":
    log.info(f'Loading from {input_file_path}.npz...')
    correction_data = np.load(input_file_path.as_posix() + '.npz')

    # with WebCam(color=False, normalize=True) as cam:
    # with SimulatedCam(normalize=True) as cam:
    with IDSCam(normalize=True, exposure_time=5e-3) as cam:
        # cam.roi = Roi(correction_data['cam_roi'])
        # cam.roi = Roi(center=cam.roi.center, shape=[400, 400])
        log.info(correction_data['two_pi_equivalent'])
        log.info(correction_data['deflection_frequency'])
        with PhaseSLM(display_target=0, deflection_frequency=correction_data['deflection_frequency'],
                      two_pi_equivalent=correction_data['two_pi_equivalent']) as slm:
            log.info(Roi(correction_data['slm_roi']))
            slm.roi = Roi(correction_data['slm_roi'])

            # slm.post_modulation_callback = lambda _: time.sleep(100e-3)  # Forces SLM to wait before returning from slm.modulate(...)

            slm_fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Only for testing. Replace slm_ax by display number for real use
            slm_ax = axs.ravel()[0]
            slm_ax.set_title('SLM')
            cam_ax = axs.ravel()[1]
            cam_ax.set_title('Camera')
            aberration_ax = axs.ravel()[2]
            aberration_ax.set_title('aberration')
            correction_ax = axs.ravel()[3]
            correction_ax.set_title('correction')

            def display(fraction=None, pupil=None):
                img = cam.acquire()

                cam_ax.imshow(img, extent=grid2extent(*cam.roi.grid), cmap=plt.get_cmap('gray'))
                current_value = np.mean(img[img.shape[0]//2 + np.arange(3)-1, img.shape[1]//2 + np.arange(3)-1])

                if fraction is not None:
                    log.info(f'Completed {fraction * 100:0.1f}%. Intensity: {current_value}.')

                if pupil is not None:
                    aberration_ax.imshow(complex2rgb(pupil, normalization=1.0))
                    correction = correction_from_pupil(pupil)
                    correction_ax.imshow(complex2rgb(correction, normalization=1.0))

                plt.show(block=False)
                plt.pause(1e-3)

                return True

            log.info('Focusing laser to first order spot.')
            slm.modulate(1.0)  # Focus to first order.
            log.info(f"The camera's region of interest is set to {cam.roi} with integration time {cam.exposure_time * 1000:0.3f} ms.")
            time.sleep(100e-3)
            display()
            time.sleep(5)

            log.info('Maximizing intensity in first order spot.')
            slm.correction = correction_data['correction']
            slm.modulate(1.0)  # Update SLM
            time.sleep(100e-3)
            display()
            time.sleep(2)

            # log.info('Sending Airy beam to first order.')
            # slm.modulate(lambda x, y: (np.sqrt(x**2 + y**2) < 300) * np.exp(2j * np.pi * 10.0 * ((x/300)**3 + (y/300)**3)))
            # # slm.modulate(lambda x, y: (np.sqrt(x**2 + y**2) < 300) * np.exp(5j * np.arctan2(x, y)))
            # time.sleep(100e-3)
            # display()
            plt.show(block=True)
            log.info('Done.')


