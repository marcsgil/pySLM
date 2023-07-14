#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Demonstrates the use of the spatial light modulator to measure and correct optical aberations.
#

import numpy as np
import matplotlib.pylab as plt
import time

from examples.instruments.slm import log
from optics.instruments.slm import PhaseSLM
from pathlib import Path
from optics.utils import Roi
from optics.utils.display import complex2rgb


input_file_path = Path('../../results/aberration_correction_2022-05-12_15-54-37.439768').resolve()
log.info(f'Loading from {input_file_path}.npz...')
correction_data = np.load(input_file_path.as_posix() + '.npz')

if __name__ == '__main__':
    with PhaseSLM(display_target=0, deflection_frequency=correction_data['deflection_frequency'],
                  two_pi_equivalent=correction_data['two_pi_equivalent']) as slm:
        log.info(Roi(correction_data['slm_roi']))
        slm.roi = Roi(correction_data['slm_roi'])
    # testing below
    # slm = PhaseSLM(screen_or_axis=0, deflection_frequency=correction_data['deflection_frequency'],
    #               two_pi_equivalent=correction_data['two_pi_equivalent'])
    # log.info(Roi(correction_data['slm_roi']))
    # slm.roi = Roi(correction_data['slm_roi'])


    # def test(m):
    #     if m==1:
    #         te = lambda u, v: (np.sqrt(u**2 + v**2) < 150) * np.exp(1j * 4 * np.arctan2(u, v))
    #     else:
    #         te=1.0
    #     return te
   #slm.modulate(lambda u, v: (np.sqrt(u**2 + v**2) < 150) * np.exp(1j * 4 * np.arctan2(u, v)))
    #slm.modulate(test(0))
        #slm.modulate(lambda u, v: (np.sqrt(u**2 + v**2) < 100))
        #slm.modulate(np.exp(2j*np.pi * -0.4))

        slm.modulate(1.0)
        log.info('Close window to close program.')
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(complex2rgb(slm.complex_field))
        axs[1].imshow(slm.image_on_slm)
        plt.show(block=True)
