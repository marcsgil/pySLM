#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

#
# Demonstration of the use of the IBeam SMART laser class.
#

import numpy as np
import time
from scipy.constants import convert_temperature

from examples.instruments import log
from optics.instruments.source.toptica import IBeamSMARTLaser

if __name__ == '__main__':
    with IBeamSMARTLaser() as laser:
        log.info(f'Using {laser}.')

        laser.power = 1e-3
        laser.emitting = True

        log.info(f'Laser temperature: {laser.temperature:0.1f} K = {convert_temperature(laser.temperature, "Kelvin", "Celsius"):0.1f}.')
        log.info(f'Fine settings: A = {laser.fine_a*100:0.1f}%, B = {laser.fine_b*100:0.1f}%.')

        # Switch SKILL on and off
        for _ in range(10):
            laser.skill = False
            log.info('Skill OFF')
            time.sleep(0.5)
            laser.skill = True
            log.info('Skill ON')
            time.sleep(0.5)

        # Ramp up the power repeatedly
        # for _ in range(600):
        #     for p in np.arange(0e-3, 10e-3, 1e-3) + 1e-3:
        #         laser.emitting = True
        #         laser.power = p
        #         log.info(f'P = {laser.power*1e3:0.1f} mW at wavelength {laser.wavelength*1e9:0.1f} nm.')
        #
        #         time.sleep(0.1)

    log.info('Done.')
