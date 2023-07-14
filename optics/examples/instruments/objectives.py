#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

#
# Demonstrates how to use the objective module, displaying some info on objectives.
#

from examples.instruments import log
from optics.instruments import objective


if __name__ == '__main__':
    objectives = [
        objective.Olympus(20, 0.50),
        objective.Nikon(40, 0.80),
        objective.Mitutoyo(20, 0.42),
        objective.EdmundOptics(100, 0.55),
    ]
    wavelength = 633e-9

    log.info(f'At a wavelength of {wavelength / 1e-9:0.0f}nm')
    for obj in objectives:
        log.info(f"{obj} has a back aperture diameter of {obj.pupil_diameter * 1e3:0.1f} mm and a focal length of {obj.focal_length * 1e3:0.1f} mm.")
        resolution = wavelength / (2 * obj.numerical_aperture)
        resolution_on_camera = resolution * obj.magnification
        log.info(f'{obj} can resolve {resolution / 1e-6:0.3f} um'
                 + f' and needs a camera with pixels smaller than {resolution_on_camera / 2 / 1e-6:0.1f} um'
                 + f' when using a tube lens of {obj.tube_lens_focal_length / 1e-3} mm.')

    log.info('Check simulate_image.py for examples on how to calculate a PSF for an objective.')
