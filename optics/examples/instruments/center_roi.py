#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Demonstrates the center_roi_around_peak_intensity method of the Cam class.
# This centers the region of interest of the camera around the brightest spot in the current region of interest.
#

import numpy as np
import matplotlib.pylab as plt

from examples.instruments import log
from optics.instruments.cam import Cam
from optics.instruments.cam.simulated_cam import SimulatedCam
from optics.instruments.cam.web_cam import WebCam
from optics.utils import Roi
from optics.utils.display import grid2extent


def center_roi():
    noise_level = 0.50
    pos = [5, 10]

    def simulate_image(c: Cam):  # Only used for SimulatedCam
        sim_img = np.zeros(c.shape)
        sim_img = noise_level * np.random.rand(*sim_img.shape)
        sim_img[pos[0], pos[1]] = 1.0
        return (sim_img * 255).astype(np.uint8)

    with SimulatedCam(get_frame_callback=simulate_image) as cam:
    # with WebCam() as cam:
        cam = cam.center_roi_around_peak_intensity(shape=(7, 7))
        img = cam.acquire()
        log.info(f"roi={cam.roi}, img.shape={img.shape}")
        log.info(f"roi.center={cam.roi.center} should be {pos}")
        if np.all(cam.roi.center == pos):
            log.info("CORRECT")
        else:
            log.info("ERROR")

        fig, ax = plt.subplots(1, 1)
        ax.imshow(img, extent=grid2extent(*cam.roi.grid), cmap=plt.get_cmap('gray'))
        plt.show(block=True)


if __name__ == "__main__":
    center_roi()
