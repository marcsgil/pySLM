#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Axes3D import has side effects, it enables using projection='3d' in add_subplot

from examples.video import log
from optics.instruments.cam.simulated_cam import SimulatedCam
from optics.instruments.cam.web_cam import WebCam
from optics.instruments.cam.ids_peak_cam import IDSPeakCam
from optics.utils import Roi

if __name__ == '__main__':
    with WebCam() as cam:
        log.info('================================')
        log.info(f'Frame time: {cam.frame_time * 1e3:0.3f} ms, exposure time: {cam.exposure_time * 1e3:0.3f} ms')
        # log.info('Setting frame time...')
        # cam.frame_time = 100e-3
        # log.info(f'Frame time: {cam.frame_time * 1e3:0.3f} ms, exposure time: {cam.exposure_time * 1e3:0.3f} ms')
        log.info('Setting exposure...')
        cam.exposure_time = 50e-3
        log.info(f'Frame time: {cam.frame_time * 1e3:0.3f} ms, exposure time: {cam.exposure_time * 1e3:0.3f} ms')
        log.info('Setting frame time...')
        cam.frame_time = 30e-3
        log.info(f'Frame time: {cam.frame_time * 1e3:0.3f} ms, exposure time: {cam.exposure_time * 1e3:0.3f} ms')
        log.info('Setting exposure...')
        cam.exposure_time = 25e-3
        log.info(f'Frame time: {cam.frame_time * 1e3:0.3f} ms, exposure time: {cam.exposure_time * 1e3:0.3f} ms')
        log.info('Setting exposure to None...')
        cam.exposure_time = None
        log.info(f'Frame time: {cam.frame_time * 1e3:0.3f} ms, exposure time: {cam.exposure_time * 1e3:0.3f} ms')
        log.info('================================')
        nb_images = 100
        start_time = time.perf_counter()
        prev_time = start_time
        with cam.stream(nb_frames=nb_images) as image_stream:
            for img in image_stream:
                # log.info(f'==================== {idx} ====================')
                # time.sleep(500e-3)
                cur_time = time.perf_counter()
                cap_time = cur_time - prev_time
                log.info(f'capture time: {cap_time * 1e3:0.3f} ms')
                prev_time = cur_time
        avg_time = (time.perf_counter() - start_time) / nb_images
        log.info(f'Average capture time: {avg_time * 1e3:0.3f} ms')
        log.info(f'Frame time: {cam.frame_time * 1e3:0.3f} ms, exposure time: {cam.exposure_time * 1e3:0.3f} ms')
