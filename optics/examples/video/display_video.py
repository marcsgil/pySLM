#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#


#
# A simple example of how to capture video frame-by-frame
# See also: examples/capture_video.py and examples/stream_video.py to see how to work with more efficient video streams.
#

import numpy as np
import matplotlib.pyplot as plt

from examples.video import log
from optics.instruments.cam import Cam
from optics.instruments.cam.simulated_cam import SimulatedCam
# from optics.instruments.cam.web_cam import WebCam
from optics.instruments.cam.ids_cam import IDSCam
from optics.utils import Roi


if __name__ == '__main__':
    plt_image = None
    #with WebCam() as cam:
    with IDSCam(1) as cam:
        cam.exposure_time = 10e-3
        cam.roi = None  # Roi(shape=(480, 640), center=cam.shape // 2)
        log.info(f'Set integration time to {cam.exposure_time} and region-of-interest to {cam.roi}.')
        for idx in range(100):
            log.info(f"Acquiring frame {idx} of shape {cam.roi.shape}...")
            img = cam.acquire()
            log.info(f"Acquired frame {idx}.")
            if np.amax(np.amax(img)) > 0:
                img /= np.amax(img)
            else:
                log.info('Image is all black!')
            if plt_image is None:
                plt_image = plt.imshow(img)
                plt.colorbar()
            else:
                plt_image.set_data(img)
            plt.title(f"frame {idx}")
            plt.pause(0.001)
            plt.draw()
            log.info(f"Displayed frame {idx}.")

        log.info("Done!")
