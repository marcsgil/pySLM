import numpy as np

import logging

log = logging.getLogger(__name__)

from optics.utils import Roi
from optics.instruments.cam import Cam


def center_roi_around_peak_intensity(cam: Cam, shape=(8, 8), target_graylevel_fraction: float = 0.75) -> Cam:
    """
    Acquires an image from the camera, finds the position of maximal intensity and centers the cam's ROI
    (a rectangle with the given shape or (8, 8) as default) around it.
    If the image is saturated it halves the cam's integration time until it isn't.

    (To do: interact with the slm to acquire the background)

    :param cam: This argument is updated and returned.
    :param shape: The shape of the roi, by default is (8, 8)
    :param target_graylevel_fraction: The camera's exposure is changed until the gray level is approximately this value
        times its maximum.
    :return: The updated camera object. Only the roi and integration_time are changed.
    """

    log.warning("Please use Cam.center_roi_around_peak_intensity(shape) now instead.")
    if np.all(cam.roi.shape > 0):
        normalize = cam.normalize
        color = cam.color
        cam.color = False

        img = cam.acquire()

        peak_index = np.argmax(img)
        max_graylevel = (2**cam.bit_depth - 1)
        # if the image is saturated:
        for attempt in range(32):
            if img.ravel()[peak_index] >= max_graylevel:
                cam.exposure_time *= 0.5
                img = cam.acquire()
                peak_index = np.argmax(img)
            else:
                break
        # Set exposure so that the dynamic range is at a given level
        cam.exposure_time /= img.ravel()[peak_index] / max_graylevel / target_graylevel_fraction

        center = np.unravel_index(peak_index, shape=img.shape)
        relative_roi = Roi(shape=shape, center=center, dtype=np.int)

        absolute_roi = relative_roi * cam.roi

        cam.roi = absolute_roi

        cam.color = color
        cam.normalize = normalize

    return cam

