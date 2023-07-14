import numpy as np
import matplotlib.pyplot as plt
import json
import time

from projects.confocal_interference_contrast import log
from optics.utils.display import complex2rgb, grid2extent
from optics.utils.ft.grid import Grid
from optics.utils.ft.subpixel import register, roll, Reference, roll_ft, Registration
from optics.utils import ft
from optics.instruments.cam.ids_cam import IDSCam
from projects.confocal_interference_contrast.complex_phase_image import Interferogram, get_registration
from optics.instruments.stage.piezoconcept import PiezoConceptStage


def scan():
    exposure_time = 10e-3
    with IDSCam(exposure_time=exposure_time, normalize=True) as cam, PiezoConceptStage() as stage:
        grid = Grid(shape=cam.shape, step=cam.pixel_pitch)
        registration = get_registration()
        complex_phase_image = Interferogram(cam.acquire(), registration=registration)  # TODO should specify registration before running

        positions = [stage.position]
        new_position = positions[0].copy()
        new_position[1] += 5e-6
        positions.append(new_position)

        interference_images = []
        phase_images = []
        for idx, pos in enumerate(positions):
            stage.position = pos
            time.sleep(0.2)
            interference_images.append(cam.acquire())
            phase_images.append(complex_phase_image.fast_complex_update_and_return_previous(interference_images[idx]))

        # returning to the old position
        stage.position = positions[0]

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for idx, ax in enumerate(axs):
        ax[0].imshow(interference_images[idx])
        ax[1].imshow(complex2rgb(phase_images[idx], 50))
    plt.show(block=False)


if __name__ == '__main__':
    scan()
    plt.show(block=True)

