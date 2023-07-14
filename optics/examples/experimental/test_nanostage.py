from optics.instruments.stage.nanostage import NanostageLT3
from optics.instruments.cam.ids_cam import IDSCam
from optics.utils.roi import Roi
from optics.utils.ft import Grid

import matplotlib.pyplot as plt
import numpy as np
import time

with IDSCam(exposure_time=4e-3, frame_time=5e-3) as camera:
    stage = NanostageLT3()
    camera.roi = Roi(center=(452, 536), shape=(1, 1))
    grid = Grid(center=[150e-6], extent=20e-6, step=50e-9)

    stage.move(2, 71, 'um')
    img = []
    for y in grid[0].flat:
        # stage.move_rel(1, 5, 'nm')
        stage.move(1, y * 1e9, 'nm')
        # time.sleep(0.01)
        img.append(camera.acquire())
    img = np.array(img).ravel()

    stage.move(1, grid.center, 'nm')

    plt.plot(np.array(grid[0].flat) * 1e6, img)
    plt.show()
