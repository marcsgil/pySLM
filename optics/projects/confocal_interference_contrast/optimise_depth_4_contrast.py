import numpy as np
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer

from optics.instruments.stage.nanostage import NanostageLT3
from optics.instruments.cam.ids_cam import IDSCam
from optics.utils.ft import Grid

time.sleep(15)
with IDSCam(exposure_time=5e-3) as cam:
    stage = NanostageLT3()

    initial_position = [150e-6, 150e-6, 150e-6]
    for idx, position in enumerate(initial_position):
        stage.move(axis=idx, value=position * 1e9, unit='nm')

    depth_axis = 0
    scan_axis = 1
    axes = (depth_axis, scan_axis)
    center = [initial_position[idx] for idx in axes]
    grid = Grid(extent=(40e-6, 12e-6), step=[0.25e-6, 1e-6], center=center)

    result = np.zeros((grid.shape[0]))
    start_time, nb_iterations = timer(), len(grid[0].flat)  # for time keeping
    for idx, depth_coord in enumerate(grid[0].flat):
        stage.move(axis=axes[0], value=depth_coord * 1e9, unit='nm')
        time.sleep(0.3)
        # One shot intensities
        avg_intensities = []
        for scan_coord in grid[1].flat:
            stage.move(axis=axes[1], value=scan_coord * 1e9, unit='nm')
            avg_intensities.append(np.mean(cam.acquire()))
        result[idx] = np.max(avg_intensities) - np.min(avg_intensities)

        # Informing about the progress
        if idx == 0:
            iteration_time = timer() - start_time
        elif idx == 4:
            iteration_time = (timer() - start_time) / 5
        print(f'progress {(idx + 1) / nb_iterations * 100:.2f}%.' +
              f' Time left: {iteration_time * (nb_iterations - idx - 1):.2f}s...')

    # returning to initial position
    for idx, position in enumerate(initial_position):
        stage.move(axis=idx, value=position * 1e9, unit='nm')

    result = np.asarray(result)
    best_contrast_coord = grid[0][np.argmax(result)]
    print(f'Best contrast for the given range is achieved at {best_contrast_coord[0] * 1e6}um')

    plt.plot(np.array(grid[0].flat) * 1e6, result)
    plt.xlabel('Depth coordinates, $\mu m$')
    plt.ylabel('Intensity contrast, au')
    plt.title('Depth coordinate vs scan contrast')
    plt.show(block=True)
