import numpy as np
import time
import matplotlib.pyplot as plt
# import matplotlib.colors
import matplotlib  # .colors.Normalize as Normalise
from mpl_toolkits.mplot3d import axes3d

from optics.instruments.stage.nanostage import NanostageLT3
from optics.instruments.cam.ids_cam import IDSCam
from optics.utils.roi import Roi
from optics.utils.ft import Grid
from optics.utils.display import grid2extent
from timeit import default_timer as timer


def scan_and_save():
    # Stage parameters
    stage = NanostageLT3()
    scan_axes = (1, 2)

    grid = Grid(extent=(299e-6, 0.5e-6), step=0.5e-6, center=(150e-6, 150e-6))
    # # TODO: Testing: remove when performing the real scan
    # grid = Grid(extent=(2.5e-6, 2.5e-6), step=0.25e-6, center=(150e-6, 115e-6))

    run_clock = timer()

    # Camera parameters
    with IDSCam(exposure_time=10e-3) as camera:
        camera.roi = Roi(center=(512, 770), shape=(32, 32))

        camera.acquire()  # This is a test frame. The first acquire command returns a darker frame

        images = []
        for idx_y, coord_y in enumerate(grid[1].flat):
            stage.move(scan_axes[1], coord_y * 1e9, 'nm')
            images.append([])
            stage.move(scan_axes[0], grid[0].flat[0] * 1e9, 'nm')
            time.sleep(0.3)
            for idx_x, coord_x in enumerate(grid[0].flat):
                stage.move(scan_axes[0], coord_x * 1e9, 'nm')
                images[idx_y].append(camera.acquire())

            # status variables
            print(f'Current points: [{coord_x / 1E-9}nm, {coord_y / 1E-9}nm]')

        images = np.asarray(images, dtype=float)
        image_intensities = np.mean(images, axis=(2, 3))
        print(image_intensities.shape)

        # recentering
        stage.move(scan_axes[0],  grid.center[0] * 1e9, 'nm')
        stage.move(scan_axes[1], grid.center[1] * 1e9, 'nm')

        scan_time = timer() - run_clock
        print(f'Scan time: {scan_time}')

        fig, axs = plt.subplots(1, 2)
        # print(np.shape(np.sum(images, axis=(1, 2))))
        axs[0].imshow(images[0, 0])
        img = axs[1].imshow(image_intensities)  # , extent=grid2extent(grid * 1e6))
        axs[1].set(xlabel='x $\mu m$', ylabel='y/z $\mu m$')
        plt.colorbar(img)
        plt.show(block=False)

        # Save images
        iteration = 0
        while True:
            try:
                iteration += 1
                np.save(f'scan{iteration}.npy', images)
                break
            except Exception as e:
                print(e)

        # send_email(f'Scan finished. Scan time -- {scan_time}s.', receiver_email='lau.valantinas@gmail.com')


if __name__ == "__main__":
    scan_and_save()
    plt.show(block=True)
