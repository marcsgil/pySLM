from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from projects.confocal_interference_contrast import log
from optics.utils.ft import Grid


def confocal(data_cube: np.ndarray, pinhole_position = (0, 0), pinhole_radius: float = 1.0) -> np.ndarray:
    pinhole_position = np.atleast_1d(pinhole_position)
    log.info(f'Processing {data_cube.shape[0]}x{data_cube.shape[1]} images of shape {data_cube.shape[2]}x{data_cube.shape[3]}...')

    grid = Grid(shape=data_cube.shape[2:4])

    mask = (grid[0] - pinhole_position[0]) ** 2 + (grid[1] - pinhole_position[1]) ** 2 < pinhole_radius ** 2
    projection = np.sum(data_cube[:, :, mask], axis=-1)

    return projection


def image_scanning_microscopy(data_cube: np.ndarray, scan_step = 0.25e-6, pixel_step = 5.3e-6 / 10)  -> np.ndarray:
    log.info(f'Processing {data_cube.shape[0]}x{data_cube.shape[1]} images of shape {data_cube.shape[2]}x{data_cube.shape[3]}...')

    output_shape = 2 * np.asarray(data_cube.shape[:2])  # up-sample by a factor of 2

    scan_grid = Grid(shape=data_cube.shape[:2], step=scan_step)
    pixel_grid = Grid(shape=data_cube.shape[2:], step=pixel_step)
    output_grid = Grid(shape=output_shape, step=np.asarray(scan_step)/2)

    projection = np.zeros(output_shape)
    for y_idx, y in enumerate(scan_grid[0].flat):
        log.info(f'y = {y*1e6:0.3f}um')  # todo: This is a very inefficient way of doing these integrals
        for x_idx, x in enumerate(scan_grid[1].flat):
            spot_interpolator = interpolate.interp2d(
                y - (pixel_grid[0] / 2).flat, x - (pixel_grid[1] / 2).flat, data_cube[y_idx, x_idx, :, :],
                kind='linear', fill_value=0.0)
            resampled_spot = spot_interpolator(output_grid[0].flat, output_grid[1].flat)
            projection += resampled_spot

    return projection




if __name__ == '__main__':
    path = Path(r'C:\Users\tvettenburg\Downloads')
    file_path = path / 'Large_ROI_scan_of_scale_5ms.npy'

    log.info(f'Reading {file_path}...')
    data_cube = np.load(str(file_path))

    plt.ion()
    fig, ax = plt.subplots(1, 1)
    img_data = image_scanning_microscopy(data_cube, scan_step = 0.25e-6, pixel_step = 5.3e-6 / 10)  # The effective magnification was 10x
    ax.imshow(img_data / np.amax(img_data))
    plt.show(block=True)
    exit(0)

    im = ax.imshow(np.zeros(data_cube.shape[:2]), vmin=0.0, vmax=1.0)

    # img_data = confocal(data_cube, pinhole_position=(1, 1), pinhole_radius=1)
    # im.set_data(img_data / np.amax(img_data))
    # plt.show(block=True)
    # exit(0)

    view_range = np.arange(-5, 5)
    for y in np.tile([*view_range, *view_range[::-1]], 1):
        for x in np.tile([*view_range, *view_range[::-1]], 1):
            img_data = confocal(data_cube, pinhole_position=(y, x), pinhole_radius=1)
            im.set_data(img_data / np.amax(img_data))
            ax.set_title(f'x, y = {x}px, {y}px')

            plt.show(block=False)
            plt.pause(0.1)

    plt.show(block=True)

    log.info('Done!')
