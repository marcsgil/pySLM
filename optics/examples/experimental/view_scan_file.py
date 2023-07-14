import numpy as np
import matplotlib.pyplot as plt

from optics.utils.ft import Grid
from optics.utils.display import complex2rgb, grid2extent


def data_analysis():
    image_data = np.load('Highlighter scan - Large area - confocal DIC.npy')
    print(image_data.shape)

    grid = Grid(image_data.shape, center=(0, 0, -4, -4))
    r = (grid[2] ** 2 + grid[3] ** 2) ** 0.5


    # cropped = image_data[:, :, 20, 20].reshape(image_data.shape[:2])
    cropped = np.mean(image_data * (r > 6), axis=(-1, -2))

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(cropped)
    img = axs[1].imshow(((r > -6) * image_data)[0, 150, :, :])
    plt.show(block=False)
    for _ in range(1, 598, 7):
        img.set_data(((r > -6) * image_data)[_, 150, :, :])
        axs[1].set_title(f'Spot {_}')
        plt.pause(0.001)


def window_avg(data, window: int = 3):
    if window < 1:
        raise

    avg_data = data  # Data that will be window averaged
    windows = np.arange(window - 1)
    windows = windows - (windows / 2).astype(int)
    for idx in windows:
        avg_data += np.roll(data, idx)
    avg_data /= window
    return avg_data

def line_segment_analysis():
    image_data = {}
    try:
        for idx in range(1, 100):
            image_data[idx] = np.load(f'linescan{idx}.npy')
    except Exception:
        pass

    moving_avg = {}
    for idx in image_data:
        image_data[idx] = np.mean(image_data[idx][..., 10:20, 10:20], axis=(-1, -2)).ravel()
        moving_avg[idx] = window_avg(image_data[idx], 9)

    plot_dimx = int(len(image_data) ** 0.5 + 0.5)
    plot_dimy = int(len(image_data) ** 0.5)
    fig, axs = plt.subplots(plot_dimx, plot_dimy)

    idx = 1
    for _ in axs:
        for ax in _:
            ax.plot(image_data[idx])
            ax.set_title(f'scan no. {idx}')
            idx += 1
    plt.show(block=False)



if __name__ == '__main__':
    data_analysis()
    line_segment_analysis()

    plt.show(block=True)
