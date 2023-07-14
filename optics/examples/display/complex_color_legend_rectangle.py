import numpy as np
import matplotlib.pyplot as plt

from optics.utils import ft
from optics.utils.display import complex2rgb, grid2extent
from optics import log


def background(shape):
    grid = ft.Grid(shape[:2])
    checker_board = sum((_//16) % 2 for _ in grid) % 2
    checker_board = np.repeat(checker_board[..., np.newaxis], 3, axis=-1).astype(float)
    return checker_board


def alpha_blend(*images):
    result = 0.0
    for _ in images:
        if _.shape[-1] > 3:
            alpha = np.clip(_[..., 3:], 0.0, 1.0)
            result *= 1.0 - alpha
            result += _[..., :3]
        else:
            result = _[..., :3]
    return result


def overlay(img):
    return alpha_blend(background(img.shape[:2]), img)


if __name__ == '__main__':
    grid = ft.Grid([256, 512], extent=[2, 2 * np.pi], first=[0, -np.pi])
    fld = grid[0] * np.exp(1j * grid[1])

    fig, axs = plt.subplots(3, 2, sharex='all', sharey='all')
    axs[0, 0].imshow(complex2rgb(fld), extent=grid2extent(grid, origin_lower=True), origin='lower')
    axs[0, 0].set_title('default')
    axs[0, 0].set(xlabel='phase', ylabel='amplitude')
    axs[0, 1].imshow(complex2rgb(fld, inverted=True), extent=grid2extent(grid, origin_lower=True), origin='lower')
    axs[0, 1].set_title('inverted')
    axs[0, 1].set(xlabel='phase', ylabel='amplitude')
    axs[1, 0].imshow(overlay(complex2rgb(fld, alpha=1.0)), extent=grid2extent(grid, origin_lower=True), origin='lower')
    axs[1, 0].set_title('alpha')
    axs[1, 0].set(xlabel='phase', ylabel='amplitude')
    axs[1, 1].imshow(overlay(complex2rgb(fld, inverted=True, alpha=1.0)), extent=grid2extent(grid, origin_lower=True), origin='lower')
    axs[1, 1].set_title('inverted alpha')
    axs[1, 1].set(xlabel='phase', ylabel='amplitude')
    axs[2, 0].imshow(overlay(complex2rgb(fld, alpha=2)), extent=grid2extent(grid, origin_lower=True), origin='lower')
    axs[2, 0].set_title('alpha 2')
    axs[2, 0].set(xlabel='phase', ylabel='amplitude')
    axs[2, 1].imshow(overlay(complex2rgb(fld, inverted=True, alpha=2)), extent=grid2extent(grid, origin_lower=True), origin='lower')
    axs[2, 1].set_title('inverted alpha 2')
    axs[2, 1].set(xlabel='phase', ylabel='amplitude')

    log.info('Showing the complex2rgb colormap as a rectangle.')
    plt.show()

