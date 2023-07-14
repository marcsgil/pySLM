import numpy as np
import matplotlib.pyplot as plt

from examples.experimental import log
from optics.utils.ft.convolution import *
from optics.utils.ft import Grid
from optics.utils import ft


if __name__ == '__main__':
    grid = Grid([128])
    obj = np.zeros(grid.shape)
    # obj[5] = 0.75
    obj[obj.size//2] = 1
    # obj[-10] = 0.5

    kernel = np.zeros(grid.shape, dtype=complex)
    for idx in range(kernel.size//4):
        kernel[idx] = 1 / (idx + 1j)
        if idx > 0:
            kernel[-idx] = -1 / (idx + 2j)
    kernel_ft = ft.fft(kernel) * (np.logical_and(-1/16 <= grid.f[0], grid.f[0] < 1/16))  # bandpass
    kernel_ft = 1 / (grid.k[0]**2 + (0.1j))
    kernel = ft.ifft(kernel_ft)

    log.info('Doing cyclic convolution...')
    c_img = cyclic_conv(obj, kernel)
    log.info('Doing padded convolution...')
    p_img = padded_conv(obj, kernel)
    log.info('Doing acyclic convolution...')
    ac_img = acyclic_conv(obj, kernel)
    log.info('Done!')

    fig, ax = plt.subplots(2, 3, sharex='all', sharey='all')
    ax[0, 0].plot(grid[0], ft.fftshift(kernel).real)
    ax[0, 0].plot(grid[0], ft.fftshift(kernel).imag)
    ax[0, 0].set(xlabel='x', title='kernel')
    # ax[0, 1].plot(grid[0], obj.real)
    # ax[0, 1].plot(grid[0], obj.imag)
    # ax[0, 1].set(xlabel='x', title='object')
    ax[1, 0].plot(grid[0], c_img.real)
    ax[1, 0].plot(grid[0], c_img.imag)
    ax[1, 0].set(xlabel='x', title='cyclic')

    # ax[0, 1].plot(p_img.real)
    # ax[0, 1].plot(p_img.imag)
    # ax[0, 1].set(xlabel='x', title='padded')
    ax[0, 1].plot(grid[0], p_img.real)
    ax[0, 1].plot(grid[0], p_img.imag)
    ax[0, 1].set(xlabel='x', title='padded')

    ax[1, 1].plot(grid[0], ac_img.real)
    ax[1, 1].plot(grid[0], ac_img.imag)
    ax[1, 1].set(xlabel='x', title='acyclic')

    ax[0, 2].plot(grid[0], (c_img - p_img).real)
    ax[0, 2].plot(grid[0], (c_img - p_img).imag)
    ax[0, 2].set(xlabel='x', title='cyclic - padded')

    ax[1, 2].plot(grid[0], (ac_img - p_img).real)
    ax[1, 2].plot(grid[0], (ac_img - p_img).imag)
    ax[1, 2].set(xlabel='x', title='acyclic - padded')

    plt.show(block=True)
