#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Axes3D import has side effects, it enables using projection='3d' in add_subplot

from examples.calc import log
from optics.utils import pol2cart
from optics.calc.pupil_equation import parse
from optics.utils.display import complex2rgb, grid2extent


if __name__ == '__main__':
    p = parse('((0.9<r) & (r<=1)) * exp(i*phi)')
    log.info(p.__str__)

    rho = np.arange(100 + 1)[:, np.newaxis] / 100
    phi = 2 * np.pi * np.arange(100 + 1)[np.newaxis, :] / 100
    y_polar, x_polar = pol2cart(rho, phi)

    rng = slice(-1, 1, 100j)
    x, y = np.ogrid[rng, rng]

    fig = plt.figure()
    axs = [plt.subplot(1, 2, 1, projection='3d'), plt.subplot(1, 2, 2)]

    axs[0].plot_surface(x_polar, y_polar, np.angle(p(x_polar, y_polar)))
    axs[0].set(xlabel='x', ylabel='y', zlabel='p(x, y)')
    plt.get_current_fig_manager().toolbar.pan()  # Switch on pan already

    axs[1].imshow(complex2rgb(p(x, y)), extent=grid2extent(x, y))
    axs[1].set(xlabel='x', ylabel='y', title='p(x, y)')

    plt.show(block=True)
