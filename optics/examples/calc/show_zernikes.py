#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from examples.calc import log
from optics.calc import zernike
from optics.utils import ft
from optics.utils.display import complex2rgb, grid2extent, format_image_axes

if __name__ == '__main__':
    grid = ft.Grid(np.full(2, 512), extent=2.5)

    aperture = sum(_**2 for _ in grid) < 1

    #
    # Display
    #
    fig = plt.figure(figsize=(12, 8))
    max_order = 6
    ax_width = 1 / (max_order + 1)
    ax_height = 1 / (max_order + 1)
    for n in range(max_order + 1):
        ax_bottom = 1 - ax_height - n * ax_height
        for m in range(-n, n + 1, 2):
            z = zernike.BasisPolynomial(n=n, m=m)
            log.info(f'Plotting {z.symbol} | {z.name}...')

            ax_left = 0.5 * (1 - ax_width) + (m / 2) * ax_width
            fld = aperture * np.exp(2j * np.pi * z.cartesian(*grid))
            ax = plt.axes([ax_left, ax_bottom, ax_width, ax_height])
            ax.imshow(complex2rgb(fld, alpha=1.0), extent=grid2extent(grid))
            ax.set(xticks=[], yticks=[])
            ax.axis('off')
            ax.add_artist(AnchoredText(f'$Z_{{{n}}}^{{{m}}}$', loc='upper left', borderpad=0.0, frameon=False,
                                       prop=dict(color=[0, 0, 0, 1], fontweight='bold', fontsize=10)))
            ax.add_artist(AnchoredText(f'{z.name}', loc='lower center', borderpad=0.0, frameon=False,
                                       prop=dict(color=[0, 0, 0, 1], fontweight='normal', fontsize=8)))

    log.info('Done.')
    plt.show(block=True)
