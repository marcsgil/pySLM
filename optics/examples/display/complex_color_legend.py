#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Displays the color legend for images shown with complex2rgb, and saves it as svg vector graphics.
#

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.lines import Line2D
import pathlib

from optics.utils.display import complex2rgb, grid2extent, complex_color_legend
from optics.utils.ft import Grid

from examples.display import log

if __name__ == '__main__':
    fig, ax = complex_color_legend.draw(inverted=False, foreground_color=[0, 0, 0])

    output_path = pathlib.Path(__file__).parent
    file_name = pathlib.Path(__file__).name[:-3]
    log.info(f'Saving to {output_path}/{file_name}...')
    fig.savefig(output_path / (file_name + '.png'), bbox_inches='tight', format='png', transparent=True)
    fig.savefig(output_path / (file_name + '.svg'), bbox_inches='tight', format='svg', transparent=True)
    fig.savefig(output_path / (file_name + '.pdf'), bbox_inches='tight', format='pdf', transparent=True)
    log.info(f'Saved to {output_path}/{file_name}...')
    plt.show(block=True)
