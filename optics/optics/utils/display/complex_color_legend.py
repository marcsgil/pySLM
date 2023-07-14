#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Displays the color legend for images shown with complex2rgb, and saves it as svg vector graphics.
#

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.lines import Line2D
from typing import Optional

from optics.utils.display import complex2rgb, grid2extent
from optics.utils.ft import Grid


__all__ = ['draw', 'draw_real']


def draw(figure_axes: plt.Axes = None, inverted:bool = False, show_out_of_bounds: bool = False,
         foreground_color=None, saturation: float = 1.0):
    grid = Grid(1024, extent=(4, 4), include_last=True)

    high_value_color = (0, 0, 0, 1) if inverted else (1, 1, 1, 1)
    low_value_color = (1, 1, 1, 1) if inverted else (0, 0, 0, 1)

    if foreground_color is None:
        foreground_color = high_value_color

    complex_values = grid[0] + 1j * grid[1]

    if figure_axes is None:
        figure, figure_axes = plt.subplots(1, 1, figsize=(5, 5), facecolor=None)
        figure.tight_layout(pad=0.0)
        figure_axes.set_position((0, 0, 1, 1))  # Use the whole figure
    else:
        figure = figure_axes.figure
    axes_size = (figure_axes.get_position().width, figure_axes.get_position().height)
    font_size = 32 * min(figure.get_size_inches() * figure.dpi * axes_size) / 500

    img = figure_axes.imshow(complex2rgb(complex_values.T[::-1, :] * saturation, inverted=inverted),
                             extent=grid2extent(grid, origin_lower=True))
    circle = Circle((0, 0), radius=1, linewidth=2, linestyle='-', edgecolor=low_value_color, facecolor=(1, 1, 1, 0))
    clip_circle = Circle((0, 0), radius=1 + show_out_of_bounds, transform=figure_axes.transData, edgecolor=(0, 0, 0, 0))
    img.set_clip_path(clip_circle)
    arrow_h = FancyArrow(-1.1, 0, 2.33, 0, width=0.04, length_includes_head=True, facecolor=foreground_color, edgecolor=(0, 0, 0, 0))
    arrow_v = FancyArrow(0, -1.1, 0, 2.33, width=0.04, length_includes_head=True, facecolor=foreground_color, edgecolor=(0, 0, 0, 0))
    font_dict = {'fontstyle': 'italic', 'fontweight': 'normal', 'fontsize': font_size, 'fontfamily': 'sans-serif'}
    label_0 = plt.text(0.06, 0.06, '0', fontdict=font_dict, color=high_value_color)
    label_1 = plt.text(1.06, 0.06, '1', fontdict=font_dict, color=foreground_color)
    label_i = plt.text(0.06, 1.06, 'i', fontdict=font_dict, color=foreground_color)
    figure_axes.add_artist(circle)
    figure_axes.add_artist(arrow_h)
    figure_axes.add_artist(arrow_v)
    figure_axes.add_artist(label_0)
    figure_axes.add_artist(label_1)
    figure_axes.add_artist(label_i)
    figure_axes.axis('off')

    return figure, figure_axes


def draw_real(figure_axes: plt.Axes = None, inverted:bool = False, show_out_of_bounds: bool = False,
              foreground_color: Optional[str] = None, saturation: float = 1.0):
    grid = Grid(1024, extent=(2.5, 2.5), include_last=True)

    high_value_color = (0, 0, 0, 1) if inverted else (1, 1, 1, 1)
    low_value_color = (1, 1, 1, 1) if inverted else (0, 0, 0, 1)

    if foreground_color is None:
        foreground_color = high_value_color

    complex_values = grid[0] + 0 * grid[1]

    if figure_axes is None:
        figure, figure_axes = plt.subplots(1, 1, figsize=(5, 5), facecolor=None)
        figure.tight_layout(pad=0.0)
        figure_axes.set_position((0, 0, 1, 1))  # Use the whole figure
    else:
        figure = figure_axes.figure
    axes_size = (figure_axes.get_position().width, figure_axes.get_position().height)
    font_size = 2*12 * min(figure.get_size_inches() * figure.dpi * axes_size) / 500

    img = figure_axes.imshow(complex2rgb(complex_values[:, ::-1] * saturation, inverted=inverted),
                             extent=grid2extent(grid, origin_lower=True))
    box_width = 2/12 * (1 + show_out_of_bounds)
    box = Rectangle((-box_width/2, -1), width=box_width, height=2, linewidth=2, linestyle='-', edgecolor=high_value_color, facecolor=(1.0, 1.0, 1.0, 0.0))
    clip_box = Rectangle((-box_width/2, -1), box.get_width(), height=box.get_height(), transform=figure_axes.transData,
                         linewidth=2, linestyle='-', edgecolor=low_value_color, facecolor=(1.0, 1.0, 1.0, 0.0))
    img.set_clip_path(clip_box)

    font_dict = {'fontstyle': 'italic', 'fontweight': 'normal', 'fontsize': font_size, 'fontfamily': 'sans-serif'}
    for _ in (-1, 0, 1):
        line = Line2D((-1.2 * box_width/2, -box_width/2), (_, _), linewidth=2, linestyle='-', color=foreground_color)
        label = plt.text(-box_width/2 - 0.03, _, f'{_}', fontdict=font_dict, color=foreground_color, ha='right', va='center')
        figure_axes.add_artist(line)
        figure_axes.add_artist(label)
    figure_axes.add_artist(box)
    figure_axes.axis('off')

    return figure, figure_axes
