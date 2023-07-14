import numpy as np
from typing import Sequence
from itertools import chain

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


phi = (1 + np.sqrt(5)) / 2  # Golden ratio
c36 = np.cos(36 * np.pi / 180)
s36 = np.sin(36 * np.pi / 180)


class Tile:
    def __init__(self, center=(0, 0), scale=1.0, orientation=0.0):
        self.center = np.array(center)
        self.scale = scale
        self.orientation = orientation

        self._arc_radius = 1 / phi
        self._arc_line_thicknesses = (20, 10)
        self._arc_colors = ('#008000', '#c00000')


class Kite(Tile):
    def __init__(self, center=(0, 0), scale=1.0, orientation=0.0):
        super().__init__(center, scale, orientation)

        c = np.cos(self.orientation)
        s = np.sin(self.orientation)
        rot = np.array([[c, s], [-s, c]])
        self.__coords = np.array([[0, 0], [c36, s36], [1, 0], [c36, -s36]]) * self.scale @ rot + self.center

        self._arc_diam = np.array([self._arc_radius, 1 - self._arc_radius]) * 2 * self.scale

    def draw(self, ax):
        poly_args = dict(edgecolor='#000000', linewidth=3, antialiased=True, joinstyle='round')
        ax.add_patch(plt.Polygon(self.__coords, facecolor='#ffffc0', **poly_args))
        ax.add_patch(mpatches.Arc(self.__coords[0],
                                  height=self._arc_diam[0], width=self._arc_diam[0],
                                  theta1=-36 + self.orientation * 180 / np.pi, theta2=36 + self.orientation * 180 / np.pi,
                                  linewidth=self._arc_line_thicknesses[0], edgecolor=self._arc_colors[0]))
        ax.add_patch(mpatches.Arc(self.__coords[2],
                                  height=self._arc_diam[1], width=self._arc_diam[1],
                                  theta1=72 + 36 + self.orientation * 180 / np.pi, theta2=-72 - 36 + self.orientation * 180 / np.pi,
                                  linewidth=self._arc_line_thicknesses[1], edgecolor=self._arc_colors[1]))

    def divide(self) -> Sequence[Tile]:
        rel_scale = self.scale / phi
        rel_or = (36 + 72) * np.pi / 180
        return [Dart(scale=rel_scale, orientation=36*np.pi/180 + self.orientation),
                Kite(center=self.__coords[1], scale=rel_scale, orientation=self.orientation - rel_or),
                Kite(center=self.__coords[3], scale=rel_scale, orientation=self.orientation + rel_or)]


class Dart(Tile):
    def __init__(self, center=(0, 0), scale=1.0, orientation=0.0):
        super().__init__(center, scale, orientation)

        c = np.cos(self.orientation)
        s = np.sin(self.orientation)
        rot = np.array([[c, s], [-s, c]])
        self.__coords = np.array([[0, 0], [c36, s36], [2 * c36 - 1, 0], [c36, -s36]]) * self.scale @ rot + self.center

        self._arc_diam = np.array([1 - self._arc_radius, 1/phi - 1 + self._arc_radius]) * 2 * self.scale

    def divide(self) -> Sequence[Tile]:
        return [self]

    def draw(self, ax):
        poly_args = dict(edgecolor='#000000', linewidth=3, antialiased=True, joinstyle='round')
        ax.add_patch(plt.Polygon(self.__coords, facecolor='#c0ffff', **poly_args))
        ax.add_patch(mpatches.Arc(self.__coords[0],
                                  height=self._arc_diam[0], width=self._arc_diam[0],
                                  theta1=-36 + self.orientation * 180 / np.pi, theta2=36 + self.orientation * 180 / np.pi,
                                  linewidth=self._arc_line_thicknesses[0], edgecolor=self._arc_colors[0]))
        ax.add_patch(mpatches.Arc(self.__coords[2],
                                  height=self._arc_diam[1], width=self._arc_diam[1],
                                  theta1=72 + self.orientation * 180 / np.pi, theta2=-72 + self.orientation * 180 / np.pi,
                                  linewidth=self._arc_line_thicknesses[1], edgecolor=self._arc_colors[1]))


if __name__ == '__main__':
    # tiles = [Kite(orientation=36*np.pi/180)]
    tiles = [Kite([0, 0], orientation=_) for _ in np.arange(5) * 72 * np.pi / 180]
    tiles = list(chain(*[_.divide() for _ in tiles]))
    # tiles = list(chain(*[_.divide() for _ in tiles]))

    fig, ax = plt.subplots(1, 1)

    for t in tiles:
        t.draw(ax)

    plt.axis('equal')

    plt.show()
