import numpy as np
from typing import Union, Sequence
import logging

from optics.utils.ft import Grid

log = logging.getLogger(__name__)


class Peak:
    """
    A class representing a peak in values. It has a peak position and its width can be determined at various heights
    """
    def __init__(self, y: Union[Sequence, np.ndarray], x: Union[Sequence, np.ndarray, Grid]=None,
                 fraction: float=0.50):
        """
        Constructor for the Peak class.
        :param y: A vector with the values.
        :param x: A vector with the coordinates of the values. If unspecified, the integers indexes of y are used.
        :param fraction: The default fraction (optional). If not provided, this is 50%.
        """
        y = np.array(y)
        if x is None:
            x = Grid(y.shape[-1], first=0)
        elif isinstance(x, Grid):
            x = x[0] if x.multidimensional else list(x)

        # If x is not sorted, sort both x and y
        x_argsort = np.argsort(x)
        self.__y = y[x_argsort]
        self.__x = x[x_argsort]

        self.__fraction = fraction

        # Find largest (real) value
        self.__index = np.argmax(y)
        self.__max = self.__y[self.__index]
        # find last equal to peak value, counting from self.__index
        center_width = np.argmax(self.__y[self.__index + 1:] < self.__max)
        self.position = self.__x[self.__index] + (center_width / 2)

    @property
    def max(self) -> float:
        """
        The peak value.
        """
        return self.__max

    @max.setter
    def max(self, new_max: float):
        """
        Scale the values so that the new peak value equals new_max
        """
        self.__y *= new_max / self.__max
        self.__max = new_max

    def width(self, fraction: float=None, value: float=None) -> float:
        """
        Return the width of the peak at a fraction of the maximum or, when specified, a specific value.
        Linear interpolation is used between positions as necessary.

        :param fraction: The cutoff fraction, defaults to the class's default.
        :param value: The optional cutoff value, replaced the fraction.
        :return: The width as a floating point number.
        """
        if value is None:
            if fraction is None:
                fraction = self.__fraction
            value = fraction * self.max
        sign = -1 if value < 0.0 else 1

        # find last above cutoff
        left_idx = self.__index - np.argmax(self.__y[self.__index - 1::-1] * sign <= value * sign)
        right_idx = self.__index + np.argmax(self.__y[self.__index + 1:] * sign <= value * sign)
        # log.info(f"center_i={center_i}, left_idx={left_idx}, right_idx={right_idx}, cutoff={cutoff}")
        if 0 <= left_idx - 1:
            # Interpolate linearly
            left = self.__x[left_idx - 1] + (self.__x[left_idx] - self.__x[left_idx - 1]) \
                   * (value - self.__y[left_idx - 1]) / (self.__y[left_idx] - self.__y[left_idx - 1])
        else:
            left = -np.inf
        if right_idx + 1 < len(self.__x):
            right = self.__x[right_idx + 1] + (self.__x[right_idx] - self.__x[right_idx + 1]) \
                    * (value - self.__y[right_idx + 1]) / (self.__y[right_idx] - self.__y[right_idx + 1])
        else:
            right = np.inf

        return right - left


def fwhm(y: Union[Sequence, np.ndarray], x: Union[Sequence, np.ndarray, Grid]=None,
         fraction: float=0.5, value: float=None) -> float:
    """
    Determines the full-width-at-half-maximum (FWHM) of y(x).
    If x is not specified, the integer indices of y are used.
    When y is complex, its real part is used. If the maximum value is negative, y is inverted.

    :param y: The function values.
    :param x: The sample points. (optional)
    :param fraction: The cut-off fraction, default 50%. (optional)
    :param value: The optional cut-off value. (optional)
    :return: The full-width-at-half-maximum in the units specified by x.
    """
    return Peak(y=y, x=x).width(fraction=fraction, value=value)


def fwtm(y: Union[Sequence, np.ndarray], x: Union[Sequence, np.ndarray, Grid]=None,
         fraction: float=0.1, value: float=None) -> float:
    """
    Determines the full-width-at-one-tenth-maximum (FWTM) of y(x).
    If x is not specified, the integer indices of y are used.
    When y is complex, its real part is used. If the maximum value is negative, y is inverted.

    :param y: The function values.
    :param x: The sample points. (optional)
    :param fraction: The cut-off fraction, default 10%. (optional)
    :param value: The optional cut-off value. (optional)
    :return: The full-width-at-one-tenth-maximum in the units specified by x.
    """
    return Peak(y=y, x=x).width(fraction=fraction, value=value)
