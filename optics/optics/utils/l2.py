from typing import Union, Sequence, Optional
from numbers import Number
import numpy as np

array_like = Union[np.ndarray, Sequence, Number]


def mse(variable: array_like, reference: Optional[array_like] = None) -> float:
    """
    Calculate the mean-square error (:math:`l^2`-norm) of the variable, optionally, with respect to that of the reference.

    :param variable: The variable to compute the mean-square error of.
    :param reference: The reference to compare the mean square error to.
    :return: The (relative) mean square error.
    """
    return rms(variable, reference) ** 2


def rms(variable: array_like, reference: Optional[array_like] = None) -> float:
    """
    Calculate the root-mean-square error (:math:`l^2`-norm) of the variable, optionally, with respect to that of the reference.
    This uses the `l2-norm <https://mathworld.wolfram.com/L2-Norm.html>`_

    :param variable: The variable to compute the root-mean-square error of.
    :param reference: The reference to compare the norm to.
    :return: The (relative) root-mean square error.
    """
    if reference is None:
        return np.linalg.norm(variable)
    else:
        return np.linalg.norm(variable - reference) / np.linalg.norm(reference)

