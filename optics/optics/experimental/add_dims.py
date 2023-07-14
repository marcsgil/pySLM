import numpy as np


def add_dims(array_to_reshape, n, total):
    """
    Adds n singleton dimension at the right and makes sure that the total number of dimensions is total
    :param array_to_reshape: The input array.
    :param n: The number of dimensions to add at the right.
    :param total: The total number of dimensions required.
    :return: The reshaped array.
    """
    for idx in range(n):
        array_to_reshape = [array_to_reshape]
    array_to_reshape = np.array(array_to_reshape)

    array_to_reshape = add_trailing_dims(array_to_reshape, total - array_to_reshape.ndim)

    return array_to_reshape


def add_trailing_dims(array_to_expand, n):
    """
    Adds n singleton dimension at the left.
    :param array_to_expand: The input array.
    :param n: The number of dimensions to add to the left.
    :return: The reshaped array.
    """
    array_to_expand = np.array(array_to_expand)
    for idx in range(n):
        array_to_expand = np.expand_dims(array_to_expand, -1)

    return array_to_expand
