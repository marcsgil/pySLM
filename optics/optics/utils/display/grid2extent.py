import numpy as np
import logging

from optics.utils.ft import Grid

log = logging.getLogger(__name__)


def grid2extent(*args, origin_lower: bool=False):
    """
    Utility function to determine extent values for imshow
    :param args: A Grid object or monotonically increasing ranges, one per dimension (vertical, horizontal)
    :param origin_lower: (default: False) Set this to True when imshow has the origin set to 'lower' to have the
    vertical axis increasing upwards.
    :return: An nd-array with 4 numbers indicating the extent of the displayed data.
    """
    if isinstance(args[0], Grid):
        ranges = [_ for _ in args[0]]
    else:
        ranges = args
    if len(ranges) > 2:
        log.warning(f"Only using the last two axes in grid2extent(...)!")
    extent = []
    for idx, rng in enumerate(ranges[:-3:-1]):
        rng = np.asarray(rng).ravel()
        step = (rng[-1] - rng[0]) / (rng.size - 1)
        first, last = rng[0] - 0.5 * step, rng[-1] + 0.5 * step
        if not origin_lower and idx == 1:
            first, last = last, first
        extent.append(first)
        extent.append(last)

    return np.asarray(extent)
