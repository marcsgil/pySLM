import numpy as np
import scipy.interpolate as interp

from optics.utils.ft.grid import Grid


def resample_to_shape(data: np.ndarray, new_shape):
    """
    Resamples n-dimensional data to have a shape equal to new_shape
    :param data: An nd-array that is to be resampled. This can be complex.
    :param new_shape: The new shape as a tuple or vector
    :return: An nd-array of shape new_shape.
    """
    in_shape = data.shape
    in_ranges = Grid(in_shape, flat=True)
    out_ranges = Grid(new_shape, np.array(in_shape).astype(np.float) / np.array(new_shape), flat=True)

    interpolator = interp.RegularGridInterpolator(in_ranges, data, method='nearest', bounds_error=False, fill_value=0.0)
    data = interpolator(np.ix_(*out_ranges))

    return data


if __name__ == '__main__':
    a = np.ones((2, 3, 4))
    print(a.shape)
    a = resample_to_shape(a, (6, 7, 8))
    print(a.shape)
