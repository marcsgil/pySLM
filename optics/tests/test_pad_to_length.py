import unittest
import numpy.testing as npt

from optics.utils.array import pad_to_length

import numpy as np


class TestPadToLength(unittest.TestCase):
    def setUp(self):
        pass

    def test_pad_to_length(self):
        npt.assert_array_equal(pad_to_length(np.array([1, 2, 3, 4]), 5), np.array([1, 2, 3, 4, 0]),
                               err_msg="Could not extend array by one.")
        npt.assert_array_equal(pad_to_length(np.array([1, 2, 3, 4]), 4), np.array([1, 2, 3, 4]),
                               err_msg="Could not extend array by zero.")
        npt.assert_array_equal(pad_to_length(np.array([1, 2, 3, 4]), 5, -1), np.array([1, 2, 3, 4, -1]),
                               err_msg="Could not extend array by one with value -1.")
        npt.assert_array_equal(pad_to_length(np.array([1, 2, 3, 4]), 6), np.array([1, 2, 3, 4, 0, 0]),
                               err_msg="Could not extend array by two.")
        npt.assert_array_equal(pad_to_length([1, 2, 3, 4], 6), np.array([1, 2, 3, 4, 0, 0]),
                               err_msg="Could not extend list by two.")
        npt.assert_array_equal(pad_to_length((1, 2, 3, 4), 6), np.array([1, 2, 3, 4, 0, 0]),
                               err_msg="Could not extend tuple by two.")
        npt.assert_array_equal(pad_to_length(10, 4), np.array([10, 0, 0, 0]),
                               err_msg="Could not extend multidimensional by 3.")

    # def test_pad_to_length_along_axis(self):
    #     # Deprecated functionality
    #     vec0 = np.array([1, 2, 3, 4]).reshape(4, 1, 1)
    #     vec1 = np.array([1, 2, 3, 4]).reshape(1, 4, 1)
    #     vec2 = np.array([1, 2, 3, 4]).reshape(1, 1, 4)
    #
    #     npt.assert_array_equal(pad_to_length(vec0, 6), np.array([1, 2, 3, 4, 0, 0]).reshape(6, 1, 1),
    #                            err_msg="Could not extend vector by 2 along axis 0.")
    #     npt.assert_array_equal(pad_to_length(vec1, 6), np.array([1, 2, 3, 4, 0, 0]).reshape(1, 6, 1),
    #                            err_msg="Could not extend vector by 2 along axis 1.")
    #     npt.assert_array_equal(pad_to_length(vec2, 6), np.array([1, 2, 3, 4, 0, 0]).reshape(1, 1, 6),
    #                            err_msg="Could not extend vector by 2 along axis 2.")


if __name__ == '__main__':
    unittest.main()
