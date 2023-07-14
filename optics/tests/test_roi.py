import unittest
import numpy.testing as npt

from optics.utils import Roi

import numpy as np


class TestRoi(unittest.TestCase):
    def setUp(self):
        self.r0 = Roi(center=(0, 0), shape=(10, 15), dtype=np.int32)
        self.r1 = Roi((10, 10, 100, 150))
        self.r2 = Roi((10, 11, 10, 15))
        self.r3 = Roi(top_left=(10, 10), bottom_right=(20, 30))

    def test_incomplete(self):
        r = Roi(top_left=(2, 3))
        npt.assert_array_equal(r.top_left, np.array((2, 3)))
        npt.assert_array_equal(r.shape, np.array((0, 0)))

        r = Roi(top=2, height=10)
        npt.assert_array_equal(r.top_left, np.array((2, 0)))
        npt.assert_array_equal(r.shape, np.array((10, 0)))

        r = Roi(center=(0, 0), height=10, width=20)
        npt.assert_array_equal(r.top_left, np.array((-5, -10)))
        npt.assert_array_equal(r.shape, np.array((10, 20)))

        r = Roi(top=-5, left=-10, height=10, width=20)
        npt.assert_array_equal(r.top_left, np.array((-5, -10)))
        npt.assert_array_equal(r.shape, np.array((10, 20)))

    def test_dtype(self):
        r = Roi([1.1, 2.2, 3.3, 4.4])
        r.dtype = int
        npt.assert_equal(r.dtype, int, 'Could not recast to int.')
        npt.assert_array_equal(r.top_left, np.array([1, 2]), 'Conversion to int not correct.')
        npt.assert_array_equal(r.shape, np.array([3, 4]), 'Conversion to int not correct.')

    def test_dict(self):
        r = Roi()
        r.__dict__ = dict(top_left=[-5, -10], shape=[10, 20])
        npt.assert_array_equal(r.top_left, np.array((-5, -10)))
        npt.assert_array_equal(r.shape, np.array((10, 20)))

    def test_init_individual_int(self):
        r = Roi(10, 20, 30, 40)
        npt.assert_almost_equal(r.top_left, np.array([10, 20]), err_msg='top_left not correct')
        npt.assert_almost_equal(r.shape, np.array([30, 40]), err_msg='shape not correct')
        npt.assert_equal(r.dtype, int, err_msg='dtype not correct')

    def test_init_individual_float(self):
        r = Roi(10.1, 20.2, 30.3, 40.4)
        npt.assert_almost_equal(r.top_left, np.array([10.1, 20.2]), err_msg='top_left not correct')
        npt.assert_almost_equal(r.shape, np.array([30.3, 40.4]), err_msg='shape not correct')
        npt.assert_equal(r.dtype, float, err_msg='dtype not correct')

    def test_init_tuple_int(self):
        r = Roi((10, 20, 30, 40))
        npt.assert_almost_equal(r.top_left, np.array([10, 20]), err_msg='top_left not correct')
        npt.assert_almost_equal(r.shape, np.array([30, 40]), err_msg='shape not correct')
        npt.assert_equal(r.dtype, int, err_msg='dtype not correct')

    def test_init_tuple_float(self):
        r = Roi((10.1, 20.2, 30.3, 40.4))
        npt.assert_almost_equal(r.top_left, np.array([10.1, 20.2]), err_msg='top_left not correct')
        npt.assert_almost_equal(r.shape, np.array([30.3, 40.4]), err_msg='shape not correct')
        npt.assert_equal(r.dtype, float, err_msg='dtype not correct')

    def test_init_center_int(self):
        r = Roi(center=(25, 40), shape=(30, 40))
        npt.assert_almost_equal(r.top_left, np.array([10, 20]), err_msg='top_left not correct')
        npt.assert_almost_equal(r.shape, np.array([30, 40]), err_msg='shape not correct')
        npt.assert_equal(r.dtype, int, err_msg='dtype not correct')

    def test_init_center_float(self):
        r = Roi(center=(25.1, 40.2), shape=(30, 40))
        npt.assert_equal(r.dtype, float, err_msg='dtype not correct')
        npt.assert_almost_equal(r.top_left, np.array([10.1, 20.2]), err_msg='top_left not correct')
        npt.assert_almost_equal(r.shape, np.array([30, 40]), err_msg='shape not correct')

    def test_clip(self):
        result = self.r2 % self.r1
        npt.assert_almost_equal(result.top_left, np.array([10, 11]), err_msg='clip top_left not correct')
        npt.assert_almost_equal(result.shape, np.array([10, 15]), err_msg='clip shape not correct')
        
    def test_relative(self):
        result = self.r2 / self.r1
        npt.assert_almost_equal(result.top_left, np.array([0, 1]), err_msg='relative top_left not correct')
        npt.assert_almost_equal(result.shape, np.array([10, 15]), err_msg='relative shape not correct')
        
    def test_absolute(self):
        result = self.r2 * self.r1
        npt.assert_almost_equal(result.top_left, np.array([20, 21]), err_msg='absolute top_left not correct')
        npt.assert_almost_equal(result.shape, np.array([10, 15]), err_msg='absolute shape not correct')

    def test_corner_definition(self):
        result = Roi(top_left=(10, 10), bottom_right=(20, 30))
        npt.assert_almost_equal(result.top_left, np.array([10, 10]), err_msg='corner definition top_left not correct')
        npt.assert_almost_equal(result.shape, np.array([10, 20]), err_msg='corner definition shape not correct')

    def test_center_definition(self):
        r = Roi(shape=(1, 2), center=(3, 4), dtype=int)
        npt.assert_almost_equal(r.center, np.array([3, 4]), err_msg='Center position incorrect')
        npt.assert_almost_equal(r.shape, np.array([1, 2]), err_msg='Center definition shape incorrect')
        r = Roi(shape=(2, 1), center=(3, 4), dtype=int)
        npt.assert_almost_equal(r.center, np.array([3, 4]), err_msg='Center position incorrect')
        npt.assert_almost_equal(r.shape, np.array([2, 1]), err_msg='Center definition shape incorrect')

        r = Roi(shape=(11, 12), center=(3, 4), dtype=int)
        npt.assert_almost_equal(r.center, np.array([3, 4]), err_msg='Center position incorrect')
        npt.assert_almost_equal(r.shape, np.array([11, 12]), err_msg='Center definition shape incorrect')
        r = Roi(shape=(12, 11), center=(3, 4), dtype=int)
        npt.assert_almost_equal(r.center, np.array([3, 4]), err_msg='Center position incorrect')
        npt.assert_almost_equal(r.shape, np.array([12, 11]), err_msg='Center definition shape incorrect')


if __name__ == '__main__':
    unittest.main()
