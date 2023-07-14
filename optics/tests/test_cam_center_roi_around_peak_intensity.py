import unittest
import numpy.testing as npt
import numpy as np

from optics.instruments.cam import Cam
from optics.instruments.cam.simulated_cam import SimulatedCam
from optics.utils import Roi


class TestCamCenterRoiAroundPeakIntensity(unittest.TestCase):
    def setUp(self):
        noise_level = 0.5

        def simulate_image(c: Cam, pos=None, brightness=1.0):
            if pos is None:
                pos = np.array(c.shape // 2, dtype=int)
            rng = np.random.RandomState(seed=1)
            sim_img = noise_level * rng.rand(*c.shape)
            sim_img[pos[0], pos[1]] = 1.0
            sim_img[(c.shape[0] // 2) + 25, (c.shape[1] // 2) + 36] = 0.9  # add second intense pixel, 90% of the first's intensity
            sim_img *= brightness  # saturate as requested
            if c.exposure_time is not None:
                sim_img *= c.exposure_time / 1e-3  # account for integration time adjustments
            return sim_img

        self.cam_center = SimulatedCam(get_frame_callback=simulate_image)
        self.cam_5_5 = SimulatedCam(get_frame_callback=lambda c: simulate_image(c, (5, 5)))
        self.cam_64_5 = SimulatedCam(get_frame_callback=lambda c: simulate_image(c, (64, 5)))

        self.cam_saturated = SimulatedCam(get_frame_callback=lambda c: simulate_image(c, (64, 5), brightness=25.0))

    def test_basic(self):
        self.cam_center.center_roi_around_peak_intensity()
        npt.assert_array_equal(self.cam_center.roi.center, np.array(self.cam_center.shape // 2, dtype=int),
                               "Not centered around peak intensity.")

        self.cam_center.roi = None
        self.cam_center.center_roi_around_peak_intensity(shape=(5, 5))
        npt.assert_array_equal(self.cam_center.roi.center, np.array(self.cam_center.shape // 2, dtype=int),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_center.roi.shape, np.array([5, 5]),
                               "Region of interest shape not correct.")

        self.cam_center.roi = None
        self.cam_center.center_roi_around_peak_intensity(shape=(4, 4))
        npt.assert_array_equal(self.cam_center.roi.center, np.array(self.cam_center.shape // 2, dtype=int),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_center.roi.shape, np.array([4, 4]),
                               "Region of interest shape not correct.")

        self.cam_center.roi = None
        self.cam_center.center_roi_around_peak_intensity(shape=(4, 15))
        npt.assert_array_equal(self.cam_center.roi.center, np.array(self.cam_center.shape // 2, dtype=int),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_center.roi.shape, np.array([4, 15]),
                               "Region of interest shape not correct.")

    def test_edge_cases(self):
        self.cam_5_5.center_roi_around_peak_intensity(shape=(3, 3))
        npt.assert_array_equal(self.cam_5_5.roi.center, np.array([5, 5]),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_5_5.roi.shape, np.array([3, 3]),
                               "Region of interest shape not correct.")

        self.cam_5_5.roi = None
        self.cam_5_5.center_roi_around_peak_intensity(shape=(3, 15))
        npt.assert_array_equal(self.cam_5_5.roi.center, np.array([5, 5]),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_5_5.roi.shape, np.array([3, 15]),
                               "Region of interest shape not correct.")

        self.cam_5_5.roi = None
        self.cam_5_5.center_roi_around_peak_intensity(shape=(15, 15))
        npt.assert_array_equal(self.cam_5_5.roi.center, np.array([5, 5]),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_5_5.roi.shape, np.array([15, 15]),
                               "Region of interest shape not correct.")

    def test_0(self):
        self.cam_5_5.center_roi_around_peak_intensity(shape=(0, 3))
        npt.assert_array_equal(self.cam_5_5.roi.center, np.array([5, 5]),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_5_5.roi.shape, np.array([0, 3]),
                               "Region of interest shape not correct.")

        self.cam_5_5.roi = None
        self.cam_5_5.center_roi_around_peak_intensity(shape=(3, 0))
        npt.assert_array_equal(self.cam_5_5.roi.center, np.array([5, 5]),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_5_5.roi.shape, np.array([3, 0]),
                               "Region of interest shape not correct.")

        self.cam_5_5.roi = None
        self.cam_5_5.center_roi_around_peak_intensity(shape=(0, 0))
        npt.assert_array_equal(self.cam_5_5.roi.center, np.array([5, 5]),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_5_5.roi.shape, np.array([0, 0]),
                               "Region of interest shape not correct.")

    def test_edge_cases_64_5(self):
        self.cam_64_5.center_roi_around_peak_intensity(shape=(3, 3))
        npt.assert_array_equal(self.cam_64_5.roi.center, np.array([64, 5]),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_64_5.roi.shape, np.array([3, 3]),
                               "Region of interest shape not correct.")

        self.cam_64_5.roi = None
        self.cam_64_5.center_roi_around_peak_intensity(shape=(3, 15))
        npt.assert_array_equal(self.cam_64_5.roi.center, np.array([64, 5]),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_64_5.roi.shape, np.array([3, 15]),
                               "Region of interest shape not correct.")

        self.cam_64_5.roi = None
        self.cam_64_5.center_roi_around_peak_intensity(shape=(15, 15))
        npt.assert_array_equal(self.cam_64_5.roi.center, np.array([64, 5]),
                               "Not centered around peak intensity.")
        npt.assert_array_equal(self.cam_64_5.roi.shape, np.array([15, 15]),
                               "Region of interest shape not correct.")

    def test_saturated(self):
        self.cam_saturated.center_roi_around_peak_intensity()
        npt.assert_array_equal(self.cam_saturated.roi.center, np.array((64, 5)),
                               "Not centered around peak intensity.")

    def test_second_peak(self):
        second_peak = (self.cam_5_5.shape // 2) + [25, 36]
        self.cam_5_5.roi = Roi(shape=(32, 32), center=second_peak)  # so that the second peak is the only one
        self.cam_5_5.center_roi_around_peak_intensity()
        npt.assert_array_equal(self.cam_5_5.roi.center, second_peak,
                               "Not centered around peak intensity.")

    def tearDown(self):
        self.cam_center.disconnect()
        self.cam_5_5.disconnect()
        self.cam_64_5.disconnect()


if __name__ == '__main__':
    unittest.main()
