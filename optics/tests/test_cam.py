import unittest
import numpy.testing as npt
import numpy as np

from optics.instruments.cam import Cam
from optics.instruments.cam.simulated_cam import SimulatedCam
from optics.utils import Roi


class TestSimulatedCam(unittest.TestCase):
    def setUp(self):
        pass

    def test_roi(self):
        cam = SimulatedCam(shape=[256, 512])
        npt.assert_array_equal(cam.shape, np.array([256, 512]))
        npt.assert_array_equal(cam.roi.shape, np.array([256, 512]))
        img = cam.acquire()
        npt.assert_array_equal(img.shape[:2], np.array([256, 512]))
        cam.roi = Roi(top_left=[20, 30], shape=[44, 33])
        npt.assert_array_equal(cam.roi.top_left, np.array([20, 30]))
        npt.assert_array_equal(cam.roi.shape, np.array([44, 33]))
        npt.assert_array_equal(cam.shape, np.array([256, 512]))
        img = cam.acquire()
        npt.assert_array_equal(img.shape[:2], np.array([44, 33]))
        cam.roi = None
        npt.assert_array_equal(cam.roi.top_left, np.array([0, 0]))
        npt.assert_array_equal(cam.shape, np.array([256, 512]))
        npt.assert_array_equal(cam.roi.shape, np.array([256, 512]))
        img = cam.acquire()
        npt.assert_array_equal(img.shape[:2], np.array([256, 512]))
        cam.roi = Roi(top_left=[-10, -5], shape=[44, 33])
        npt.assert_array_equal(cam.roi.top_left, np.array([-10, -5]))
        npt.assert_array_equal(cam.roi.shape, np.array([44, 33]))
        npt.assert_array_equal(cam.shape, np.array([256, 512]))
        img = cam.acquire()
        npt.assert_array_equal(img.shape[:2], np.array([44, 33]))

    def test_dtype(self):
        cam = SimulatedCam()
        npt.assert_equal(cam.normalize, True)
        img = cam.acquire()
        npt.assert_equal(img.dtype == float, True)
        cam.normalize = False
        img = cam.acquire()
        npt.assert_equal(cam.normalize, False)
        npt.assert_equal(img.dtype == np.uint8, True)
        cam.normalize = True
        img = cam.acquire()
        npt.assert_equal(img.dtype == float, True)

    def test_callback_full_field(self):
        def simulate_image_float(c: Cam):
            sim_img = np.zeros(shape=c.shape, dtype=float)
            sim_img[5, 5] = 1.0
            return sim_img

        cam_float = SimulatedCam(get_frame_callback=simulate_image_float)
        img = cam_float.acquire()
        npt.assert_equal(img.dtype == float, True)
        npt.assert_equal(img[5, 5], 1.0)
        cam_float.normalize = False
        img = cam_float.acquire()
        npt.assert_equal(img.dtype == np.uint8, True)
        npt.assert_equal(img[5, 5], 255)

        def simulate_image_uint8(c: Cam):
            sim_img = np.zeros(shape=c.shape, dtype=np.uint8)
            sim_img[5, 5] = 255
            return sim_img

        cam_uint8 = SimulatedCam(get_frame_callback=simulate_image_uint8)
        img = cam_uint8.acquire()
        npt.assert_equal(img.dtype == float, True)
        npt.assert_equal(img[5, 5], 1.0)
        cam_uint8.normalize = False
        img = cam_uint8.acquire()
        npt.assert_equal(img.dtype == np.uint8, True)
        npt.assert_equal(img[5, 5], 255)

    def test_callback_part_field(self):
        def simulate_image(c: Cam):
            shape = (30, 40)
            sim_img = np.zeros(shape=shape, dtype=float)
            sim_img[int(shape[0]/2), int(shape[1]/2)] = 1.0
            return sim_img

        cam = SimulatedCam(shape=(33, 44), get_frame_callback=simulate_image)
        img = cam.acquire()
        ref_img = np.zeros(shape=cam.shape, dtype=float)
        ref_img[int(33/2), int(44/2)] = 1.0
        npt.assert_array_equal(img, ref_img)

        def simulate_image2(c: Cam):
            shape = (3, 4)
            sim_img = np.zeros(shape=shape, dtype=float)
            sim_img[int(shape[0]/2), int(shape[1]/2)] = 1.0
            return sim_img

        cam = SimulatedCam(shape=(33, 44), get_frame_callback=simulate_image2)
        img = cam.acquire()
        ref_img = np.zeros(shape=cam.shape, dtype=float)
        ref_img[int(33/2), int(44/2)] = 1.0
        npt.assert_array_equal(img, ref_img)

    def test_callback_roi(self):
        def simulate_image(c: Cam):
            sim_img = np.zeros(shape=c.shape, dtype=float)
            sim_img[int(c.shape[0]/2), int(c.shape[1]/2)] = 1.0

            sim_img = sim_img[
                np.clip(c.roi.grid[0], 0, c.shape[0] - 1).astype(dtype=int),
                np.clip(c.roi.grid[1], 0, c.shape[1] - 1).astype(dtype=int), ...]

            return sim_img

        cam = SimulatedCam(shape=(33, 44), get_frame_callback=simulate_image)
        img = cam.acquire()
        ref_img = np.zeros(shape=cam.shape, dtype=float)
        ref_img[int(33/2), int(44/2)] = 1.0
        npt.assert_array_equal(img, ref_img)

        cam.roi = Roi(shape=(6, 7), center=(int(33/2), int(44/2)))
        raw_img = simulate_image(cam)
        npt.assert_array_equal(raw_img.shape, cam.roi.shape)
        img = cam.acquire()
        ref_img = np.zeros(shape=cam.roi.shape, dtype=float)
        ref_img[int(6/2), int(7/2)] = 1.0
        npt.assert_array_equal(img, ref_img)

        cam.roi = Roi(shape=(1, 7), center=(16, 20))
        raw_img = simulate_image(cam)
        npt.assert_array_equal(raw_img.shape, cam.roi.shape)
        img = cam.acquire()
        ref_img = np.zeros(shape=cam.roi.shape, dtype=float)
        ref_img[0, 5] = 1.0
        npt.assert_array_equal(img, ref_img)

    def test_saturated(self):
        def simulate_image_float(c: Cam):
            sim_img = np.zeros(shape=c.shape, dtype=float)
            sim_img[5, 5] = 1.1
            return sim_img

        cam = SimulatedCam(get_frame_callback=simulate_image_float)
        img = cam.acquire()
        ref_img = np.zeros(shape=cam.shape, dtype=float)
        ref_img[5, 5] = 1.0
        npt.assert_equal(img.dtype == float, True)
        npt.assert_array_equal(img, ref_img)

        cam.normalize = False
        img = cam.acquire()
        ref_img = np.zeros(shape=cam.shape, dtype=np.uint8)
        ref_img[5, 5] = 255
        npt.assert_equal(img.dtype == np.uint8, True)
        npt.assert_array_equal(img, ref_img)

    def test_non_saturated(self):
        def simulate_image_float(c: Cam):
            sim_img = np.zeros(shape=c.shape, dtype=float)
            sim_img[2, 2] = 0.5
            return sim_img

        cam = SimulatedCam(shape=(3, 3), get_frame_callback=simulate_image_float)

        cam.normalize = False
        img = cam.acquire()
        ref_img = np.zeros(shape=cam.shape, dtype=np.uint8)
        ref_img[2, 2] = 128
        npt.assert_equal(img.dtype == np.uint8, True)
        npt.assert_array_equal(img, ref_img)

        cam.normalize = True
        img = cam.acquire()
        ref_img = np.zeros(shape=cam.shape, dtype=float)
        ref_img[2, 2] = 128.0 / 255
        npt.assert_equal(img.dtype == float, True)
        npt.assert_array_equal(img, ref_img)
