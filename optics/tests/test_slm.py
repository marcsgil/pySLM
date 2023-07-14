import unittest

import numpy.testing as npt
import numpy as np

from optics.instruments.slm import SLM, DisplaySLM, PhaseSLM, DualHeadSLM


class TestPhaseSLM(unittest.TestCase):
    def setUp(self):
        self.slm = PhaseSLM(shape=(8, 9))
        self.positive_pixels = np.mod(self.slm.grid[0] + self.slm.grid[1], 2) > 0
        self.negative_pixels = np.logical_not(self.positive_pixels)

    def test_init(self):
        npt.assert_array_equal(self.slm.shape, np.array([8, 9]))
        npt.assert_array_equal(self.slm.deflection_frequency, np.array([0, 0, 0]))
        self.slm.deflection_frequency = (1/4, 0)
        npt.assert_array_equal(self.slm.deflection_frequency, np.array([1/4, 0, 0]))
        npt.assert_equal(self.slm.two_pi_equivalent, 1.0)
        self.slm.two_pi_equivalent = 0.5
        npt.assert_equal(self.slm.two_pi_equivalent, 0.5)
        self.slm.two_pi_equivalent = 1.5
        npt.assert_equal(self.slm.two_pi_equivalent, 1.5)
        npt.assert_equal(self.slm.grid[0], np.arange(-4, 4)[:, np.newaxis])
        npt.assert_equal(self.slm.grid[1], np.arange(-4, 5)[np.newaxis, :])

    def test_modulate_0(self):
        complex_field = np.ones(self.slm.shape, dtype=complex)
        image_on_slm = np.ones([*self.slm.shape, 3], dtype=np.uint8)

        def pre_callback(s: SLM):
            complex_field.ravel()[:] = s.complex_field.ravel()

        def post_callback(s: DisplaySLM):
            image_on_slm.ravel()[:] = s.image_on_slm.ravel()

        self.slm.pre_modulation_callback = pre_callback
        self.slm.post_modulation_callback = post_callback

        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(complex_field, fld)
        npt.assert_array_equal(self.slm.complex_field, fld)
        npt.assert_array_equal(image_on_slm, slm_img)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        fld = np.zeros(self.slm.shape, dtype=complex)
        slm_img = np.zeros([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[self.positive_pixels, :2] = 128 + 64
        slm_img[self.negative_pixels, :2] = 128 - 64
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(complex_field, fld)
        npt.assert_array_equal(self.slm.complex_field, fld)
        npt.assert_array_equal(image_on_slm, slm_img)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        fld = np.ones(self.slm.shape, dtype=complex)
        fld[int(fld.shape[0]/2) + np.arange(4), int(fld.shape[1]/2)] = np.array([1j, 1j, -1, -1])
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[:, :, 2] = 255
        slm_img[int(fld.shape[0]/2) + np.arange(4), int(fld.shape[1]/2), :2] += \
            np.array([64, 64, 128, 128], dtype=np.uint8)[:, np.newaxis]
        self.slm.modulate(fld)
        npt.assert_array_equal(complex_field, fld)
        npt.assert_array_equal(self.slm.complex_field, fld)
        npt.assert_array_equal(image_on_slm, slm_img)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        fld = -np.ones(self.slm.shape, dtype=complex)
        slm_img = 0 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(complex_field, fld)
        npt.assert_array_equal(self.slm.complex_field, fld)
        npt.assert_array_equal(image_on_slm, slm_img)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        fld = 0.5 * np.ones(self.slm.shape, dtype=complex)
        slm_img = np.zeros([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[self.positive_pixels, :2] = 128 + round(256 / 6)
        slm_img[self.negative_pixels, :2] = 128 - round(256 / 6)
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(complex_field, fld)
        npt.assert_array_equal(self.slm.complex_field, fld)
        npt.assert_array_equal(image_on_slm, slm_img)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

    def test_deflection(self):
        self.slm.deflection_frequency = (1/2, 0)
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[1::2, :, :2] += 128
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.deflection_frequency = (0, 1/2)
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[:, 1::2, :2] += 128
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.deflection_frequency = (1/2, 1/2)
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.zeros([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[np.mod(self.slm.grid[0]+self.slm.grid[1], 2) <= 0, :2] += 128
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm[:, -1, 0], slm_img[:, -1, 0], "Rounding error?")
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.deflection_frequency = (1/4, 0)
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[1::4, :, :2] += 64
        slm_img[2::4, :, :2] += 128
        slm_img[3::4, :, :2] -= 64
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

    def test_modulation_1(self):
        self.slm.deflection_frequency = (1/2, 0)
        fld = 1j * np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[1::2, :, :2] += 128
        slm_img[:, :, :2] += 64
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.deflection_frequency = (1/4, 0)
        fld = 0.000001 * np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[1::4, :, :2] += 64
        slm_img[2::4, :, :2] += 128
        slm_img[3::4, :, :2] -= 64
        slm_img[self.positive_pixels, :2] += 64
        slm_img[self.negative_pixels, :2] -= 64
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm[:4, :4, 0], slm_img[:4, :4, 0], "Rounding error?")
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.deflection_frequency = (1/4, 0)
        fld = 0.0 * np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[1::4, :, :2] += 64
        slm_img[2::4, :, :2] += 128
        slm_img[3::4, :, :2] -= 64
        slm_img[self.positive_pixels, :2] += 64
        slm_img[self.negative_pixels, :2] -= 64
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img, "Phase confused for 0 input.")

    def test_modulation_two_pi_equivalent(self):
        self.slm.deflection_frequency = (1/4, 0)
        self.slm.two_pi_equivalent = 0.50
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = np.full([*self.slm.shape, 3], 0.5 * (self.slm.two_pi_equivalent * 256 + 1), dtype=np.uint8)
        slm_img[1::4, :, :2] += 32
        slm_img[2::4, :, :2] -= 64
        slm_img[3::4, :, :2] -= 32
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.two_pi_equivalent = 2.00
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = np.full([*self.slm.shape, 3], 0.5 * (1 * 256 + 1), dtype=np.uint8)
        slm_img[1::4, :, :2] = 255
        slm_img[2::4, :, :2] = 0
        slm_img[3::4, :, :2] = 0
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)


class TestDualHeadSLM(unittest.TestCase):
    def setUp(self):
        self.slm = DualHeadSLM(shape=(8, 9))

    def test_init(self):
        npt.assert_array_equal(self.slm.shape, np.array([8, 9]))
        npt.assert_array_equal(self.slm.deflection_frequency, np.array([0, 0, 0]))
        self.slm.deflection_frequency = (1/4, 0)
        npt.assert_array_equal(self.slm.deflection_frequency, np.array([1/4, 0, 0]))
        npt.assert_equal(self.slm.two_pi_equivalent, 1.0)
        self.slm.two_pi_equivalent = 0.5
        npt.assert_equal(self.slm.two_pi_equivalent, 0.5)
        self.slm.two_pi_equivalent = 1.5
        npt.assert_equal(self.slm.two_pi_equivalent, 1.5)
        npt.assert_equal(self.slm.grid[0], np.arange(-4, 4)[:, np.newaxis])
        npt.assert_equal(self.slm.grid[1], np.arange(-4, 5)[np.newaxis, :])

    def test_modulate_0(self):
        complex_field = np.ones(self.slm.shape, dtype=complex)
        image_on_slm = np.ones([*self.slm.shape, 3], dtype=np.uint8)

        def pre_callback(s: SLM):
            complex_field.ravel()[:] = s.complex_field.ravel()

        def post_callback(s: DisplaySLM):
            image_on_slm.ravel()[:] = s.image_on_slm.ravel()

        self.slm.pre_modulation_callback = pre_callback
        self.slm.post_modulation_callback = post_callback

        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(complex_field, fld)
        npt.assert_array_equal(self.slm.complex_field, fld)
        npt.assert_array_equal(image_on_slm, slm_img)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        fld = np.zeros(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[:, :, 2] = 0
        self.slm.modulate(fld)
        npt.assert_array_equal(complex_field, fld)
        npt.assert_array_equal(self.slm.complex_field, fld)
        npt.assert_array_equal(image_on_slm, slm_img)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        fld = np.ones(self.slm.shape, dtype=complex)
        fld[int(fld.shape[0]/2) + np.arange(4), int(fld.shape[1]/2)] = np.array([1j, 1j, -1, -1])
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[:, :, 2] = 255
        slm_img[int(fld.shape[0]/2) + np.arange(4), int(fld.shape[1]/2), :2] += \
            np.array([64, 64, 128, 128], dtype=np.uint8)[:, np.newaxis]
        self.slm.modulate(fld)
        npt.assert_array_equal(complex_field, fld)
        npt.assert_array_equal(self.slm.complex_field, fld)
        npt.assert_array_equal(image_on_slm, slm_img)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        fld = -np.ones(self.slm.shape, dtype=complex)
        slm_img = 0 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(complex_field, fld)
        npt.assert_array_equal(self.slm.complex_field, fld)
        npt.assert_array_equal(image_on_slm, slm_img)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        fld = 0.499 * np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[:, :, 2] = 127
        self.slm.modulate(fld)
        npt.assert_array_equal(complex_field, fld)
        npt.assert_array_equal(self.slm.complex_field, fld)
        npt.assert_array_equal(image_on_slm, slm_img)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        fld = 0.501 * np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[:, :, 2] = 128
        self.slm.modulate(fld)
        npt.assert_array_equal(complex_field, fld)
        npt.assert_array_equal(self.slm.complex_field, fld)
        npt.assert_array_equal(image_on_slm, slm_img)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

    def test_deflection(self):
        self.slm.deflection_frequency = (1/2, 0)
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[1::2, :, :2] += 128
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.deflection_frequency = (0, 1/2)
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[:, 1::2, :2] += 128
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.deflection_frequency = (1/2, 1/2)
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.zeros([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[np.mod(self.slm.grid[0]+self.slm.grid[1], 2) <= 0, :2] += 128
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm[:, -1, 0], slm_img[:, -1, 0], "Rounding error?")
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.deflection_frequency = (1/4, 0)
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[1::4, :, :2] += 64
        slm_img[2::4, :, :2] += 128
        slm_img[3::4, :, :2] -= 64
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

    def test_modulation_1(self):
        self.slm.deflection_frequency = (1/2, 0)
        fld = 1j * np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[1::2, :, :2] += 128
        slm_img[:, :, :2] += 64
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.deflection_frequency = (1/4, 0)
        fld = 0.000001 * np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[1::4, :, :2] += 64
        slm_img[2::4, :, :2] += 128
        slm_img[3::4, :, :2] -= 64
        slm_img[:, :, 2] = 0
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm[:, 1, 0], slm_img[:, 1, 0])
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.deflection_frequency = (1/4, 0)
        fld = 0.0 * np.ones(self.slm.shape, dtype=complex)
        slm_img = 128 * np.ones([*self.slm.shape, 3], dtype=np.uint8)
        slm_img[1::4, :, :2] += 64
        slm_img[2::4, :, :2] += 128
        slm_img[3::4, :, :2] -= 64
        slm_img[:, :, 2] = 0
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img, "Phase confused for 0 input.")

    def test_modulation_two_pi_equivalent(self):
        self.slm.deflection_frequency = (1/4, 0)
        self.slm.two_pi_equivalent = 0.50
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = np.full([*self.slm.shape, 3], 0.5 * (self.slm.two_pi_equivalent * 256 + 1), dtype=np.uint8)
        slm_img[1::4, :, :2] += 32
        slm_img[2::4, :, :2] -= 64
        slm_img[3::4, :, :2] -= 32
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)

        self.slm.two_pi_equivalent = 2.00
        fld = np.ones(self.slm.shape, dtype=complex)
        slm_img = np.full([*self.slm.shape, 3], 0.5 * (1 * 256 + 1), dtype=np.uint8)
        slm_img[1::4, :, :2] = 255
        slm_img[2::4, :, :2] = 0
        slm_img[3::4, :, :2] = 0
        slm_img[:, :, 2] = 255
        self.slm.modulate(fld)
        npt.assert_array_equal(self.slm.image_on_slm, slm_img)
