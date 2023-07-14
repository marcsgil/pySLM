import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from typing import List, Sequence, Union, Optional

from optics.utils import ft
from projects.confocal_interference_contrast.complex_phase_image import Interferogram
from optics.utils.roi import Roi
from projects.confocal_interference_contrast.polarization_memory_effect import log
from optics.utils.display import complex2rgb

# from optics.instruments.cam.ids_peak_cam import IDSPeakCam

dtype = np.complex64

array_like = Union[Sequence, np.ndarray, int, float, complex]


class MemoryEffect:
    class Metric:
        """A subclass to report the coefficients."""
        def __init__(self, img_pair):
            self.norms = np.linalg.norm(img_pair, axis=(-2, -1), keepdims=True)
            self.difference = np.linalg.norm(img_pair[1] - img_pair[0])
            self.inner_product = np.sum(img_pair[0] * np.conjugate(img_pair[1])).real
            self.normalized_difference = self.difference / np.linalg.norm(self.norms)
            self.normalized_inner_product = self.inner_product / np.prod(self.norms)

        def __array__(self, dtype=None) -> np.ndarray:
            return np.asarray([self.difference, self.inner_product, self.normalized_difference, self.normalized_inner_product, *self.norms.ravel()], dtype=dtype)

        def __str__(self) -> str:
            return f'|A-B| = {self.normalized_difference:0.3f}, |〈A|B〉| = {self.normalized_inner_product:0.3f}'

    def __init__(self, images: array_like, fringe_periods: Optional[array_like] = None):
        """
        :param images: NORMALIZED (0-1) images set of dimensions: Optional(nb_images), y_image, x_image
        :param fringe_periods: An optional frequency-space registration to use to remove the interference fringes.
        If None specified, the first image in the sequence is used to detect it.
        """
        images = np.asarray(images)
        if np.ndim(images) == 2:
            images = images[np.newaxis, ...]
        self.__images = self.__split_images(images)  # nb_images * left-right * y_image * x_image

        if fringe_periods is None:
            first_interferogram_pair = [Interferogram(_) for _ in self.__images[0]]
            registration_pair = [_.registration for _ in first_interferogram_pair]
        else:
            fringe_periods = np.atleast_2d(fringe_periods)
            if fringe_periods.shape[0] < 2:
                fringe_periods = np.repeat(fringe_periods, axis=0, repeats=2)
            registration_pair = [ft.subpixel.Registration(shift=fp) for fp in fringe_periods]
        # log.info(f'Spatial frequencies: {registration_pair[0].shift} and {registration_pair[1].shift} cycles/px, amplitudes: {registration_pair[0].factor} and {registration_pair[1].factor}.')

        self.__interferogram_registration_pair = registration_pair

        self.__complex_images = None

    @property
    def interferogram_registration_pair(self):
        """The Fourier-space registration for the interference fringe frequency."""
        return self.__interferogram_registration_pair

    def __align_image_pair(self, image_pair):
        interferogram_pair = [Interferogram(img, registration=reg) for img, reg in zip(image_pair, self.interferogram_registration_pair)]
        complex_pupil_image_pair = np.asarray([_.__array__(keep_magnitude=True, keep_phase=True) for _ in interferogram_pair])

        # Correct the pupil position. TODO: This can probably be done faster
        abs_registration = ft.subpixel.register(np.abs(complex_pupil_image_pair[1]), np.abs(complex_pupil_image_pair[0]))
        # log.info(f'Shifting the images by {abs_registration.shift} px based on amplitude of complex pupils...')
        complex_pupil_image_pair[1] = ft.roll(complex_pupil_image_pair[1], -abs_registration.shift)

        # Minimize phase tilt and piston, but don't change the magnitude.
        interferogram_registration = ft.Reference(
            reference_data_ft=ft.ifftshift(complex_pupil_image_pair[0])).register(subject_ft=ft.ifftshift(complex_pupil_image_pair[1]), keep_magnitude=True)
        complex_pupil_image_pair[1] = ft.fftshift(interferogram_registration.image_ft)

        # complex_pupil_image_pair /= np.linalg.norm(complex_pupil_image_pair, axis=(-2, -1), keepdims=True)

        return complex_pupil_image_pair

    @property
    def complex_images(self) -> np.ndarray:
        if self.__complex_images is None:
            self.__complex_images = np.empty(self.__images.shape, dtype=dtype)
            for c_img, img_pair in zip(self.__complex_images, self.__images):  # tqdm(self.__images, desc='Calculating coefficients')):
                c_img[:] = self.__align_image_pair(img_pair)

        return self.__complex_images

    @property
    def metrics(self) -> List[Metric]:
        """The metrics, with a name"""
        return [self.Metric(img_pair) for img_pair in self.complex_images]

    @property
    def metric_array(self) -> np.ndarray:
        """The coefficients. Just get the numbers for each metric in an nd-array"""
        return np.array([_.__array__() for _ in self.metrics])

    @staticmethod
    def __split_images(images):
        images = np.array([images[..., :images.shape[-1] // 2], images[..., images.shape[-1] // 2:]], dtype=np.float32)

        images = np.swapaxes(images, 0, 1)  # nb_images X left-right X y_image X x_image

        return images


if __name__ == "__main__":
    #image_data = Image.open('test_data/hv_comparison.png')
    # cam = IDSPeakCam(serial=4104465739, gain=0, exposure_time=5e-3)
    # cam.roi = Roi(top=400, shape=(1700, 3840))
    print('a')

    # image_data = cam.acquire()
    # image_data = np.array(image_data)
    # memeffect = MemoryEffect(image_data)  # Can process multiple images at once for speed. Registration is then reused
    # print(memeffect.calculate_coefficients())
    # Coefficient of a detached image
    # img0, img1 = memeffect.split_images(image_data[...])
    # coefficient = memeffect.calculate_next_coefficient(img0, img1)
    # print(coefficient)
    # Coefficients from the set of loaded images
    # meme_coefficients = memeffect.calculate_coefficients()
    # print(np.angle(meme_coefficients))
    #
    # # Create a figure with 2 subplots
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    #
    # # Add title and axis labels to the first subplot
    # ax1.set_title('Amplitude')
    #
    #
    # # Add title and axis labels to the second subplot
    # ax2.set_title('Phase')
    # ax1.plot(np.real(meme_coefficients))
    # ax2.plot(np.angle(meme_coefficients))
    # # Show the figure
    # plt.show()

