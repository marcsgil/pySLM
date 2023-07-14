import logging
log = logging.getLogger(__name__)
from . import Display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.axes import Axes
from typing import Union, Sequence, Optional

array_like = Union[np.ndarray, Sequence]


class SimulatedDisplay(Display):
    """
    A class to represent simulated full-screen displays for testing.
    """
    def __init__(self, image_or_axes: Union[AxesImage, Axes, None] = None, shape: Optional[array_like] = None):
        """
        Construct an object to represent a simulated full-screen display.

        :param image_or_axes: (optional) An :py:class:`AxesImage` or an :py:class:`Axes` object in which to show the
            simulated full-screen display. The shape is automatically detected from this.
        :param shape: (optional) The shape of the full screen display when no image or axes is specified.
        """
        if isinstance(image_or_axes, AxesImage):
            self.__image = image_or_axes
        elif isinstance(image_or_axes, Axes):
            # self.__figure = image_axes.get_figure()
            images = image_or_axes.get_images()
            if len(images) > 0:
                self.__image = images[0]
            else:
                if shape is None:
                    shape = (150, 200)
                if len(shape) < 2:
                    shape = (shape, shape)
                else:
                    shape = shape[:2]

                image = np.zeros((*shape, 3))
                self.__image = image_or_axes.imshow(image)
        else:
            self.__image = None

        if self.__image is not None:
            shape = self.__image.get_array().shape[:2]

        super().__init__(shape=shape)

    def _show(self, image_array):
        """
        Display an image on the screen.
        :param image_array: A numpy ndarray with a shape matching that of this screen
        """
        if self.__image is not None:
            # If grayscale, expand to all pixels
            if image_array.shape[0] == 1 and image_array.shape[1] == 1:
                image_array = np.ones((*self.shape, 1), dtype=np.uint8) * image_array

            # Copy bytes to full screen window
            self.__image.set_data(image_array)
            plt.show(block=False)
            # time.sleep(0.001)  # allow time for figure to update


if __name__ == '__main__':
    import time

    with SimulatedDisplay() as fs:
        img = np.zeros((*fs.shape, 3), dtype=np.uint8)
        img[::2, ::2, :] = 255
        img[1::2, 1::2, :] = 255
        rng = range(0, fs.shape[1], 20)
        start_time = time.time()
        for idx in rng:
            img[:, idx, 1] = 255
            fs.show(img)

        total_time = time.time() - start_time
        frames_per_second = len(rng) / total_time
        log.info("FullScreen display at {:0.1f} frames per second.".format(frames_per_second))

