import numpy as np
import tkinter as tk
import threading

import logging
log = logging.getLogger(__name__)
from . import Display
from optics.utils import Roi
from optics.gui.container import Window
from optics.gui.io import Canvas


class TkDisplay(Display):
    def __init__(self, index):
        super().__init__(index)

        self.__window_lock = threading.Lock()

        # Set up a window and make it full-screen
        self.__window = Window(title=f'FullScreen Window {index}', display_device=index)

        # Prepare the bitmap
        image_array = np.zeros((self.screen.height, self.screen.width, 3), dtype=np.uint8)
        self.__raw_bytes_header = b'P6\n%i %i\n255\n' % (image_array.shape[1], image_array.shape[0])
        self.__image = tk.PhotoImage(data=self.__raw_bytes_header + image_array.tobytes())
        self.__canvas = Canvas(self.__window, height=self.screen.height, width=self.screen.width)
        self.__canvas.create_image(0, 0, anchor=tk.NW, image=self.__image)
        self.__canvas.pack()
        self.__window.roi = Roi(left=self.screen.x, top=self.screen.y, height=self.screen.height, width=self.screen.width)

    def _show(self, image_array):
        with self.__window_lock:
            if self.__window is not None:
                # If grayscale, expand to all pixels
                if image_array.shape[0] == 1 and image_array.shape[1] == 1:
                    image_array = np.ones((*self.shape, 1), dtype=np.uint8) * image_array
                def update_image():
                    # Copy bytes to window on full screen display
                    self.__image.put(self.__raw_bytes_header + image_array.tobytes())  # https://en.wikipedia.org/wiki/Netpbm_format
                self.__window.actions += update_image
                if not self.__window.running and threading.current_thread() is threading.main_thread():
                    self.__window.update()
            else:
                log.info("Fullscreen window closed while trying to update it.")

    def _disconnect(self):
        with self.__window_lock:
            # self.__set_fullscreen(False)
            self.__window.close()
            self.__window = None


if __name__ == '__main__':
    import time

    fs = TkDisplay(0)
    img = np.zeros((*fs.shape, 3), dtype=np.uint8)
    img[::2,::2,:] = 255
    img[1::2,1::2,:] = 255
    rng = range(0, fs.shape[1], 20)
    start_time = time.time()
    for idx in rng:
        img[:, idx, 1] = 255
        fs.show(img)

    total_time = time.time() - start_time
    frames_per_second = len(rng) / total_time
    log.info("FullScreen display at {:0.1f} frames per second.".format(frames_per_second))

    fs.disconnect()
