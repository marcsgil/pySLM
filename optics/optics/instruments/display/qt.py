import logging
log = logging.getLogger(__name__)
from . import Display
import numpy as np
from PySide6.QtGui import QPixmap, QKeyEvent
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt


class QtFullScreenWindow(QWidget):
    def __init__(self, screen_idx, screen_shape):
        super().__init__()
        self.setWindowTitle("FullScreen{:d}".format(screen_idx))
        self.setGeometry(0, 0, screen_shape[1], screen_shape[0])
        # self.resize(pixmap.width(),pixmap.height())
        # self.showFullScreen()
        self.show()

        # Prepare the bitmap
        image_array = np.zeros((*screen_shape[:2], 3))
        self.__raw_bytes_header = b'P6\n%i %i\n255\n' % (image_array.shape[1], image_array.shape[0])
        self.__image = QPixmap()
        self.__image.loadFromData(self.__raw_bytes_header + image_array.tobytes())

        self.__label = QLabel(self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.__label, alignment=Qt.AlignTop)
        self.setLayout(vbox)

        self.show()

    def keyPressEvent(self, event: QKeyEvent):
        # Just for safety
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def show_image(self, image_array):
        self.__image.loadFromData(self.__raw_bytes_header + image_array.tobytes())
        self.__label.setPixmap(self.__image)
        self.__label.show()
        self.show()
        time.sleep(0.001)


class QtDisplay(Display):
    def __init__(self, index):
        super().__init__(index)

        self.__app = QApplication([])
        self.__window = QtFullScreenWindow(index, self.shape)

    def _show(self, image_array):
        # If grayscale, expand to all pixels
        if image_array.shape[0] == 1 and image_array.shape[1] == 1:
            image_array = np.ones((*self.shape, 1), dtype=np.uint8) * image_array
        self.__window.show_image(image_array)

    def _disconnect(self):
        self.__window.destroy()


if __name__ == '__main__':
    import time

    fs = QtDisplay(0)
    img = np.zeros((*fs.shape, 3), dtype=np.uint8)
    img[::2,::2,:] = 255
    img[1::2,1::2,:] = 255
    rng = range(0, fs.shape[1], 200)
    start_time = time.time()
    for idx in rng:
        img[:, idx, 1] = 255
        fs.show(img)

    total_time = time.time() - start_time
    frames_per_second = len(rng) / total_time
    log.info("FullScreen display at {:0.1f} frames per second.".format(frames_per_second))

    fs.disconnect()
