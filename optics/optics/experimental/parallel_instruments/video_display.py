import numpy as np
import sys
import time
from typing import Optional
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from multiprocessing import Lock
from optics.utils.ft import Grid
from optics.utils import Roi

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
# QtWidgets.QStyle.SP_DirOpenIcon


class Window(QtWidgets.QMainWindow):
    def __init__(self, roi: Optional[Roi] = None, *args, **kwargs):
        if roi is None:
            roi = Roi(dtype=np.dtype)
        self.__roi = roi
        self.__original_roi = Roi(*roi, dtype=np.float32)
        self.__updating_roi = None
        self.__zooming = False
        super().__init__(*args, **kwargs)

    @property
    def roi(self) -> Roi:
        return self.__roi

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        pos = (event.position().y(), event.position().x())
        left = event.buttons() & Qt.LeftButton
        middle = event.buttons() & Qt.MiddleButton
        right = event.buttons() & Qt.RightButton
        self.__updating_roi = Roi(top_left=pos, shape=(0, 0), dtype=np.float32)
        if left or middle:
            print(f'START DRAG AT {pos}')
            self.__zooming = False
        elif right:
            print(f'START RUBBERBAND ZOOM AT {pos}')
            self.__zooming = True

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        pos = np.asarray((event.position().y(), event.position().x()))
        print(f'MOUSE MOVE {pos}')
        self.__updating_roi.shape = pos - self.__updating_roi.top_left
        if self.__zooming:
            print(f'RUBBERBAND {self.__updating_roi}')
        else:
            self.__roi.center += self.__updating_roi.shape
            # self.__updating_roi.bottom_right = self.__updating_roi.top_left
            # self.__updating_roi.shape = (0, 0)
            self.__updating_roi[:] = [*pos, 0, 0]  # reset updating roi while dragging

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        pos = (event.position().y(), event.position().x())
        print(f'MOUSE RELEASE {pos}')
        if self.__zooming:
            self.__roi.shape = self.__updating_roi.shape
            self.__roi.center = self.__updating_roi.center
            self.__updating_roi = None

    def wheelEvent(self, event: QtGui.QWheelEvent):
        movement = [event.angleDelta().y(), event.angleDelta().x()]
        print(f'WHEEL MOVE BY {movement}')
        self.__roi.shape *= 1.10 ** (-movement[0] / 120)
        self.__roi.center += movement[1]

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        print('RESET FOV')
        self.__roi[:] = self.__original_roi

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        shift = event.modifiers() & Qt.ShiftModifier
        control = event.modifiers() & Qt.ControlModifier
        alt = event.modifiers() & Qt.AltModifier
        key_desc = ''.join([
            'S' if shift else ' ',
            'C' if control else ' ',
            'A' if alt else ' ',
        ])
        key_code = event.key()
        up = key_code == Qt.Key_Up
        down = key_code == Qt.Key_Down
        left = key_code == Qt.Key_Left
        right = key_code == Qt.Key_Right
        page_up = key_code == Qt.Key_PageUp
        page_down = key_code == Qt.Key_PageDown
        home = key_code == Qt.Key_Home
        end = key_code == Qt.Key_End
        full_screen = key_code in (Qt.Key_F, Qt.Key_F11)
        escape = key_code in (Qt.Key_Escape, Qt.Key_Exit)
        minus = key_code in (Qt.Key_Minus, Qt.Key_Underscore, Qt.Key_BracketLeft, Qt.Key_BraceLeft)
        plus = key_code in (Qt.Key_Plus, Qt.Key_Equal, Qt.Key_BracketRight, Qt.Key_BracketRight)
        # alt_gr = key_code == Qt.Key.Key_AltGr
        key_code_desc = ''.join([
            'u' if up else ' ',
            'd' if down else ' ',
            'l' if left else ' ',
            'r' if right else ' ',
            'U' if page_up else ' ',
            'D' if page_down else ' ',
            'L' if home else ' ',
            'R' if end else ' ',
            '+' if plus else ' ',
            '-' if minus else ' ',
            'EXIT FULLSCREEN' if escape else ' ',
            'FULLSCREEN' if full_screen else ' ',
        ])
        key_desc = f'{key_code}: {key_code_desc} - {key_desc}'
        print(f'KEYPRESS: {key_desc}')
        if control:
            if key_code == Qt.Key_C:
                self.close()
            elif key_code == Qt.Key_Q:
                self._quit()

    def closeEvent(self, event: QtCore.QEvent):
        settings = QtCore.QSettings('Optics', 'Monitor')
        settings.setValue('main/geometry', self.saveGeometry())
        settings.setValue('main/windowState', self.saveState())
        super().closeEvent(event)

    def _quit(self):
        print('QUIT')
        self.close()


class MainWindow(Window):
    def __init__(self, img0, img1, nb_frames):
        super().__init__()
        self.img0 = img0
        self.img1 = img1
        self.nb_frames = nb_frames

        self.count = 0
        self.times = np.zeros(nb_frames)
        self._lock = Lock()

        self.label = QtWidgets.QLabel()
        self.label.setCursor(Qt.CrossCursor)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(0, 32, 0))
        self.label.setPalette(palette)
        self.label.setAutoFillBackground(True)
        self.label.setMinimumSize(1, 1)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.update_image(self.img0)

        self.setCentralWidget(self.label)

        self.__restore_settings()

        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(lambda: self.flip_image())
        # self.timer.start(1000/60)  # [msec]

    # def event(self, event: QtCore.QEvent):
    #     if event.type() == QtCore.QEvent.UpdateRequest:
    #         # print(f'UPDATE {event}')
    #         self.flip_image()
    #         self.windowHandle().requestUpdate()
    #     # elif event.type() == QtCore.QEvent.Gesture:
    #     #     print(f'GESTURE {event}')
    #     # elif event.type() in (QtCore.QEvent.HoverMove, QtCore.QEvent.Wheel):
    #     #     pass
    #     # else:
    #     #     print(f'OTHER {event}')
    #
    #     # return super().event(event)
    #     return False

    def resizeEvent(self, event: QtGui.QResizeEvent):
        self.flip_image()

    def flip_image(self):
        with self._lock:
            if self.count < self.nb_frames:
                self.update_image(self.img0 if self.count % 2 == 0 else self.img1)
                # self.times[self.count] = time.perf_counter()
                #
                # self.count += 1
                #
                # if self.count == self.nb_frames:
                #     time_diffs = np.diff(self.times)
                #     print(f'pyside: {1 / np.median(time_diffs):0.3f} fps')
                #     self.close()

    def __add_legend(self, pixmap: QtGui.QPixmap):
        # create painter instance with pixmap
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        # set rectangle color and thickness
        legend_color = QtGui.QColor(255, 255, 255, 255)
        coord_color = QtGui.QColor(255, 255, 0, 255)
        bar_length = 100
        bar_height = bar_length / 5
        bar_margin = bar_height / 4
        bar_rect = QtCore.QRect(bar_margin, pixmap.rect().height() - bar_margin - bar_height, bar_length, bar_height)
        unit_font = QtGui.QFont('Sans', 1.25 * bar_height * 0.8)
        fm = QtGui.QFontMetrics(unit_font)
        unit_text = f'{pixmap.rect().width():0.1f} mm {time.perf_counter()}'
        unit_width = fm.horizontalAdvance(unit_text)
        unit_height = fm.height()
        unit_rect = QtCore.QRect(bar_margin, bar_rect.y() - unit_height, unit_width, unit_height)
        coord_font = QtGui.QFont('Sans', 12)
        fm = QtGui.QFontMetrics(coord_font)
        coord_text = f'x = {pixmap.rect().width():0.1f} px = {pixmap.rect().width():0.1f} mm\n' + \
                     f'y = {pixmap.rect().height():0.1f} px = {pixmap.rect().height():0.1f} mm'
        coord_box_width = fm.horizontalAdvance(coord_text.split('\n')[0])
        coord_box_height = fm.lineSpacing() * 2
        coord_rect = QtCore.QRect(pixmap.rect().width() - bar_margin - coord_box_width,
                                  pixmap.rect().height() - bar_margin - coord_box_height,
                                  coord_box_width, coord_box_height)

        # draw the scale bar
        painter.fillRect(bar_rect, legend_color)
        painter.setPen(QtGui.QPen(legend_color))
        painter.setFont(unit_font)
        painter.drawText(unit_rect, QtGui.Qt.AlignBottom, unit_text)
        # Draw the coordinates
        painter.setPen(QtGui.QPen(coord_color))
        painter.setFont(coord_font)
        painter.drawText(coord_rect, QtGui.Qt.AlignBottom | QtGui.Qt.AlignHCenter, coord_text)
        painter.end()

    def update_image(self, img):
        # Convert image to pixmap
        nb_channels = img.shape[-1] if img.ndim == 3 else 1
        form = (QtGui.QImage.Format_Grayscale8,
                QtGui.QImage.Format_Grayscale16,
                QtGui.QImage.Format_RGB888,
                QtGui.QImage.Format_RGBA8888,
                )[nb_channels - 1]
        qimage = QtGui.QImage(img.tobytes('C'), img.shape[1], img.shape[0], img.strides[0], form)
        # pixmap = QtGui.QPixmap.fromImage(qimage).scaled(img.shape[1] * 2, img.shape[0] * 2, Qt.KeepAspectRatio)
        pixmap = QtGui.QPixmap.fromImage(qimage).scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)

        # Draw legend over it
        self.__add_legend(pixmap)

        # Add to label for display
        self.label.setPixmap(pixmap)

        # Temporary debug output
        print(super().roi)

    def __restore_settings(self):
        settings = QtCore.QSettings('Optics', 'Monitor')
        self.restoreGeometry(settings.value('main/geometry'))
        self.restoreState(settings.value('main/windowState'))


if __name__ == "__main__":
    data_shape = np.array((768, 1024)) // 2

    grid = Grid(data_shape)
    rep = lambda img: np.repeat(img[..., np.newaxis], repeats=3, axis=-1)
    r = np.sqrt(sum(_ ** 2 for _ in grid))
    img0 = rep(r < np.amax(data_shape) / 4).astype(np.uint8) * 255
    img1 = rep((r > np.amax(data_shape) / 4) & (r < np.amax(data_shape) / 3)).astype(np.uint8) * 128
    if img0.ndim >= 3 and img0.shape[-1] > 1:
        img0[:, :, 0] = 0
        img0[:, :, 1] = 0
        img1[:, :, 0] = 0
        img1[:, :, 2] = 0
        if img0.shape[-1] > 3:
            img0[:, :, -1] = 255
            img1[:, :, -1] = 255
    # img0 = img0.astype(np.float32) / 255.5
    # img1 = img1.astype(np.float32) / 255.5

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    im = ax.imshow(img0)

    nb_frames = 100

    # times = np.zeros(nb_frames)
    # for _ in range(nb_frames // 2):
    #     im.set_data(img1)
    #     plt.show(block=False)
    #     times[_ * 2] = time.perf_counter()
    #     plt.pause(1e-6)
    #     im.set_data(img0)
    #     plt.show(block=False)
    #     times[_ * 2 + 1] = time.perf_counter()
    #     plt.pause(1e-6)

    # time_diffs = np.diff(times)
    # print(f'matplotlib: {1 / np.median(time_diffs):0.3f} fps')

    plt.close(fig)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(img0, img1, nb_frames)
    window.show()
    sys.exit(app.exec())

