import numpy as np
# import matplotlib.pyplot as plt

from PySide6.QtCore import QSize, Qt, QRegularExpression
from PySide6.QtWidgets import QApplication, QMainWindow, QCompleter, \
    QPushButton, QSlider, QLabel, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, \
    QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtGui import QPalette, QColor, QFont, QRegularExpressionValidator, QValidator, QImage, QPixmap, \
    QMouseEvent, QWheelEvent, QKeyEvent, QResizeEvent, QTabletEvent

from optics import log
from optics.utils import ft, Roi


class DragZoomFrame(QLabel):
    def __init__(self, *args, **kwargs):
        self.__qimage = None
        self.__pixmap = None
        self.__roi = Roi(shape=[1, 1])
        self.__mouse_press_pos = None
        self.__prev_roi_center = None

        super().__init__(*args, **kwargs)

        # self.setAutoFillBackground(False)

        # self.setMinimumSize(QSize(80, 80))
        # self.setMaximumSize(QSize(4000, 4000))
        self.grabKeyboard()

        self.__update()

    def __get_image(self, roi: Roi) -> np.ndarray:
        r = np.sqrt(sum(_ ** 2 for _ in roi.grid))
        # img = ((r % 20.0) * 255.0 / 20.0)
        img = (r < 50).astype(np.uint8) * 128
        img = np.repeat(img[..., np.newaxis], repeats=3, axis=-1)
        return img

    def __update(self):
        img = self.__get_image(self.__roi)
        self.__qimage = QImage(img.tobytes('C'), img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        self.__pixmap = QPixmap.fromImage(self.__qimage)  # .scaled(img.shape[1] * 2, img.shape[0] * 2, Qt.KeepAspectRatio)
        self.setPixmap(self.__pixmap)
        print(self.__roi)

    def resizeEvent(self, event: QResizeEvent):
        shape = (event.size().height(), event.size().width())
        log.info(f'RESIZE to {shape}')
        self.__roi = Roi(center=self.__roi.center, shape=shape)
        self.__update()

    def __event_desc(self, event):
        try:
            pos = (event.localPos().y(), event.localPos().x())
            left = event.buttons() & Qt.LeftButton
            middle = event.buttons() & Qt.MiddleButton
            right = event.buttons() & Qt.RightButton
            button_desc = ''.join([
                'L' if left else ' ',
                'M' if middle else ' ',
                'R' if right else ' ',
            ])
        except AttributeError:
            pos = (0, 0)
            button_desc = '   '
        try:
            shift = event.modifiers() & Qt.ShiftModifier
            control = event.modifiers() & Qt.ControlModifier
            alt = event.modifiers() & Qt.AltModifier
            key_desc = ''.join([
                'S' if shift else ' ',
                'C' if control else ' ',
                'A' if alt else ' ',
            ])
        except AttributeError:
            key_desc = '   '
        try:
            key_code = event.key()
            up = key_code == Qt.Key_Up
            down = key_code == Qt.Key_Down
            left = key_code == Qt.Key_Left
            right = key_code == Qt.Key_Right
            page_up = key_code == Qt.Key_PageUp
            page_down = key_code == Qt.Key_PageDown
            home = key_code == Qt.Key_Home
            end = key_code == Qt.Key_End
            full_screen = key_code == Qt.Key_F or key_code == Qt.Key_F11
            esc = key_code == Qt.Key_Escape or key_code == Qt.Key_Exit
            plus = key_code == Qt.Key_Escape or key_code == Qt.Key_Plus
            minus = key_code == Qt.Key_Escape or key_code == Qt.Key_Minus
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
            ])
            key_desc = f'{key_code}: {key_code_desc} - {key_desc}'
            if full_screen:
                key_desc += ' FULLSCREEN'
            if esc:
                key_desc += ' ESCAPE'
        except AttributeError as e:
            pass
        return f'{pos} | {button_desc} {key_desc}'

    def mouseMoveEvent(self, event: QMouseEvent):
        log.info(f'MOUSE MOVE {self.__event_desc(event)}')
        rel_pos = np.array([event.localPos().y(), event.localPos().x()]) - self.__mouse_press_pos
        self.__roi = Roi(center=self.__prev_roi_center - rel_pos, shape=self.__roi.shape)
        self.__update()

    def mousePressEvent(self, event: QMouseEvent):
        self.__mouse_press_pos = np.array([event.localPos().y(), event.localPos().x()])
        self.__prev_roi_center = self.__roi.center
        log.info(f'MOUSE PRESS {self.__event_desc(event)}')

    def mouseReleaseEvent(self, event: QMouseEvent):
        log.info(f'MOUSE RELEASE {self.__event_desc(event)}')
        self.__mouse_press_pos = None
        self.__prev_roi_center = None

    def wheelEvent(self, event: QWheelEvent):
        log.info(f'WHEEL {self.__event_desc(event.globalPosition())}')

    def tabletEvent(self, event: QTabletEvent):
        log.info(f'TABLET {self.__event_desc(event)}')

    def keyPressEvent(self, event: QKeyEvent):
        log.info(f'KEY PRESS {self.__event_desc(event)}')

    def keyReleaseEvent(self, event: QKeyEvent):
        log.info(f'KEY RELEASE {self.__event_desc(event)}')


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Waves')

        self.drag_zoom_frame = DragZoomFrame()

        self.drag_zoom_frame.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.drag_zoom_frame)


if __name__ == '__main__':
    # grid = ft.Grid()

    app = QApplication()

    window = Window()
    window.show()

    app.exec()
