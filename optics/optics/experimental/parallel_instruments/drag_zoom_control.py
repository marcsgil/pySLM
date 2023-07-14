from __future__ import annotations

import sys
from typing import Callable, TypeVar, Generic, Any, List
import numpy as np
import logging

from PySide6.QtCore import QSize, Qt, QRegularExpression
from PySide6.QtWidgets import QApplication, QMainWindow, QCompleter, \
    QPushButton, QSlider, QLabel, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, \
    QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtGui import QPalette, QColor, QFont, QRegularExpressionValidator, QValidator, QImage, QPixmap, \
    QMouseEvent, QWheelEvent, QKeyEvent, QResizeEvent, QTabletEvent

from optics.utils import unit_registry
from optics.utils import Roi

log = logging.getLogger(__name__)

_T = TypeVar('_T')


class NumericEdit(QDoubleSpinBox):
    def __init__(self, unit: str = '', *args, **kwargs):
        self.unit = unit
        super().__init__(*args, **kwargs)
        self.setAccelerated(False)
        self.setAlignment(Qt.AlignRight)
        self.setKeyboardTracking(False)
        self.setRange(0, 1)

        # Add a custom LineEdit
        line_edit = QLineEdit()
        line_edit.setAlignment(Qt.AlignRight)
        validator = QRegularExpressionValidator(
            QRegularExpression(r'[+-]?\d*[\.\d]\d*\s?([YZEPTGMkhdcmμunpfazy]|da)?' + f'{self.unit}?'), line_edit)
        line_edit.setValidator(validator)
        self.setLineEdit(line_edit)

        self.setDecimals(3)
        self.setSingleStep(0.001)

    def validate(self, input: str, pos: int) -> QValidator.State:
        return self.lineEdit().validator().validate(input, pos)

    def valueFromText(self, input: str) -> float:
        input = input.replace(' ', '').replace('μ', 'u')
        dummy_unit = 's'  # not an SI unit prefix
        if not input.endswith(dummy_unit):
            input += dummy_unit
        quantity = unit_registry.parse_expression(input)
        return quantity.to_base_units().magnitude

    def textFromValue(self, value: float) -> str:
        quantity = unit_registry.Quantity(value=value, units=self.unit)
        return f'{quantity.to_reduced_units().to_compact():0.3~f}'


class DragZoomFrame(QLabel):
    def __init__(self, *args, **kwargs):
        self.__qimage = None
        self.__pixmap = None
        self.__roi = Roi(shape=[1, 1])
        self.__mouse_press_pos = None
        self.__prev_roi_center = None

        super().__init__(*args, **kwargs)

        # self.setAutoFillBackground(False)

        self.setMinimumSize(QSize(80, 80))
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
        print(f'RESIZE to {shape}')
        self.__roi = Roi(center=self.__roi.center, shape=shape)
        self.__update()

    def __event_desc(self, event):
        try:
            pos = (event.position().y(), event.position.x())
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
        print(f'MOUSE MOVE {self.__event_desc(event)}')
        rel_pos = np.array([event.position().y(), event.position().x()]) - self.__mouse_press_pos
        self.__roi = Roi(center=self.__prev_roi_center - rel_pos, shape=self.__roi.shape)
        self.__update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        print(f'DOUBLE CLICK RESET {self.__event_desc(event)}')
        # rel_pos = np.array([event.position().y(), event.position().x()]) - self.__mouse_press_pos
        shape = [self.size().height(), self.size().width()]

        self.__roi = Roi(center=[0, 0], shape=shape)
        self.__update()

    def mousePressEvent(self, event: QMouseEvent):
        self.__mouse_press_pos = np.array([event.position().y(), event.position().x()])
        self.__prev_roi_center = self.__roi.center
        print(f'MOUSE PRESS {self.__event_desc(event)}')

    def mouseReleaseEvent(self, event: QMouseEvent):
        print(f'MOUSE RELEASE {self.__event_desc(event)}')
        self.__mouse_press_pos = None
        self.__prev_roi_center = None

    def wheelEvent(self, event: QWheelEvent):
        print(f'WHEEL {self.__event_desc(event.globalPosition())}')

    def tabletEvent(self, event: QTabletEvent):
        print(f'TABLET {self.__event_desc(event)}')

    def keyPressEvent(self, event: QKeyEvent):
        print(f'KEY PRESS {self.__event_desc(event)}')

    def keyReleaseEvent(self, event: QKeyEvent):
        print(f'KEY RELEASE {self.__event_desc(event)}')


class DragZoomControlWindow(QMainWindow):
    def __init__(self, value_prop: Prop, text_prop: value_prop):
        super().__init__()
        self.setWindowTitle('Drag-zoom Control')

        numeric_edit = NumericEdit('s')

        label = QLabel()
        label.setFont(QFont('Arial', 48))
        label.setAlignment(Qt.AlignRight)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(100)
        self.slider.setRange(0, 1000)
        self.slider.setSingleStep(10)
        self.slider.setPageStep(100)

        self.line_edit = QLineEdit()
        completer = QCompleter(['Aardvark', 'abacus', 'alter'])
        self.line_edit.setCompleter(completer)
        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setCompleter(completer)

        self.drag_zoom_frame = DragZoomFrame()

        # scn = QGraphicsScene()
        # pixmap_item = QGraphicsPixmapItem(pixmap)
        # scn.addItem(pixmap_item)
        # self.image_label = QGraphicsView(scn)
        # self.image_label.fitInView(pixmap_item)
        # gfxPixItem = scn.addPixmap(pixmap)
        # self.image_label = QGraphicsView(scn)
        # self.image_label.fitInView(gfxPixItem)
        # self.image_label.show()

        # self.drag_zoom_frame.setPixmap(pixmap)
        # self.image_label.setPixmap(pixmap.scaled(60, 80, Qt.KeepAspectRatio))

        # Connect the controls to the properties
        numeric_edit.valueChanged.connect(value_prop.on_next)
        self.slider.valueChanged.connect(lambda _: value_prop.on_next(_ / 1000))
        value_prop.subscribe(numeric_edit.setValue)
        value_prop.subscribe(lambda _: self.slider.setValue(_ * 1000))
        value_prop.subscribe(lambda _: label.setText(f'{_:0.3f}'))

        self.line_edit.editingFinished.connect(lambda: text_prop.on_next(self.line_edit.text()))
        self.combo.currentTextChanged.connect(lambda: text_prop.on_next(self.combo.currentText()))
        text_prop.subscribe(self.line_edit.setText)
        text_prop.subscribe(self.combo.setCurrentText)

        # self.setMinimumSize(QSize(400, 300))

        line_layout = QHBoxLayout()
        line_layout.addWidget(self.slider)

        line_widget = QWidget()
        line_widget.setLayout(line_layout)

        layout = QVBoxLayout()
        layout.addWidget(numeric_edit)
        layout.addWidget(line_widget)
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(self.line_edit)
        layout.addWidget(self.combo)
        layout.addStretch()
        self.drag_zoom_frame.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.drag_zoom_frame)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


class Prop(Generic[_T]):
    def __init__(self, value: _T):
        self.__subscribers: List[Callable[[_T], Any]] = []
        self.__value: _T = value

    @property
    def value(self) -> _T:
        return self.__value

    @value.setter
    def value(self, new_value: _T):
        self.on_next(new_value)

    def subscribe(self, on_next: Callable[[_T], Any]):
        self.__subscribers.append(on_next)
        on_next(self.value)

    def on_next(self, value: _T):
        self.__value = value
        for _ in self.__subscribers:
            _(self.__value)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    value_prop = Prop(1 / 3)
    text_prop = Prop('x^2 + y^2 < 100^2')

    window = DragZoomControlWindow(value_prop, text_prop)
    window.show()

    # window2 = Window(value_prop, text_prop)
    # window2.show()

    def show(new_val):
        log.info(f'{text_prop.value}: {value_prop.value}')

    value_prop.subscribe(show)
    text_prop.subscribe(show)

    sys.exit(app.exec())


