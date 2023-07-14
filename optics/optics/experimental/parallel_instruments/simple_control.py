from __future__ import annotations

import sys
from typing import Callable, TypeVar, Generic, Any, List
import logging

from PySide6.QtCore import QSize, Qt, QRegularExpression
from PySide6.QtWidgets import QApplication, QMainWindow, QCompleter, \
    QPushButton, QSlider, QLabel, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, \
    QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtGui import QPalette, QColor, QFont, QRegularExpressionValidator, QValidator, QImage, QPixmap, \
    QMouseEvent, QWheelEvent, QKeyEvent, QResizeEvent, QTabletEvent

from optics.utils import unit_registry
from optics.experimental.parallel_instruments.drag_zoom_control import DragZoomFrame

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


class SimpleControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Example Control')

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

    window = SimpleControlWindow()
    window.show()

    sys.exit(app.exec())


