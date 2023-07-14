from abc import ABC, abstractmethod
from typing import List, Union, Callable
import screeninfo
import numpy as np
import re
import logging

from optics.utils import Roi

from optics.instruments import InstrumentError, Instrument, InstrumentDescriptor

log = logging.getLogger(__name__)


class DisplayError(InstrumentError):
    """
    An Exception for Display objects.
    """
    pass


class DisplayDescriptor(InstrumentDescriptor):
    def __init__(self, id: str, constructor: Callable, available: bool = True, index: int = -1, roi: Roi = None, name: str = ''):
        super().__init__(id, constructor, available)
        self.__index: int = index
        self.__roi: Roi = roi
        self.__name: str = name

    @property
    def index(self) -> int:
        """The index of the display."""
        return self.__index

    @property
    def roi(self) -> Roi:
        """The region of interest corresponding to this display."""
        return self.__roi

    @property
    def name(self) -> str:
        """The interal name of the display of this display."""
        return self.__name


class Display(Instrument):
    @classmethod
    def list(cls, recursive: bool = False, include_unavailable: bool = False) -> List[Union[DisplayDescriptor, List]]:
        """
        Return all constructors.
        :return: A dictionary with as key the class and as value, a dictionary with subclasses.
        """
        # Detect the available screen devices
        available_screens = cls.__get_screen_infos()
        display_discriptors = super().list(recursive=recursive, include_unavailable=include_unavailable)
        if len(display_discriptors) == 0:  # a leaf class
            for _, s in enumerate(available_screens):
                if s.name is not None:
                    number_match = re.search(r'\d+$', s.name)
                    index = number_match.group(0) if number_match is not None else ''
                else:
                    index = _
                display_discriptors.append(
                    DisplayDescriptor(f'{cls.__name__}{index}_{s.width}x{s.height}_at_{s.x}_{s.y}',
                                      lambda: Display(),
                                      index=index,
                                      roi=Roi(width=s.width, height=s.height, left=s.x, top=s.y),
                                      name=s.name,
                                      )
                )

        return display_discriptors

    @classmethod
    def __get_screen_infos(cls):
        # Detect the available screen devices
        available_screens = screeninfo.get_monitors()
        for s in available_screens:
            s.index = s.name
        return available_screens

    def __init__(self, index=None, shape=None):
        """
        Constructor of abstract Screen class to implement full-screen displays.
        :param index: The number of the display to use, or None.
        :param shape: The shape of the screen (height x width) as a numpy vector.
        This option is ignored if a display is provided.
        """
        super().__init__()

        if index is not None and index >= 0:
            screens = type(self).__get_screen_infos()
            if index >= len(screens):
                raise ValueError(f"Only screens {np.arange(len(screens)).tolist()} available. Requested screen {index}.")
            screen = screens[index]
            shape = (screen.height, screen.width)
        else:  # Useful for TestScreen
            screen = None

        self.__screen = screen
        self.__shape = shape
        self.__callback = None

    @property
    def shape(self):
        return self.__shape

    @property
    def screen(self):
        return self.__screen

    @property
    def callback(self):
        return self.__callback

    @callback.setter
    def callback(self, new_callback):
        self.__callback = new_callback

    def show(self, image_array):
        image_array = np.array(image_array)
        if image_array.dtype == float:
            # Convert to uint8 array
            image_array = np.array(np.clip(image_array, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)
        # Make 3D array
        while image_array.ndim < 3:
            image_array = np.expand_dims(image_array, axis=image_array.ndim)
        # If a 3-vector, interpret as a uniform color
        if image_array.size == 3:
            image_array.reshape((1, 1, 3))
        # If 2D input, interpret as grayscale
        if image_array.shape[2] == 1:
            image_array = np.repeat(image_array[:, :, 0:1], repeats=3, axis=2)

        self._show(image_array)  # Delegate this

        # Handle callback if any
        if self.callback is not None:
            self.callback(image_array)

    @abstractmethod
    def _show(self, image_array):
        """
        Display an image on the screen.
        :param image_array: A uint8 numpy ndarray with a shape matching that of this screen
        """
        raise NotImplementedError

