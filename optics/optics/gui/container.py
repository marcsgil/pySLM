from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, List, Union, Sequence
import numpy as np
import time
import screeninfo
import logging

from optics.utils import Roi, Event, event
from optics.gui.key_descriptor import KeyDescriptor
from optics.gui.io import Element

log = logging.getLogger(__name__)

__all__ = ['Container', 'Window', 'Frame']


class ActionSet:
    def __init__(self):
        self.__action_list = []

    def __iter__(self):
        current_list = self.__action_list.copy()
        for _ in current_list:
            self.__action_list.remove(_)
            yield _

    def __call__(self):
        for _ in self:
            _()

    def __len__(self) -> int:
        return len(self.__action_list)

    def __iadd__(self, other: Union[Callable, Sequence[Callable]]) -> ActionSet:
        if not isinstance(other, Sequence):
            other = [other]
        self.__action_list += other

        return self

    def __isub__(self, other: Union[Callable, Sequence[Callable]]) -> ActionSet:
        if not isinstance(other, Sequence):
            other = [other]
        for task in other:
            self.__action_list.remove(task)

        return self


class Container:
    """
    Represents a GUI element that can can contain other GUI elements.
    """
    def __init__(self, container: tk.BaseWidget):
        self.__container: tk.BaseWidget = container

    @property
    def tk_container(self):
        return self.__container

    @property
    def title(self) -> str:
        return self.__tk_container.winfo_toplevel().title()

    @title.setter
    def title(self, new_title: str):
        self.__tk_container.winfo_toplevel().title(new_title)


class Window(Container):
    __root: tk.Tk = None
    __open_windows: List[Window] = []  # The Window class manages all windows.
    __root_on_update = Event('Window root event')
    __running = False

    @property
    def running(self) -> bool:
        return self.__running

    @classmethod
    def run(cls, action=None, frame_rate: Optional[float] = 25.0, update_interval: Optional[float] = None):
        cls.__running = True
        start_time = time.perf_counter()
        if action is None:
            action = lambda: True
        if update_interval is None:
            update_interval = 1.0 / frame_rate
        def all_actions() -> bool:
            for win in cls.__open_windows:
                win.update()
            return action() in [None, True]
        while all_actions() and cls.update():
            spare_time = time.perf_counter() - start_time - update_interval
            if spare_time > 0:
                time.sleep(spare_time)
            start_time = time.perf_counter()
        cls.__running = False

    @classmethod
    def update(cls) -> bool:
        """
        Updates the GUI Windows and returns True if any Window is still open.
        """
        for win in cls.__open_windows:
            win.on_update()  # handle all events on open windows
        if cls.__root is not None:
            try:
                if len(Window.__open_windows) > 0:
                    # cls.__root.update_idletasks()
                    cls.__root.update()
                    return True
                else:
                    cls.__root.quit()
                    cls.__root.destroy()
                    return False
            except tk.TclError:
                return False
        else:
            return False

    
    @classmethod
    def close_all(cls):
        log.info(f'Closing all {len(cls.__open_windows)} open windows...')
        for win in cls.__open_windows.copy():
            win.close()

    def __init__(self, master=None, title: str = '', roi=None, display_device: Optional[int] = None):
        """
        Initializes a tkinter Window.

        :param master: The master window, if not specified, one tk.Tk() is created, hidden, and reused.
        :param title: An optional title for the window.
        :param roi: The position and shape of the window.
        :param display_device: If specified, roi is ignored and the window will be displayed full-screen.
        """
        if master is None:
            if Window.__root is None:
                Window.__root = tk.Tk()
                Window.__root.withdraw()
            master = Window.__root
        else:
            Window.__root = master

        self.__actions = ActionSet()

        self.__on_update = Event(f'Window {self} update event')
        self.on_update += self.actions
        self.__on_key = Event(f'Window {self} key event')
        self.on_key += self.__on_key_initial_handler
        self.__on_change = Event(f'Window {self} change event')
        self.on_change += self.__on_change_initial_handler
        self.__on_close = Event(f'Window {self} close event')

        self.__full_screen_roi = None
        self.__full_screen = False
        self.__display_device = None

        self.__tk_toplevel_window = tk.Toplevel(master)
        self.__tk_toplevel_window.protocol('WM_DELETE_WINDOW', self.close)
        self.__tk_toplevel_window.bind('<Key>', lambda _: self.on_key(KeyDescriptor(_)))
        def on_configure(event):
            if not hasattr(self, 'previous_roi') or self.roi != self.previous_roi:
                self.on_change(self.roi)  # only act if something really changed
                self.previous_roi = self.roi
        self.__tk_toplevel_window.bind('<Configure>', on_configure)
        self.__tk_toplevel_window.title(title)

        super().__init__(self.__tk_toplevel_window)

        if display_device is not None:
            self.display_device = display_device
            self.full_screen = True
        else:
            self.roi = roi

        Window.__open_windows.append(self)

    @event
    def on_update(self) -> Event:
        """Called every frame."""
        return self.__on_update

    @event
    def on_key(self) -> Event:
        """Called with the KeyDescriptor every time a key is hit."""
        return self.__on_key

    @event
    def on_change(self) -> Event:
        """Called with the Roi every time the window is moved."""
        return self.__on_change

    @event
    def on_close(self) -> Event:
        """Called when the window is closed."""
        return self.__on_close

    @property
    def actions(self) -> ActionSet:
        return self.__actions

    @actions.setter
    def actions(self, actions: ActionSet):
        if actions != self.__actions:
            raise ValueError('Cannot change the ActionSet')

    @property
    def roi(self) -> Roi:
        return Roi(left=self.__tk_toplevel_window.winfo_x(), top=self.__tk_toplevel_window.winfo_y(),
                   width=self.__tk_toplevel_window.winfo_width(), height=self.__tk_toplevel_window.winfo_height())

    @roi.setter
    def roi(self, new_roi: Optional[Roi]):
        if new_roi is None:
            half_screen = np.array((self.__tk_toplevel_window.winfo_screenheight(), self.__tk_toplevel_window.winfo_screenwidth()), dtype=int) // 2
            new_roi = Roi(center=half_screen, shape=half_screen, dtype=int)
        self.__tk_toplevel_window.geometry(f'{new_roi.width}x{new_roi.height}+{new_roi.left}+{new_roi.top}')
        self.__tk_toplevel_window.update_idletasks()
        log.debug(f'Window set to {new_roi}.')

    @staticmethod
    def __screen_at_index(index: int):
        available_screens = screeninfo.get_monitors()
        if index >= len(available_screens):
            raise ValueError(f"Only {len(available_screens)} display devices available. Requested device {index}.")
        return available_screens[index]

    @property
    def full_screen(self) -> bool:
        return self.__full_screen

    @full_screen.setter
    def full_screen(self, state: bool):
        # index = self.display_device
        # if index is None:
        #     index = 0
        # screen = self.__screen_at_index(index)
        if state:
            self.__tk_toplevel_window.overrideredirect(state)
        # if screen.x == 0 and screen.y == 0 and False:
        #   self.__tk_toplevel_window.attributes('-fullscreen', state)  # Remove the task bar if this is the main screen
        self.__tk_toplevel_window.state('zoomed' if state else 'normal')
        # try:
        #     self.__tk_toplevel_window.attributes('-zoomed', state)   # Linux
        # except tk.TclError:
        #     log.debug('Could not use -zoomed, only works on Linux')
        self.__tk_toplevel_window.attributes('-topmost', state)
        self.__tk_toplevel_window.focus_set()  # restricted access main menu
        # self.__tk_toplevel_window.update()
        if not state:
            self.__tk_toplevel_window.overrideredirect(state)

        # Set cursor
        self.__tk_toplevel_window.config(cursor='none' if state else 'arrow')

        if state:
            self.__full_screen_roi = self.roi

        self.__full_screen = state

    @property
    def maximized(self) -> bool:
        return self.__tk_toplevel_window.state().lower() != 'normal'

    @maximized.setter
    def maximized(self, state: bool):
        if state:
            try:
                self.__tk_toplevel_window.state('zoomed')
            except Exception as e:
                log.info(e)
                log.info("Cannot maximize window using 'zoomed', setting maxsize() instead.")
                width = self.__tk_toplevel_window.winfo_screenwidth()
                height = self.__tk_toplevel_window.winfo_screenheight()
                self.roi = Roi(width=width, height=height)
        else:
            self.__tk_toplevel_window.state('normal')
        self.__tk_toplevel_window.update()

    @property
    def display_device(self) -> Optional[int]:
        return self.__display_device

    @display_device.setter
    def display_device(self, index: Optional[int]):
        if index is not None and index >= 0:
            screen = self.__screen_at_index(index)
            self.roi = Roi(left=screen.x, top=screen.y, height=screen.height, width=screen.width)
            self.full_screen = True
        else:
            self.full_screen = False
        self.__display_device = index

    @property
    def menu(self) -> tk.Menu:
        return self.tk_container.children['!menu']

    @menu.setter
    def menu(self, menu: tk.Menu):
        self.tk_container.config(menu=menu)

    def close(self):
        if self in Window.__open_windows:
            # Mark as closed before calling on_close (which may end up invoking this)
            Window.__open_windows.remove(self)
            # Execute user functions prior to actual closing.
            self.on_close()
            # Actually close the window.
            self.__tk_toplevel_window.destroy()

    def __on_key_initial_handler(self, key_desc: KeyDescriptor):
        log.debug(f'{key_desc} key pressed.')

        if key_desc == 'F11':
            self.full_screen = not self.full_screen
        elif key_desc == 'ESCAPE':  # Just escape, with the CTRL
            self.full_screen = False
            self.__tk_toplevel_window.state('normal')
            log.debug('Resetting region of interest.')


    def __on_change_initial_handler(self, new_roi: Roi):
        if self.full_screen and new_roi != self.__full_screen_roi:
            log.warn(f"Fullscreen window moved to {new_roi}! This shouldn't happen, exiting fullscreen mode now.")
            self.full_screen = False

class Frame(Element, Container):
    """
    Represents a positionable section of a GUI window, which can contain other GUI elements.
    """
    def __init__(self, master: Container):
        if isinstance(master, Container):  # needed for backwards compatibility
            master = master.tk_container
        master = ttk.Frame(master=master)
        Element.__init__(self, master)
        Container.__init__(self, master)

