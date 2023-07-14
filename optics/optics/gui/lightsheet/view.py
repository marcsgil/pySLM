import tkinter as tk
from abc import ABC, abstractmethod
import numpy as np
from tkinter.messagebox import showinfo
import webbrowser
import dataclasses
import queue
import threading

from . import log
from optics import __version__
from optics.gui.key_descriptor import KeyDescriptor
from optics.utils import Roi


class ViewError(Exception):
    def __init__(self, message):
        super().__init__(message)


@dataclasses.dataclass
class ViewSettings:
    window_roi: Roi = Roi()


class View(ABC):
    """
    Common base for Cam and SLM viewer windows opened by a Monitor object.
    """
    def __init__(self, monitor):
        self._monitor = monitor
        self._window = tk.Toplevel(monitor.root)
        # self._window.minsize(int(monitor.root.winfo_screenwidth()/2), int(monitor.root.winfo_screenheight()/2))
        self._window.protocol("WM_DELETE_WINDOW", self.close)
        self._window.bind('<Key>', self.__on_key)
        self.__previous_window_rect = None

        self.__title = ''
        self.title = ''

        # The menu bar
        menu_bar = tk.Menu(self._window)

        # # The File menu
        menu_file = tk.Menu(menu_bar, tearoff=False)
        # # # The New menu
        menu_new = tk.Menu(menu_file)
        # # # # The Cam menu
        menu_cam = tk.Menu(menu_new)

        def create_callback(view, name):
            return lambda: view(name)
        for cam_name in self._monitor.available_cam_names:
            menu_cam.add_command(label=cam_name, command=create_callback(self._monitor.open_cam_viewer, cam_name))
        menu_new.add_cascade(label='Camera', underline=0, menu=menu_cam)
        # # # # The SLM menu
        menu_slm = tk.Menu(menu_new)
        for slm_name in self._monitor.available_slm_names:
            menu_slm.add_command(label=slm_name, command=create_callback(self._monitor.open_slm_viewer, slm_name))
        # # # # The DM menu
        menu_dm = tk.Menu(menu_new)
        for dm_name in self._monitor.available_dm_names:
            menu_dm.add_command(label=dm_name, command=create_callback(self._monitor.open_dm_viewer, dm_name))
        menu_new.add_cascade(label='SLM', underline=0, menu=menu_slm)
        menu_new.add_cascade(label='DM', underline=0, menu=menu_dm)
        menu_file.add_cascade(label="New", underline=0, menu=menu_new)
        # The final File menus
        menu_file.add_separator()
        menu_file.add_command(label="Close", command=self.close)
        menu_file.add_command(label="Quit", command=self._monitor.close, accelerator='Alt-F4')

        menu_help = tk.Menu(menu_bar, tearoff=False)

        def show_controls():
            showinfo("Navigation", """
Left-click and drag the mouse to translate the region-of-interest.
Right-click drag to zoom a selected area.
Double-click or hit Escape to reset the view.
The region-of-interest of the camera is scaled correspondingly.

Zoom in with Ctrl-+
Zoom out with Ctrl--
Reset the zoom with Ctrl-0
Fullscreen mode: Ctrl-F
Maximize window: Ctrl-M
Reset window and exit fullscreen mode: Escape
Place window on the left: Ctrl-L
Place window on the right: Ctrl-R

Close a window with Alt-F4
Close the application with Ctrl-Q 
            """)
        menu_help.add_command(label='Controls', command=show_controls)
        menu_help.add_command(label='Source Code', command=lambda: webbrowser.open("https://github.com/tttom/lab/tree/master/code/python/optics#optics-package-for-the-lab", new=2))
        menu_help.add_separator()

        def show_about():
            showinfo("About", f"Monitor GUI version {__version__}")
        menu_help.add_command(label='About', command=show_about)

        # Put the menubar together
        menu_bar.add_cascade(label="File", underline=0, menu=menu_file)
        menu_bar.add_cascade(label="Camera", underline=0, menu=menu_cam)
        menu_bar.add_cascade(label="SLM", underline=0, menu=menu_slm)
        menu_bar.add_cascade(label="DM", underline=0, menu=menu_dm)
        menu_bar.add_cascade(label="Help", underline=0, menu=menu_help)
        self._window.config(menu=menu_bar)

        self.__full_screen = False

        # Set up an infinite loop to display whatever is pushed onto the queue
        self._running = True
        self.__trigger_queue = queue.Queue(maxsize=1)
        self.__image_queue = queue.Queue(maxsize=1)

        # Set up a thread to update the SLM pattern and display it
        self.__update_thread = threading.Thread(target=self._update_on_trigger, daemon=True)

        log.info('Starting continuous image display update...')
        self._window.after_idle(self.__continuous_display_update)

        # Continue with the sub-class constructor, and call _complete_initialization after that is done

    def _complete_initialization(self, settings: ViewSettings):
        self.roi = settings.window_roi

        # Size it for the first time, keep a reference to canvas_pil_image and canvas_tk_image
        self.fit_canvas_to_window()

        # self._window.pack_propagate(True)  # Make larger if widgets don't fit

        self._window.bind('<Configure>', self.__on_window_change)

        # Exit the constructor but let the thread run

        # The update thread waits for triggers on the trigger queue. For every trigger it takes an image and places
        # it on the image queue.
        # The main thread displays the GUI and pulls new images from the image queue for display.
        self.__update_thread.start()
        self.__trigger_queue.put(True)
        log.info("Viewer initialized.")

    def __enqueue_new_image(self):
        """
        Only called from thread
        """
        try:
            img = self._get_image()  # Ask sub-class to get an updated image
            log.debug('Got image, enqueueing it.')
            try:
                self.__image_queue.put(img)
                log.debug('Enqueued image.')
            except queue.Full:
                log.info("Dropping an image from the display queue.")
        except ViewError as ve:
            log.error(f"Ignoring a ViewError that occurred in the canvas updated loop: '{ve}'.")

    def _update_on_trigger(self):
        """
        Parallel running thread to feed new images to the View.
        """
        try:
            while self.__trigger_queue.get():
                log.debug('Got triggered, enqueueing new image...')
                self.__enqueue_new_image()
                log.debug('Enqueued image.')
                if self._running:
                    try:
                        log.debug('Running, triggering self again...')
                        # automatically trigger again, else: wait for external trigger
                        self.__trigger_queue.put(True, block=False)
                    except queue.Full:
                        log.warn('Trigger queue full!')
        except Exception as e:
            log.error(f'Uncaught Exception: {e}!')
            raise e

        log.info('Exiting update loop.')
        # log.info('No more triggers, sending exit signal to the image display loop...')
        # with self.__image_queue.mutex:
        #     self.__image_queue.queue.clear()
        #     self.__image_queue.queue.append(False)
        # log.info('Sent the exit signal to the image display loop.')

    def __continuous_display_update(self):
        try:
            image_array = self.__image_queue.get(block=False)   # todo True, timeout=0.001)
            if image_array is not False and image_array.size != 0:
                # log.debug(f'Received image of shape {image_array.shape} on display queue (from {self.__update_thread.name}).')
                self._canvas.image = image_array
            else:
                log.debug('Received termination signal from queue.')
                return
        except queue.Empty:
            pass
        self._window.after(40, func=self.__continuous_display_update)

    def __on_key(self, event):
        key_desc = KeyDescriptor(event)
        log.debug(f'{key_desc} key pressed.')

        if key_desc.control:
            if key_desc == 'Q':  # Ctrl-Q
                self._monitor.disconnect()
            # elif key_desc == 'C':  # Ctrl-C
            #     self.close()
            elif key_desc == 'F':  # F
                self.full_screen = not self.full_screen
            elif key_desc == 'M':  # Ctrl-M
                log.debug("Toggle Maximize")
                self.full_screen = False
                if self._window.state() == 'normal':
                    try:
                        self._window.state('zoomed')
                    except:
                        log.info("Cannot maximize window using 'zoomed', setting maxsize() instead.")
                        width = self._window.winfo_screenwidth()
                        height = self._window.winfo_screenheight()
                        self._window.geometry(f'{width}x{height}+0+0')
                else:
                    self._window.state('normal')
                self._window.update()
            elif key_desc == 'L':  # Ctrl-L
                self.full_screen = False
                self._window.state('normal')
                width = int(self._window.winfo_screenwidth() / 2)
                height = self._window.winfo_screenheight()
                self._window.geometry(f'{width}x{height}+0+0')
                self._window.update()
            elif key_desc == 'R':  # Ctrl-R
                self.full_screen = False
                self._window.state('normal')
                width = int(self._window.winfo_screenwidth() / 2)
                height = self._window.winfo_screenheight()
                self.roi = Roi(width=width, height=height, top=0, left=width+1)
            elif key_desc == 'EQUAL':  # Ctrl-= or Ctrl-+
                log.debug("Zoom in")
                self._canvas.zoom(1.0)
            elif key_desc == 'MINUS':  # Ctrl--
                log.debug("Zoom out")
                self._canvas.zoom(-1.0)
            elif key_desc == '0':  # Ctrl-0
                log.debug("Reset drag and zoom")
                self._canvas.reset_drag_and_zoom()
        elif key_desc == 'ESCAPE':  # Just escape, with the CTRL
            self.full_screen = False
            self._window.state('normal')
            log.debug('Resetting region of interest.')
            self._canvas.reset_drag_and_zoom()

    def __on_window_change(self, event):
        new_rect = Roi(top_left=(event.y, event.x), shape=(event.height, event.width))
        if self.__previous_window_rect is None or np.any(new_rect.shape != self.__previous_window_rect.shape):
            self._update_window()
        self.__previous_window_rect = new_rect

    @property
    def full_screen(self):
        return self.__full_screen

    @full_screen.setter
    def full_screen(self, value):
        self._window.attributes("-fullscreen", value)
        self._window.overrideredirect(True)
        self.__full_screen = value

    def _update_window(self):
        self._canvas.fit_to_window()

    @abstractmethod
    def _get_image(self):
        """
        This gets called from self.__enqueue_new_image() on the self.__update_thread

        :return: A 3d uint8 numpy ndarray with the red-green-blue image planes
        """
        pass

    def fit_canvas_to_window(self):
        self._canvas.fit_to_window()

    @property
    def roi(self):
        return Roi(left=self._window.winfo_x(), top=self._window.winfo_y(),
                   width=self._window.winfo_width(), height=self._window.winfo_height())

    @roi.setter
    def roi(self, new_roi):
        if new_roi is not None:
            if np.any(new_roi.shape == 0):
                new_roi = Roi(top_left=[0, 0], shape=[480, 640])
            self._window.geometry(f'{new_roi.width}x{new_roi.height}+{new_roi.left}+{new_roi.top}')
            log.info(f'Setting window to {new_roi}...')
            self._window.update_idletasks()
            log.info(f'Window set to {new_roi}.')

    settings = ViewSettings()

    def close(self):
        settings = self.settings  # Store settings before destroying the window (and its position)
        log.debug(f'Sending the stop trigger signal for thread {self.__update_thread.name}...')
        self.__trigger_queue.put(False)
        log.debug(f'Clearing space on the image queue to unblock thread {self.__update_thread.name} if necessary...')
        # with self.__image_queue.mutex:
        #     self.__image_queue.queue.clear()
        try:
            self.__image_queue.get(block=False)
        except queue.Empty:
            pass
        log.debug(f'Waiting for thread {self.__update_thread.name} to end...')
        self.__update_thread.join()
        log.debug(f'Thread {self.__update_thread.name} ended.')
        log.debug('Trying to destroy window...')
        self._window.destroy()
        log.debug('Destroyed window.')
        self._monitor.closed_window(self, settings)
