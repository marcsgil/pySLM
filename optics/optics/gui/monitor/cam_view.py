import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter import ttk
import numpy as np
from datetime import datetime
import dataclasses
import re
from pathlib import Path
from PIL import Image
import cv2
import logging

from .view import View, ViewSettings, ViewError
from optics import __version__
from optics.utils import Roi
from optics.gui import SwitchButton, Combobox, OptionMenu, Frame, DragZoomCanvas
from optics.utils import round125
from optics.instruments.cam import CamError

log = logging.getLogger(__name__)


@dataclasses.dataclass
class CamSettings:
    name: str = ''
    exposure_time: str = '20'
    gain: str = '1'
    roi: Roi = Roi()
    magnification: float = 1.0


@dataclasses.dataclass
class CamViewSettings(ViewSettings):
    cam: CamSettings = CamSettings()
    file_name: str = ''
    normalize: bool = False


class CamControl:
    def __init__(self, master, monitor, settings: CamViewSettings):
        self.__monitor = monitor
        self.__settings = settings
        self.__recording = False

        if settings.cam.name == '':
            settings.cam.name = self.__monitor.available_cam_names[0]
        self.__cam = self.__monitor.get_cam(settings.cam.name)

        # Create the frame to hold everything
        self.__frm = Frame(master)

        frm_top = Frame(master=self.__frm)

        cam_names = self.__monitor.available_cam_names
        self.__dd_cam = OptionMenu(frm_top, value=self.settings.cam.name, width=16,
                                   values=("", *cam_names), command=self.__update_cam)
        self.__dd_cam.pack(side=tk.LEFT)
        self.__cb_magnification = Combobox(frm_top, width=4, value=f"{self.settings.cam.magnification:0.0f}x",
                                           values=[f"{v:0.0f}x" for v in [1, 5, 10, 20, 50, 100]],
                                           command=self.__update_magnification)
        self.__cb_magnification.pack(side=tk.LEFT)
        self.__cb_exposure_time = Combobox(frm_top, width=4, value=self.settings.cam.exposure_time,
                                           values=[f"{round125(t):0.0f}" for t in 10**(np.arange(0, 9)/3)],
                                           command=self.__update_exposure_time)
        self.__cb_exposure_time.pack(side=tk.LEFT)
        ttk.Label(frm_top.tk_container, text="ms, ").pack(side=tk.LEFT)
        ttk.Label(frm_top.tk_container, text="gain:").pack(side=tk.LEFT)
        self.__cb_gain = Combobox(frm_top, width=4, value=self.settings.cam.gain, values=[0, 1, 2, 5, 10, 20, 50, 100],
                                  command=self.__update_gain)
        self.__cb_gain.pack(side=tk.LEFT)

        def set_normalize(value: int):
            self.normalize = value
        self.__but_normalize = SwitchButton(frm_top, text=('FULL', 'AUTO', 'FIX'), command=set_normalize)
        self.__but_normalize.pack(side=tk.RIGHT)

        def set_background(value):
            if value:
                self.cam.background = self.cam.acquire()
            else:
                self.cam.background = None
        self.__but_background = SwitchButton(frm_top, text='DARK', command=set_background)
        self.__but_background.pack(side=tk.RIGHT)

        frm_top.pack(side=tk.TOP, fill=tk.X, expand=True)

        # Next row of buttons
        frm_bottom = Frame(master=self.__frm)

        home = str(Path.home())
        self.__cb_file = Combobox(frm_bottom, value=self.settings.file_name, values=[home],
                                  command=self.__update_file, width=60)
        self.__cb_file.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(frm_bottom.tk_container, text='Folder', command=self.__select_file).pack(side=tk.LEFT)
        ttk.Button(frm_bottom.tk_container, text='Record', command=self.__record_to_file).pack(side=tk.LEFT)

        frm_bottom.pack(side=tk.TOP, fill=tk.X, expand=True)

        self.__frm.pack(side=tk.TOP, fill=tk.X)

        self.__but_normalize.toggle_state = self.settings.normalize
        self.normalize = self.settings.normalize

        self.__update_cam(self.__dd_cam)

    @property
    def cam(self):
        return self.__cam

    @property
    def magnification(self) -> float:
        return self.__settings.cam.magnification

    @magnification.setter
    def magnification(self, value: float):
        self.__settings.cam.magnification = value

    @property
    def normalize(self) -> int:
        return self.__settings.normalize

    @normalize.setter
    def normalize(self, value: int):
        self.__settings.normalize = value

    @property
    def recording(self):
        return self.__recording

    @recording.setter
    def recording(self, value):
        self.__recording = value

    @property
    def file_name(self) -> str:
        return self.__settings.file_name

    @property
    def settings(self):
        return self.__settings

    @property
    def background_button_switch_state(self) -> bool:
        return self.__but_background.switch_state

    @background_button_switch_state.setter
    def background_button_switch_state(self, switch_state: bool):
        self.__but_background.switch_state = switch_state

    def __set_toggle_state_normalize(self, new_state: int):
        self.__but_normalize.toggle_state = new_state
        self.normalize = self.__but_normalize.toggle_state

    def __update_cam(self, widget):
        cam_name = widget.value

        def set_cam(cam_name):
            self.__cam = self.__monitor.get_cam(cam_name)
            self.__settings.cam.name = cam_name
            self.__update_exposure_time(self.__cb_exposure_time)
            self.__update_gain(self.__cb_gain)

        try:
            set_cam(cam_name)
        except Exception as e:
            set_cam(self.__monitor.available_cam_names[0])

        self.__frm.title = f'{cam_name} | Monitor GUI - v{__version__}'

    def __update_magnification(self, widget):
        magnification_str = widget.value
        try:
            magnification = float(re.match("^-?\d*\.?\d*", magnification_str)[0])
            log.info(f"Setting magnification to {magnification}.")
            self.__settings.cam.magnification = magnification
            return True
        except ValueError:
            log.warning(f"Could not set magnification to '{magnification_str}'.")
            return False

    def __update_exposure_time(self, widget):
        exposure_time_str = widget.value
        try:
            exposure_time = float(exposure_time_str) * 1e-3
            log.info(f"Setting exposure time to {exposure_time}s.")
            self.cam.exposure_time = exposure_time
            self.__settings.cam.exposure_time = exposure_time_str
            return True
        except ValueError:
            log.warning(f"Could not set exposure time to '{exposure_time_str}s'.")
            return False

    def __update_gain(self, widget):
        gain_str = widget.value
        try:
            gain = float(gain_str)
            log.info(f"Setting gain to {gain:0.1f}.")
            self.cam.gain = gain
            self.__settings.cam.gain = gain_str
            return True
        except ValueError:
            log.warning(f"Could not set gain to '{gain_str}'.")
            return False

    def __update_file(self, widget):
        if self.__recording:
            self.recording = False
        value = widget.value
        self.__settings.file_name = value
        return True

    def __select_file(self):
        value = filedialog.askdirectory(initialdir=Path.cwd(),
                                        title="Select recording location and format",
                                        )
        if value is not None:
            self.__cb_file.value = value
            return True
        else:
            return False

    def __record_to_file(self):
        self.__recording = not self.__recording
        if self.__recording:
            log.info(f'Recording to {self.file_name}...')  # TODO


class CamView(View):
    def __init__(self, monitor, settings: CamViewSettings=CamViewSettings()):
        super().__init__(monitor)

        self.__max_value_to_display = 1.0

        # Set up the camera control toolbar
        self.cam_control = CamControl(self._window, monitor, settings=settings)

        self.__video_writer = None

        # Define the canvas
        def get_roi():
            """
            A callback to get the region of interest
            """
            return self.cam_control.cam.roi

        def set_roi(new_roi):
            """
            A callback to set the region of interest
            """
            self.cam_control.cam.roi = new_roi
            self.cam_control.background_button_switch_state = self.cam_control.cam.background is not None

        def get_pixel_unit():
            value = self.cam_control.cam.pixel_pitch[1]
            if not np.isclose(self.cam_control.magnification, 0.0):
                value /= self.cam_control.magnification
            else:
                log.info(f'Ignoring magnification of {self.cam_control.magnification}.')
            return value, 'm', self.cam_control.cam.shape / 2.0

        self._canvas = DragZoomCanvas(container=self._window, get_roi_callback=get_roi, set_roi_callback=set_roi,
                                      get_pixel_unit_callback=get_pixel_unit)\
            .pack(fill=tk.BOTH, side=tk.TOP, expand=True)

        # Complete the window's initialization
        self._complete_initialization(settings)
        log.debug('Cam View initialized.')

    def _get_image(self):
        try:
            image_array = self.cam_control.cam.acquire()
        except CamError as ce:
            raise ViewError(ce)

        # Check for saturation
        if self.cam_control.cam.background is not None:
            saturated = image_array + self.cam_control.cam.background >= 1.0
        else:
            saturated = image_array >= 1.0
        if saturated.ndim > 2:
            saturated = np.any(saturated, axis=2)
        # Normalize as needed
        if self.cam_control.normalize > 0:
            if self.cam_control.normalize == 1:
                self.__max_value_to_display = np.maximum(np.amax(image_array), 2.0**(-24))
            image_array = np.clip(image_array / self.__max_value_to_display, 0.0, 1.0)
        else:
            self.__max_value_to_display = 1.0
        # Convert to uint8
        image_array = (image_array * 255.0).astype(np.uint8)
        # Mark saturation
        if image_array.ndim < 3:
            image_array = np.repeat(image_array[:, :, np.newaxis], repeats=3, axis=2)
        if np.any(saturated):
            image_array[saturated, 0] = 255
            image_array[saturated, 1:] = 0

        if self.cam_control.recording:
            # This should push the image on a queue and delegate to another thread
            file_path = self.cam_control.file_name.strip()
            if file_path == '':
                file_path = Path.cwd() / 'monitor_snapshot_*.png'
                log.info(f'No output path chosen, defaulting to {file_path}')
            else:
                file_path = Path(file_path)  # Convert to a Path object
            #
            file_path = Path(file_path.as_posix().replace('*', datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]))
            file_path.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
            extension = file_path.suffix[1:].lower()
            if extension in ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tif', 'tiff']:
                Image.fromarray(image_array).save(file_path, compress_level=9, quality=95, compression='tiff_deflate')
                log.info(f'Saved image to {file_path}.')
                self.cam_control.recording = False
            elif extension in ['mp4', 'avi', 'wmv', 'mpeg', 'webm']:
                # todo: fire up a thread to handle this with a context manager
                if self.__video_writer is None:
                    log.debug(f'Assuming that {extension} is the extension of video {file_path}.')
                    codecs = {'mp4': 'mp4v', 'avi': 'fmp4', 'wmv': 'm4s2', 'webm': 'vp80'}
                    fourcc = cv2.VideoWriter_fourcc(*codecs[extension]) if extension in codecs else -1
                    frame_rate = int(np.maximum(1.0, 1.0 / self.cam_control.cam.frame_time))
                    output_size = tuple(np.asarray(image_array.shape)[[1, 0]])
                    self.__video_writer = cv2.VideoWriter(str(file_path), fourcc, frame_rate, output_size)
                    log.info(f'Recording to {file_path} at {frame_rate} frames per second...')

                # convert colors from BGR to RGB and  write the frame
                self.__video_writer.write(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        else:
            if self.__video_writer is not None:
                self.__video_writer.release()
                self.__video_writer = None

        return image_array

    @property
    def settings(self):
        settings = self.cam_control.settings
        settings.window_roi = self.roi
        return settings

    def close(self):
        if self.__video_writer is not None:
            self.__video_writer.release()
        super().close()
