import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter import ttk
import re
import numpy as np
import dataclasses
import numexpr as ne
import time
import queue
from datetime import datetime
from pathlib import Path
from scipy.io import savemat, loadmat
import logging

from .view import View, ViewSettings, ViewError
from optics import __version__
from optics.utils import Roi
from optics.instruments.slm import measure_aberration
from optics.calc import correction_from_pupil
from optics.calc import pupil_equation
from optics.utils.display import complex2rgb
from optics.gui import SwitchButton, Combobox, OptionMenu, Frame, DragZoomCanvas

log = logging.getLogger(__name__)


@dataclasses.dataclass
class SLMSettings:
    name: str = ''
    deflection_frequency: str = '1/10 1/10'
    two_pi_equivalent: str = '100'
    roi: Roi = Roi()
    modulation: str = '0'


@dataclasses.dataclass
class SLMViewSettings(ViewSettings):
    slm: SLMSettings = SLMSettings()
    file_name: str = ''


class SLMControl:
    def __init__(self, master, monitor, settings: SLMViewSettings):
        self.__monitor = monitor
        self.__settings = settings
        self.__measuring = False
        self.__display_roi = None
        self.__show_target_field = True
        self.__modulation_update_queue = queue.Queue(maxsize=10)
        self.__updating_correction = False

        if settings.slm.name == '':
            settings.slm.name = self.__monitor.available_slm_names[0]
        self.__slm = self.__monitor.get_slm(settings.slm.name)

        self.__frm = Frame(master)
        frm_top = Frame(self.__frm)
        slm_names = self.__monitor.available_slm_names
        self.__dd_slm = OptionMenu(frm_top, value=self.__settings.slm.name, values=('', *slm_names),
                                   width=32, command=self.__update_slm)
        self.__dd_slm.pack(side=tk.LEFT)
        self.__cb_2pi = Combobox(frm_top, value=self.__settings.slm.two_pi_equivalent, width=4,
                                 values=[f"{pct:0.0f}" for pct in range(100, 20, -5)],
                                 command=self.__update_two_pi_equivalent)
        self.__cb_2pi.pack(side=tk.LEFT)
        self.__cb_deflection = Combobox(frm_top, width=16, value=self.__settings.slm.deflection_frequency,
                                        values=["0.05 0.05", "0.10 0.10", "0.20 0.20", "0.10 0.10 0.00001"],
                                        command=self.__update_deflection)
        self.__cb_deflection.pack(side=tk.LEFT)
        ttk.Label(frm_top.tk_container, text='/px ').pack(side=tk.LEFT)
        self.__cb_modulation = Combobox(frm_top, value=self.__settings.slm.modulation,
                                        values=["r < 100",
                                                "(r<100) * exp(2i*pi * 5 * ((x/100)^3 + (y/100)^3))",
                                                "(r<100) & (r>=90)",
                                                "(r<100) * exp(i * (phi - t))",
                                                "(r<100) & ((abs(r*cos(p-t*pi/10)) < 2) | (abs(r*sin(p-t*pi/10)) < 2))",
                                                "(r<100) & ((t - floor(t)) < 0.5)",
                                                "1",
                                                "0"],
                                        width=60,command=self.__update_modulation
                                        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        frm_top.pack(side=tk.TOP)

        frm_bottom = Frame(master=self.__frm)

        self.__cb_file = Combobox(frm_bottom, value="", width=40, command=self.__update_file)
        self.__cb_file.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(frm_bottom.tk_container, text='Folder', command=self.__select_file).pack(side=tk.LEFT)
        self.__but_measuring = SwitchButton(frm_bottom, text=('Measure', 'Measuring'), command=self.__measure_correction)
        self.__but_measuring.pack(side=tk.LEFT)

        def set_show_target_field(status):
            self.__show_target_field = not status
        self.__but_target_field = SwitchButton(frm_bottom, text=('Target', 'Actual'), command=set_show_target_field)
        self.__but_target_field.pack(side=tk.LEFT)
        frm_bottom.pack(side=tk.TOP, fill=tk.X, expand=True)

        self.__frm.pack(side=tk.TOP)

        self.__modulation_function = lambda x, y, t, w: 0.0 * (x + y)
        self.__update_slm(self.__dd_slm)

        self.__start_time = time.perf_counter()
        log.info("SLMControl initialized.")

    @property
    def slm(self):
        return self.__slm

    @property
    def measuring(self):
        return self.__measuring

    @measuring.setter
    def measuring(self, value):
        self.__measuring = value

    @property
    def file_name(self) -> str:
        return self.__settings.file_name

    @property
    def settings(self) -> ViewSettings:
        return self.__settings

    def __measure_correction(self, start_stop=True):
        # Variables that should be settable from the GUI:
        probe_shape = (15, 15, 3)
        average_shape = np.array([1, 1])
        attenuation_limit = 2
        aberration_file_name = f'aberration_measurement_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f%z")}.mat'

        if start_stop:
            log.info('Calculating correction...')
            self.slm.modulate(0)
            self.__monitor.cam.background = self.__monitor.cam.acquire()  # todo: This is ignored after roi changes
            self.slm.modulate(1)
            cam = self.__monitor.cam
            original_roi = cam.roi
            cam.center_roi_around_peak_intensity()
            average_shape = np.minimum(average_shape, cam.roi.shape)

            log.warning('Measuring correction... (not working yet!)')

            def measurement_function():
                time.sleep(0.100)
                img = cam.acquire()
                img = cam.acquire()
                img = img[img.shape[0]//2 + np.arange(average_shape[0]) - average_shape[0] // 2,
                          img.shape[0]//2 + np.arange(average_shape[1]) - average_shape[1] // 2]
                img = img.astype(np.float)
                return np.mean(img.ravel())

            last_update_time = [-np.infty]  # Use a list so that it is not a number but an object

            def progress_callback(fraction, pupil):
                current_time = time.perf_counter()
                if current_time >= last_update_time[0] + 5.0:
                    log.info(f'Measuring aberration: {fraction*100:0.1f}%')
                    last_update_time[0] = current_time

            pupil_function = measure_aberration(self.slm, measurement_function, measurement_grid_shape=probe_shape,
                                                progress_callback=progress_callback)
            self.__but_measuring.switch_state = False
            aberration_fullfile = Path.home() / aberration_file_name
            log.info(f'Storing aberration and correction to {aberration_fullfile.as_posix()}...')
            correction = correction_from_pupil(pupil_function, attenuation_limit=attenuation_limit)
            savemat(aberration_fullfile.as_posix(), {'amplificationLimit': attenuation_limit,
                                                     'centerPos': [*cam.roi.top_left, *cam.roi.shape],
                                                     'measuredPupilFunction': pupil_function,
                                                     'initialCorrection': np.ones([1,1], dtype=np.complex),
                                                     'pupilFunctionCorrection': correction,
                                                     'referenceDeflectionFrequency': self.slm.deflection_frequency,
                                                     'slmRegionOfInterest': [self.slm.roi.top_left, self.slm.roi.shape],
                                                     'twoPiEquivalent': self.slm.two_pi_equivalent})
            # Update GUI and SLM
            self.__cb_file.value = aberration_fullfile.as_posix()
            self.__update_file(self.__cb_file)
        else:
            log.info('Stopping calculation of correction.')

    def __update_slm(self, widget):
        slm_name = widget.value
        if slm_name is None:
            slm_name = self.__monitor.available_slm_names[0]

        self.__slm = self.__monitor.get_slm(slm_name)
        self.set_display_roi()
        self.__settings.slm.name = slm_name
        self.__update_two_pi_equivalent(self.__cb_2pi)
        self.__update_deflection(self.__cb_deflection)
        self.__update_file(self.__cb_file)
        self.__update_modulation(self.__cb_modulation)

        self.__frm.title = f'{slm_name} | Monitor GUI - v{__version__}'

    def __update_two_pi_equivalent(self, widget):
        two_pi_equivalent = widget.value
        try:
            self.slm.two_pi_equivalent = float(two_pi_equivalent) / 100.0
            self.__settings.slm.two_pi_equivalent = two_pi_equivalent
            self.__update_modulation(self.__cb_modulation)
            return True
        except ValueError:
            return False

    def __update_deflection(self, widget):
        deflection_str = widget.value
        deflection = np.array([float(ne.evaluate(s.strip(' '), local_dict=dict()))
                               for s in re.sub(r"[,;|]+", ' ', deflection_str).split(' ') if s != ''])
        log.info(f"Setting deflection frequency to {deflection} [px].")
        self.slm.deflection_frequency = deflection
        self.__settings.slm.deflection_frequency = deflection_str
        self.__update_modulation(self.__cb_modulation)
        return True

    def __update_file(self, widget):
        if self.__measuring:
            self.recording = False
        self.__settings.file_name = widget.value.strip()
        if len(self.__settings.file_name) == 0:
            self.slm.correction = 1
            log.info('Not doing any aberration correction with SLM.')
        else:
            try:
                fullfilepath = Path(self.__settings.file_name)
                mdict = loadmat(fullfilepath.as_posix())
                # {'amplificationLimit': attenuation_limit,
                #  'centerPos': [*cam.roi.top_left, *cam.roi.shape],
                #  'measuredPupilFunction': pupil_function,
                #  'initialCorrection': np.ones([1,1], dtype=np.complex),
                #  'pupilFunctionCorrection': correction,
                #  'referenceDeflectionFrequency': self.slm.deflection_frequency,
                #  'slmRegionOfInterest': [self.slm.roi.top_left, self.slm.roi.shape],
                #  'twoPiEquivalent': self.slm.two_pi_equivalent}
                self.slm.correction = mdict['pupilFunctionCorrection']
                log.info(f'Loaded aberration correction pattern on SLM from {fullfilepath.as_posix()}.')
            except FileNotFoundError:
                log.info(f'Could not load aberration from file "{fullfilepath.as_posix()}".')
        self.__update_modulation(self.__cb_modulation)

    def __update_modulation(self, widget):
        modulation_str = widget.value
        try:
            new_function = pupil_equation.parse(modulation_str)
            self.__modulation_function = new_function
            try:
                self.__modulation_update_queue.put(True, block=False)
            except queue.Full:
                pass
            self.__settings.slm.modulation = modulation_str
            log.info(f'Updated pupil equation to "{modulation_str}"')
        except pupil_equation.ParseError as err:
            log.warning(f'Error in pupil equation "{modulation_str}"')

    def set_display_roi(self, new_display_roi: Roi=None):
        if new_display_roi is None:
            new_display_roi = self.__monitor.slm.roi
        else:
            new_display_roi.dtype = np.int
        self.__display_roi = new_display_roi

    def get_display_roi(self) -> Roi:
        return self.__display_roi

    def update(self):
        try:
            self.__modulation_update_queue.put(True, block=False)
        except queue.Full:
            pass
            
    def modulate_and_return(self):
        if self.__modulation_function is None:
            log.error('self.__modulation_function is None')
        updated = self.__modulation_update_queue.get()
        if updated:
            self.__monitor.slm.modulate(  # can take > 1/3 s
                lambda y, x: self.__modulation_function(y, x, t=time.perf_counter()-self.__start_time, w=1)
            )
            if self.__modulation_function.time_dependent:
                # Continuously update the modulation
                try:
                    self.__modulation_update_queue.put(True, block=False)
                except queue.Full:
                    pass
            if self.__show_target_field:
                img = complex2rgb(self.__monitor.slm.complex_field, dtype=np.uint8)
            else:
                img = self.__monitor.slm.image_on_slm
            # Crop the region of interest out of the image
            ranges = self.__display_roi.grid
            if np.all(np.asarray(img.shape[:2]) > 1):
                log.info(f'img.shape = {img.shape}, ranges={ranges}')
                img = img[np.clip(ranges[0], 0, img.shape[0]-1), np.clip(ranges[1], 0, img.shape[1]-1)]
        else:
            img = None
            log.info('Modulation not updated, exiting without an image.')
        return img

    def __select_file(self):
        value = filedialog.askopenfilename(initialdir=Path.home(), title="Select correction file",
                                           filetypes=(("Mat files", "*.mat"), ("all files", "*.*")))
        if value is not None:
            self.__cb_file.value = value
            self.__update_file(self.__cb_file)

    def close(self):
        log.debug('Unblock modulate_and_return if necessary....')
        try:
            self.__modulation_update_queue.put(False, block=False)
        except queue.Full:
            pass
        log.debug('Unblocked modulate_and_return.')


class SLMView(View):
    def __init__(self, monitor, settings):
        self.__complex_field = 0
        super().__init__(monitor)

        self.slm_control = SLMControl(self._window, monitor=monitor, settings=settings)
        log.debug('SLMControl initialized.')

        # Define the canvas so that it draws the picture provided by the SLMView
        self._canvas = DragZoomCanvas(container=self._window,
                                      get_roi_callback=self.slm_control.get_display_roi,
                                      set_roi_callback=self.slm_control.set_display_roi,
                                      get_pixel_unit_callback=lambda: (self.slm_control.slm.pixel_pitch[0], 'm', self.slm_control.slm.shape / 2.0))\
            .pack(fill=tk.BOTH, side=tk.TOP, expand=True)
        log.debug('SLM Canvas initialized.')

        # Complete the window's initialization
        self._complete_initialization(settings)
        log.debug('SLM View initialized.')

    def _update_window(self):
        self.slm_control.update()
        super()._update_window()

    def _get_image(self):
        start_time = time.perf_counter()
        try:
            img = self.slm_control.modulate_and_return()
        except pupil_equation.ParseError as parse_error:
            raise ViewError(parse_error)
        finally:
            elapsed_time = time.perf_counter() - start_time
            wait_time = 0.040 - elapsed_time
            if wait_time > 0:
                time.sleep(wait_time)

        return img

    @property
    def settings(self):
        settings = self.slm_control.settings
        settings.window_roi = self.roi
        return settings

    def close(self):
        self.slm_control.close()
        super().close()
