from __future__ import annotations

import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter import ttk
import re
import os
import numpy as np
import dataclasses
import numexpr as ne
import time
import queue
import logging

from .view import View, ViewSettings, ViewError
from optics import __version__
from optics.utils import Roi, array
from optics.utils.ft import Grid
from optics.utils.display import hsv2rgb
from optics.calc import pupil_equation, zernike
from optics.utils.display import complex2rgb
from optics.gui import SwitchButton, Combobox, OptionMenu, Frame, DragZoomCanvas

log = logging.getLogger(__name__)


@dataclasses.dataclass
class DMSettings:
    name: str = ''
    deflection: str = '0 0'
    wavelength: str = '532'
    modulation: str = '0'


@dataclasses.dataclass
class DMViewSettings(ViewSettings):
    dm: DMSettings = DMSettings()
    file_name: str = ''


class DMControl:
    def __init__(self, master, monitor, settings: DMViewSettings):
        self.__monitor = monitor
        self.__settings = settings
        self.__measuring = False
        self.__display_roi = None
        self.__show_target_actual_stroke = 0
        self.__modulation_update_queue = queue.Queue(maxsize=1)
        self.__updating_correction = False

        if settings.dm.name == '':
            settings.dm.name = self.__monitor.available_dm_names[0]
        self.__dm = self.__monitor.get_dm(settings.dm.name)

        self.__frm = Frame(master)
        frm_top = Frame(self.__frm)
        dm_names = self.__monitor.available_dm_names
        self.__dd_dm = OptionMenu(frm_top, value=self.__settings.dm.name, values=('', *dm_names),
                                  width=32, command=self.__update_dm).pack(side=tk.LEFT)
        self.__cb_wavelength = Combobox(frm_top, value=self.__settings.dm.wavelength, width=6,
                                        values=[f"{pct:0.1f}" for pct in [405, 488.0, 509, 532, 632.8, 1064]],
                                        command=self.__update_wavelength).pack(side=tk.LEFT)
        ttk.Label(frm_top.tk_container, text="nm ").pack(side=tk.LEFT)
        self.__cb_deflection = Combobox(frm_top, width=16, value=self.__settings.dm.deflection,
                                        values=["0.05 0.05", "0.10 0.10", "0.20 0.20", "0.10 0.10 0.00001"],
                                        command=self.__update_deflection).pack(side=tk.LEFT)
        ttk.Label(frm_top.tk_container, text=u'λ ').pack(side=tk.LEFT)
        self.__cb_modulation = Combobox(frm_top, value=self.__settings.dm.modulation,
                                        values=["5 * r^2",
                                                "5 * x^2",
                                                "5 * (x^3 + y^3)"],
                                        width=60, command=self.__update_modulation
                                        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        frm_top.pack(side=tk.TOP)

        frm_bottom = Frame(master=self.__frm)

        self.__cb_file = Combobox(frm_bottom, value="", width=40, command=self.__update_file)
        self.__cb_file.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(frm_bottom.tk_container, text='Folder', command=self.__select_file).pack(side=tk.LEFT)
        self.__but_measuring = SwitchButton(frm_bottom, text=('Measure', 'Measuring'),
                                            command=self.__measure_correction).pack(side=tk.LEFT)

        def set_show_target_actual_stroke(status):
            self.__show_target_actual_stroke = status
        self.__but_target_field = SwitchButton(frm_bottom, text=('Target', 'Actual', 'Stroke'),
                                               command=set_show_target_actual_stroke)
        self.__but_target_field.pack(side=tk.LEFT)
        frm_bottom.pack(side=tk.TOP, fill=tk.X, expand=True)

        self.__frm.pack(side=tk.TOP)

        self.__modulation_function = lambda x, y, t, w: 0.0 * (x + y)
        self.__update_dm(self.__dd_dm)

        self.__start_time = time.perf_counter()
        log.info("DMControl initialized.")

    @property
    def dm(self):
        return self.__dm

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
        if start_stop:
            log.info('Calculating correction...')
            self.dm.modulate(0)
            self.__monitor.cam.background = self.__monitor.cam.acquire()  # todo: This is ignored after roi changes
            self.dm.modulate(1)
            self.__monitor.cam.center_roi_around_peak_intensity()

            log.warning('Measuring correction... (NOT YET IMPLEMENTED!)')  # todo implement
        else:
            log.info('Stopping calculation of correction.')

    def __update_dm(self, widget):
        dm_name = widget.value
        if dm_name is None:
            dm_name = self.__monitor.available_dm_names[0]

        self.__dm = self.__monitor.get_dm(dm_name)
        self.__canvas_image_shape = np.full(2, 101)
        self.set_display_roi()
        self.__settings.dm.name = dm_name
        self.__update_wavelength(self.__cb_wavelength)
        self.__update_deflection(self.__cb_deflection)
        self.__update_file(self.__cb_file)
        self.__update_modulation(self.__cb_modulation)

        self.__frm.title = f'{dm_name} | Monitor GUI - v{__version__}'

    def __update_wavelength(self, widget):
        wavelength = widget.value
        try:
            self.dm.wavelength = float(wavelength) * 1e-9
            self.__settings.dm.wavelength = wavelength
            self.__update_modulation(self.__cb_modulation)
            return True
        except ValueError:
            return False

    def __update_deflection(self, widget):
        deflection_str = widget.value
        coeffs = np.array([float(ne.evaluate(s.strip(' '), local_dict=dict()))
                               for s in re.sub(r"[,;|]+", ' ', deflection_str).split(' ') if s != ''])
        coeffs *= self.dm.wavelength
        coeffs = array.pad_to_length(coeffs, 3)
        log.info(f'Setting deformable mirror deflection to {coeffs * 1e-6}um.')

        self.dm.deflection = lambda ny, nx: coeffs[0] * zernike.tip.cartesian(ny, nx) \
                                            + coeffs[1] * zernike.tilt.cartesian(ny, nx) \
                                            + coeffs[2] * zernike.defocus.cartesian(ny, nx)
        self.__settings.dm.deflection = deflection_str
        self.__update_modulation(self.__cb_modulation)
        return True

    def __update_file(self, widget):
        if self.__measuring:
            self.recording = False
        self.__settings.file_name = widget.value
        log.warn('Correction file loading not yet implemented!')   # TODO
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
            self.__settings.dm.modulation = modulation_str
            log.info(f'Updated pupil equation to "{modulation_str}"')
        except pupil_equation.ParseError as err:
            log.warning(f'Error in pupil equation "{modulation_str}"')

    def set_display_roi(self, new_display_roi: Roi=None):
        if new_display_roi is None:
            new_display_roi = Roi(shape=self.__canvas_image_shape)
        else:
            new_display_roi.dtype = np.int
        self.__display_roi = new_display_roi

    def get_display_roi(self) -> Roi:
        return self.__display_roi

    def get_pixel_unit(self):
        value = self.dm.radius / (self.__canvas_image_shape[1] - 1)
        return value, "m", self.__canvas_image_shape / 2.0

    def update(self):
        try:
            self.__modulation_update_queue.put(True, block=False)
        except queue.Full:
            pass

    def modulate_and_return(self):
        if self.__modulation_function is None:
            log.error('self.__modulation_function is None')
        log.info('Waiting for modulation update...')
        updated = self.__modulation_update_queue.get()
        if updated:
            log.info('Modulation updated.')
            self.__monitor.dm.modulate(
                lambda y, x: self.__modulation_function(y, x,
                                                        t=time.perf_counter()-self.__start_time,
                                                        w=self.__monitor.dm.wavelength
                                                        ) * self.__monitor.dm.wavelength
            )
            if self.__modulation_function.time_dependent:
                # Continuously update the modulation
                try:
                    self.__modulation_update_queue.put(True, block=False)
                except queue.Full:
                    pass

            pupil_grid = Grid(shape=self.__canvas_image_shape, center=0, extent=2.0)
            if self.__show_target_actual_stroke == 0:
                dm_field = self.__monitor.dm.field(*pupil_grid, actual=False)
                img = complex2rgb(dm_field, dtype=np.uint8)  # convert complex to uint8
            elif self.__show_target_actual_stroke == 1:
                dm_field = self.__monitor.dm.field(*pupil_grid, actual=True)
                img = complex2rgb(dm_field, dtype=np.uint8)  # convert complex to uint8
            else:
                dm_stroke = self.__monitor.dm.stroke(*pupil_grid, actual=True)
                outside = np.isnan(dm_stroke)
                dm_stroke[outside] = 0
                inside = 1 - outside

                def near(section, width):
                    return np.exp(-0.5 * (abs(np.mod(dm_stroke / section + 0.5, 1) - 0.5) * section / width)**2)

                saturation = 1 - near(10e-6, 0.1e-6)
                hue = np.mod(dm_stroke / 1e-6, 1)
                value = inside * (1 - near(10e-6, 0.1e-6))
                img = hsv2rgb(hue, saturation, value)
                img = (img * 255.0 + 0.5).astype(np.uint8)
            # Crop the region of interest out of the image
            ranges = self.__display_roi.grid
            img = img[np.clip(ranges[0], 0, img.shape[0]-1), np.clip(ranges[1], 0, img.shape[1]-1), ...]
        else:
            img = None
            log.info('Modulation not updated, exiting without an image.')
        return img

    def __select_file(self):
        value = filedialog.asksaveasfilename(initialdir=os.getcwd(), title="Select correction file",
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


class DMView(View):
    def __init__(self, monitor, settings):
        self.__complex_field = 0
        super().__init__(monitor)

        self.dm_control = DMControl(self._window, monitor=monitor, settings=settings)
        log.debug('DMControl initialized.')

        # Define the canvas so that it draws the picture provided by the DMView
        self._canvas = DragZoomCanvas(container=self._window,
                                      get_roi_callback=self.dm_control.get_display_roi,
                                      set_roi_callback=self.dm_control.set_display_roi,
                                      get_pixel_unit_callback=self.dm_control.get_pixel_unit)\
            .pack(fill=tk.BOTH, side=tk.TOP, expand=True)
        log.debug('DM Canvas initialized.')

        # Complete the window's initialization
        self._complete_initialization(settings)
        log.debug('DM View initialized.')

    def _update_window(self):
        self.dm_control.update()
        super()._update_window()

    def _get_image(self):
        start_time = time.perf_counter()
        try:
            img = self.dm_control.modulate_and_return()
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
        settings = self.dm_control.settings
        settings.window_roi = self.roi
        return settings

    def close(self):
        self.dm_control.close()
        super().close()
