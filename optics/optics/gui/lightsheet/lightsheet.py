#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import tkinter as tk
from tkinter import ttk
import re
import jsonpickle
import os
import numpy as np
import dataclasses
from contextlib import AbstractContextManager

from optics import log, __version__
# from optics.gui.lightsheet.view import ViewSettings
# from optics.gui.lightsheet.stage_view import StageViewSettings
from optics.instruments.cam import Cam, SimulatedCam
from optics.instruments.stage import Stage, SimulatedStage, ThorlabsKinesisStage
from optics.gui.key_descriptor import KeyDescriptor
from optics.gui.element import SwitchButton, Combobox, OptionMenu
from optics.utils.array import MutableGrid


@dataclasses.dataclass
class Settings:
    version: str = __version__


class LightSheet(AbstractContextManager):
    def __init__(self):
        self.__settings_full_file_name = os.path.splitext(__file__)[0] + '_settings.json'
        self.__settings = self.load_settings()
        # Build the GUI
        self.__window = tk.Tk()
        self.__full_screen = False
        # Add event handlers
        self.__window.protocol("WM_DELETE_WINDOW", self.close)
        # self.__window.bind('<Key>', self.__on_key)
        # # Add menubar
        # menu_bar = tk.Menu(self.__window)
        # # # The File menu
        # menu_file = tk.Menu(menu_bar, tearoff=False)
        # # # # The New menu
        # menu_new = tk.Menu(menu_file)
        # # # # # The Cam menu
        # menu_cam = tk.Menu(menu_new)

        self.__cam = SimulatedCam()
        self.__stage: Stage = SimulatedStage()  #ThorlabsKinesisStage()

        self.__scan = MutableGrid(step=1e-6, shape=100)

        self.__frm_stage = ttk.Frame(self.__window)
        self.__frm_stage_step = ttk.Frame(self.__frm_stage)
        ttk.Label(self.__frm_stage_step, text='Step size (dz):').pack(side=tk.LEFT)
        self.__cb_step = Combobox(self.__frm_stage_step, width=32, value=self.__scan.step[0] * 1e6,
                                  values=['0.100', '0.200', '0.500', '1.000', '2.000', '5.000', '10.000', '20.000', '50.000'],
                                  command=self.__update_step).pack(side=tk.LEFT)
        ttk.Label(self.__frm_stage_step, text='μm').pack(side=tk.LEFT)
        self.__frm_stage_step.pack(side=tk.TOP)
        self.__frm_stage_extent = ttk.Frame(self.__frm_stage)
        ttk.Label(self.__frm_stage_extent, text='Scan extent (∆z):').pack(side=tk.LEFT)
        self.__cb_extent = Combobox(self.__frm_stage_extent, width=32, value=self.__scan.extent[0] * 1e6,
                                    values=['0.100', '0.200', '0.500', '1.000', '2.000', '5.000', '10.000', '20.000', '50.000'],
                                    command=self.__update_extent).pack(side=tk.LEFT)
        ttk.Label(self.__frm_stage_extent, text='μm').pack(side=tk.LEFT)
        self.__frm_stage_extent.pack(side=tk.TOP)
        self.__frm_stage_shape = ttk.Frame(self.__frm_stage)
        ttk.Label(self.__frm_stage_shape, text='Number of slices (#z):').pack(side=tk.LEFT)
        self.__cb_shape = Combobox(self.__frm_stage_shape, width=32, value=self.__scan.shape[0],
                                   values=['1', '2', '5', '10', '20', '50', '100', '200', '500', '1000'],
                                   command=self.__update_shape).pack(side=tk.LEFT)
        self.__frm_stage_shape.pack(side=tk.TOP)
        self.__frm_stage.pack(side=tk.TOP)

        self.__open = True

        log.info("LightSheet GUI initialized =================================================")

    def __update_step(self, widget: Combobox):
        self.__scan.step = float(widget.value)
        self.__update_display()

    def __update_extent(self, widget: Combobox):
        extent = float(widget.value)
        self.__scan.step = extent / self.__scan.shape
        self.__update_display()

    def __update_shape(self, widget: Combobox):
        self.__scan.shape = float(widget.value)
        self.__update_display()

    def __update_display(self):
        pass

    def __enter__(self):
        return self

    def __call__(self):
        log.info("LightSheet GUI started =====================================================")
        self.__window.mainloop()
        self.close()
        log.info('LightSheet GUI terminated =====================================================')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            log.warning('Exception while using LightSheet object: ' + str(exc_type))
            raise exc_val
        self.__del__()
        return True

    def __del__(self):
        self.close()

    @property
    def cam(self) -> Cam:
        return self.__cam

    @property
    def stage(self) -> Stage:
        return self.__stage

    def close(self):
        if self.__open:
            log.info('Shutting down now...')
            self.save_settings()

            log.info('Closing GUI Window...')
            self.__window.quit()
            self.__window.destroy()

            log.info('Closing instruments...')
            self.__cam.disconnect()
            self.__stage.disconnect()

            self.__open = False

    @property
    def full_screen(self):
        return self.__full_screen

    @full_screen.setter
    def full_screen(self, value):
        self.__window.attributes("-fullscreen", value)
        self.__window.overrideredirect(True)
        self.__full_screen = value

    def __on_key(self, event):
        key_desc = KeyDescriptor(event)
        log.debug(f'{key_desc} key pressed.')

        if key_desc.control:
            if key_desc == 'Q':  # Ctrl-Q
                self.__window.close()
            # elif key_desc == 'C':  # Ctrl-C
            #     self.close()
            elif key_desc == 'F':  # F
                self.full_screen = not self.full_screen
            elif key_desc == 'M':  # Ctrl-M
                log.debug("Toggle Maximize")
                self.full_screen = False
                if self.__window.state() == 'normal':
                    try:
                        self.__window.state('zoomed')
                    except Exception as exc:
                        log.info(f"Cannot maximize window using 'zoomed' ({exc}), setting maxsize() instead.")
                        width = self.__window.winfo_screenwidth()
                        height = self.__window.winfo_screenheight()
                        self.__window.geometry(f'{width}x{height}+0+0')
                else:
                    self.__window.state('normal')
                self.__window.update()
            elif key_desc == 'L':  # Ctrl-L
                self.full_screen = False
                self.__window.state('normal')
                width = int(self.__window.winfo_screenwidth() / 2)
                height = self.__window.winfo_screenheight()
                self.__window.geometry(f'{width}x{height}+0+0')
                self.__window.update()
            elif key_desc == 'R':  # Ctrl-R
                self.full_screen = False
                self.__window.state('normal')
                width = int(self.__window.winfo_screenwidth() / 2)
                height = self.__window.winfo_screenheight()
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
            self.__window.state('normal')

    def load_settings(self) -> Settings:
        settings = Settings()
        if os.path.isfile(self.__settings_full_file_name):
            log.info(f"Loadings settings from {self.__settings_full_file_name}...")
            with open(self.__settings_full_file_name) as settings_file:
                settings_str = settings_file.read()
            try:
                settings = jsonpickle.loads(settings_str)
            except Exception as e:
                log.warning(f'Could not interpret settings file {self.__settings_full_file_name}, ignoring it.\nError: {e}.')
        else:
            log.info(f"Settings file {self.__settings_full_file_name} not found, using defaults.")

        return settings

    def save_settings(self):
        # Set version number to current version
        self.__settings.version = __version__

        jsonpickle.set_encoder_options('json', sort_keys=False, indent=2)
        settings_str = jsonpickle.dumps(self.__settings)
        with open(self.__settings_full_file_name, 'w') as settings_file:
            settings_file.write(settings_str)

        log.info(f"Saved configuration to {self.__settings_full_file_name}.")


if __name__ == "__main__":
    LightSheet()()
