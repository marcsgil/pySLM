#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import re
import jsonpickle
import os
import numpy as np
import screeninfo
import dataclasses
from typing import Set
from contextlib import AbstractContextManager
import time
import logging

from optics import __version__
from optics.utils import ft
from optics.gui.container import Window
from optics.instruments.display import TkDisplay as ImplDisplay
from optics.gui.monitor.view import ViewSettings, View
from optics.gui.monitor.cam_view import CamView, CamViewSettings
from optics.gui.monitor.slm_view import SLMView, SLMViewSettings
from optics.gui.monitor.dm_view import DMView, DMViewSettings
from optics.instruments.cam import CamError, Cam
from optics.instruments.cam.simulated_cam import SimulatedCam
from optics.instruments.cam.web_cam import WebCam

log = logging.getLogger(__name__)

try:
    from optics.instruments.cam.ids_cam import IDSCam
except ImportError as exc:
    log.warning('Could not load IDSCam driver: ' + str(exc))
    IDSCam = SimulatedCam
from optics.instruments.slm import SLMError, SLM, PhaseSLM, DualHeadSLM
from optics.instruments.dm import DM, SimulatedDM
try:
    from optics.instruments.dm.alpao_dm import AlpaoDM
except Exception as e:
    log.error(e)


@dataclasses.dataclass
class Settings:
    version: str = __version__
    cam_view: CamViewSettings = CamViewSettings()
    slm_view: SLMViewSettings = SLMViewSettings()
    dm_view: DMViewSettings = DMViewSettings()


class Monitor(AbstractContextManager):
    def __init__(self):
        self.__settings_full_file_name = os.path.splitext(__file__)[0] + '_settings.json'
        self.__settings = self.load_settings()

        self.__opened_views: Set[View] = set()

        # Initialize the views
        self.__cam: Cam = None
        self.__cam_name: str = None
        self.__slm: SLM = None
        self.__slm_name: str = None
        self.__dm: DM = None
        self.__dm_name: str = None

        self.__opened = True

        log.info("Monitor GUI initialized =================================================")

    def __enter__(self):
        return self

    def __call__(self):
        log.info("Monitor GUI started =====================================================")
        # Open the default views
        log.debug('Opening Views...')
        # self.open_cam_viewer()
        # log.info('Opened Cam Viewer.')
        self.open_slm_viewer()
        log.info('Opened SLM Viewer.')
        # self.open_dm_viewer()
        # log.info('Opened DM Viewer.')

        Window.run()

        self.close()
        log.info('Monitor GUI terminated =====================================================')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            log.warning('Exception while using Monitor object: ' + str(exc_type))
            raise exc_val
        self.__del__()
        return True

    def __del__(self):
        if self.__opened:
            self.close()

    def open_cam_viewer(self, cam_name: str=None):
        if cam_name is not None:
            self.__settings.cam_view.cam.name = cam_name
        log.info(f'open_cam_viewer self.__settings.cam_view.cam.name={self.__settings.cam_view.cam.name}')
        new_view = CamView(self, self.__settings.cam_view)
        self.__opened_views.add(new_view)

    def open_slm_viewer(self, slm_name: str=None):
        if slm_name is not None:
            self.__settings.slm_view.slm.name = slm_name
        new_view = SLMView(self, self.__settings.slm_view)
        self.__opened_views.add(new_view)

    def open_dm_viewer(self, dm_name: str=None):
        if dm_name is not None:
            self.__settings.dm_view.dm.name = dm_name
        new_view = DMView(self, self.__settings.dm_view)
        self.__opened_views.add(new_view)

    def closed_window(self, v, settings: ViewSettings):
        if v in self.__opened_views:
            if isinstance(v, CamView):
                self.__settings.cam = settings
                log.info('Closed CAMView.')
            elif isinstance(v, SLMView):
                self.__settings.slm = settings
                log.info('Closed SLMView.')
            elif isinstance(v, DMView):
                self.__settings.dm = settings
                log.info('Closed DMView.')
            self.__opened_views.remove(v)
        else:
            log.warning(f'Closed View window {v} was not registered, this should not have happened!')
        if len(self.__opened_views) == 0:
            self.__shutdown()  # Close the application now too

    @property
    def available_cam_names(self):
        cams = ['Simulated Cam 0', 'Simulated Cam 1', 'Web Cam 0', 'Web Cam 1', 'iDS Cam 0', 'iDS Cam 1']
        return cams

    @property
    def available_slm_names(self):
        screens = screeninfo.get_monitors()
        screen_descriptors = ["{:d} [{:d}x{:d}]".format(idx + 1, s.width, s.height)  # Add one to the screen number
                              + (' MAIN' if (s.x == 0 and s.y == 0) else '')
                              for idx, s in enumerate(screens)]
        slm_descriptors = [f"{slm} SLM {scr}"
                           for slm in ('Phase', 'Dual Head') for scr in ('Simulated', *screen_descriptors)]
        return slm_descriptors

    @property
    def available_dm_names(self):
        cams = ['Simulated DM 69', 'Alpao BAX240']
        return cams

    @property
    def cam(self) -> Cam:
        return self.__cam

    @property
    def slm(self) -> SLM:
        return self.__slm

    @property
    def dm(self) -> DM:
        return self.__dm

    def get_cam(self, cam_name: str) -> Cam:
        if cam_name == self.__cam_name:
            return self.__cam
        else:
            log.info(f"Getting cam {cam_name}...")
        try:
            new_cam = None
            if re.search(r"^Web Cam", cam_name):
                cam_idx = int(re.search(r"\b(\d+)", cam_name).group())
                new_cam = WebCam(cam_idx)
            elif re.search(r"^iDS Cam", cam_name):
                cam_idx = int(re.search(r"\b(\d+)", cam_name).group())
                new_cam = IDSCam(cam_idx)
            elif re.search(r"^Sim", cam_name) or self.__cam is None:
                new_cam = SimulatedCam()
                new_cam.get_frame_callback = self.__get_test_image
            if new_cam is not None:
                new_cam.normalize = True
                if self.__cam is not None:
                    self.__cam.disconnect()
                self.__cam = new_cam
                self.__cam_name = cam_name
        except CamError as e:
            log.error(e)

        return self.__cam

    def get_slm(self, slm_name: str) -> SLM:
        if slm_name == self.__slm_name:
            return self.__slm
        else:
            log.info(f"Getting slm {slm_name}...")
        log.info(f"Setting slm_name to {slm_name}")
        try:
            new_slm = None
            slm_screen = re.search(r"\b(\d+)", slm_name)
            if slm_screen is not None:  # Convert to a fullscreen object
                slm_screen = int(slm_screen.group()) - 1  # subtract again
                try:
                    slm_screen = ImplDisplay(slm_screen)
                except ValueError as e:
                    log.warning(e)
                    slm_screen = None
            if re.search(r"^Phase SLM", slm_name):
                new_slm = PhaseSLM(slm_screen)
            elif re.search(r"^Dual Head SLM", slm_name):
                new_slm = DualHeadSLM(slm_screen)
                log.info("Switched to Dual Head SLM.")
            if new_slm is not None:
                if self.__slm is not None:
                    self.__slm.disconnect()

                self.__slm_name = slm_name
                self.__slm = new_slm
            # Add a waiting time for the SLM to update and the camera queue to clear
            self.__slm.post_modulation_callback = lambda _: time.sleep(100e-3)
        except SLMError as e:
            log.error(e)

        return self.__slm

    def get_dm(self, dm_name: str) -> DM:
        if dm_name == self.__dm_name:
            return self.__dm
        else:
            log.info(f"Getting Deformable Mirror {dm_name}...")
        log.info(f'Setting dm_name to {dm_name}')
        try:
            new_dm = None
            if re.search(r"^Alpao", dm_name):
                serial_name = dm_name.split()[1]
                log.info(f'Switching to Alpao DM {serial_name}...')
                new_dm = AlpaoDM(serial_name=serial_name)
                log.info(f'Switched to Alpao DM {serial_name}.')
            elif re.search(r'^Sim', dm_name) or self.__dm is None:
                nb_actuators = int(re.search(r'\d+', dm_name)[0])
                new_dm = SimulatedDM(nb_actuators=nb_actuators, radius=5e-3, max_stroke=30e-6)
                log.info('Switched to Simulated DM.')
            if new_dm is not None:
                if self.__dm is not None:
                    self.__dm.disconnect()

                self.__dm_name = dm_name
                self.__dm = new_dm
        except SLMError as e:
            log.error(e)

        return self.__dm

    def close(self):
        log.debug('Closing all View windows...')
        for view in list(self.__opened_views):
            log.debug(f'Closing {view}...')
            view.close()
            log.debug(f'Closed {view}.')
        self.__opened = False

    def __shutdown(self):
        log.debug('Shutting down now...')
        self.save_settings()

        if self.__cam is not None:
            self.__cam.disconnect()
            self.__cam = None
        if self.__slm is not None:
            self.__slm.disconnect()
            self.__slm = None

    def __get_test_image(self, cam):
        if self.slm is not None:
            slm_image = self.slm.image_on_slm
        else:
            slm_image = np.zeros(shape=(1, 1))
        slm_bit_depth = 8
        nb_gray_levels = 2**slm_bit_depth
        well_depth = 40e3
        dark_photon_electrons = 10.0
        photo_electrons_per_gray_level = well_depth / nb_gray_levels

        slm_image = np.array(slm_image, dtype=np.float) / (nb_gray_levels - 1)
        if slm_image.ndim >= 3:
            # 3 channel, could be a Dual Head SLM
            pupil = slm_image[:, :, 2] * np.exp(2j * np.pi * slm_image[:, :, 1])  # normalized in amplitude
        else:
            # monochrome SLM image
            pupil = np.exp(2j * np.pi * slm_image)

        # log.debug(f"Pupil shape {pupil.shape} and values {np.min(np.abs(pupil.flatten()))}-{np.max(np.abs(pupil.flatten()))}")

        # Calculate the simulated image with a Fourier transform
        field_at_camera = ft.fftshift(ft.ifft2(ft.ifftshift(pupil[::-1, ::-1])))
        img = np.abs(field_at_camera)**2

        # # select only pixels in the region-of-interest
        # img = img[np.clip(cam.roi.grid[0], 0, img.shape[0]-1).astype(int),
        #           np.clip(cam.roi.grid[1], 0, img.shape[1]-1).astype(int)]

        # Simulate noise
        img *= well_depth  # Convert to photo electrons
        img += dark_photon_electrons  # Add dark noise
        integration_time_units = cam.exposure_time / 1e-3  # Assume that the laser power is adjusted for an integration time of 1 ms
        img *= integration_time_units
        img += np.sqrt(img) * np.random.randn(*img.shape)
        img *= (1.10**(cam.gain * 100)) / photo_electrons_per_gray_level
        img = np.clip(img, 0, nb_gray_levels-1).astype(np.uint8)

        return img

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

        log.debug(f"Saving settings to {self.__settings_full_file_name}...")
        jsonpickle.set_encoder_options('json', sort_keys=False, indent=2)
        settings_str = jsonpickle.dumps(self.__settings)
        with open(self.__settings_full_file_name, 'w') as settings_file:
            settings_file.write(settings_str)

        log.info(f"Saved configuration to {self.__settings_full_file_name}.")


if __name__ == "__main__":
    Monitor()()
