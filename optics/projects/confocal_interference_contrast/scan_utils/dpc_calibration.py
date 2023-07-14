import numpy as np
import json

from optics.instruments.cam.ids_cam import IDSCam
from projects.confocal_interference_contrast.scan_utils.open_slm import open_slm
from projects.confocal_interference_contrast.polarization_memory_effect.polarization_memory_effect_functions import create_2_spots, calibration_protocol, \
    cams_p1_p2_exposure_time


class DPC_calibration:
    def __init__(self, load, open_cams=True):

        ### opening instruments
        self.__slm = open_slm()
        self.__open_cams = []
        exposure = self.__load(setting="cam exposure")
        cam_serial_numbers = self.__load(setting="cam serial numbers")
        if open_cams:
            self.open_all_cams(exposure, cam_serial_numbers)
        # TODO: Open the wide sensor camera and run rel_amp = spots_intensity_calibration(slm, cam_rel_amp) from polarisation_memory_effect_functions
        # TODO: rel amplitude with camera serial=4103697390

        ### calibrating setup
        if load:
            relative, self.__linear_a, self.__linear_b = self.__load(setting="DPC settings").values()
            self.__mask = np.load("cam_roi_mask.npy")
            create_2_spots(self.slm, relative, -0.5e-6)
        else:
            cams_p1_p2_exposure_time(self.slm, self.cam_reflection, self.cam_transmission)
            relative, self.__linear_a, self.__linear_b, self.__mask = calibration_protocol(self.slm, self.cam_aux,
                                                               self.cam_reflection, self.cam_transmission)

            # save spot calibration
            self.__save_spot_calibration(relative, self.__linear_a, self.__linear_b)
            np.save("cam_roi_mask.npy", self.mask)

            # save exposure
            for cam, cam_name in zip([self.cam_aux, self.cam_reflection, self.cam_transmission], exposure):
                exposure[cam_name] = cam.exposure_time
            self.__save_calibration(exposure, setting="cam exposure")

    def open_all_cams(self, exposure, cam_serial_numbers):
        self.__cam_aux = self.__open_cam(cam_serial_numbers["cam_aux"], exposure["cam_aux"])
        self.__cam_reflection = self.__open_cam(cam_serial_numbers["cam_reflection"], exposure["cam_reflection"])
        self.__cam_transmission = self.__open_cam(cam_serial_numbers["cam_transmission"], exposure["cam_transmission"])

    @property
    def slm(self):
        return self.__slm

    @property
    def cam_aux(self):
        return self.__cam_aux

    @property
    def cam_reflection(self):
        return self.__cam_reflection

    @property
    def cam_transmission(self):
        return self.__cam_transmission

    @property
    def linear_constants(self):
        return self.__linear_a, self.__linear_b

    @property
    def mask(self):
        return self.__mask

    def __open_cam(self, serial_nb, exposure_time):
        try:
            cam = IDSCam(serial=serial_nb, exposure_time=exposure_time, normalize=False)
            self.__open_cams.append(cam)
        except Exception as e:
            print(e)
            self.close()
            raise
        return cam

    def __save_spot_calibration(self, relative, a, b):
        a = a.tolist() if type(a) == np.ndarray else a
        b = b.tolist() if type(b) == np.ndarray else b
        spot_calibration_dict = {
                "relative_amplitude": relative,
                "linear_a": a,
                "linear_b": b
        }
        self.__save_calibration(dict2save=spot_calibration_dict, setting="DPC settings")

    @staticmethod
    def __load(setting="DPC settings", settings_path=r"C:\Users\lab\OneDrive - University of Dundee\Documents\lab\code\python\optics\projects\confocal_interference_contrast\scan_utils\calibration_settings.json"):
        with open(
                settings_path,
                'r') as file:
            calibration = json.load(file)[setting]
        return calibration

    @staticmethod
    def __save_calibration(dict2save, setting="DPC settings", settings_path=r"C:\Users\lab\OneDrive - University of Dundee\Documents\lab\code\python\optics\projects\confocal_interference_contrast\scan_utils\calibration_settings.json"):
        with open(
                settings_path,
                'r+') as file:
            calibration = json.load(file)
            calibration[setting] = dict2save
            file.seek(0)
            json.dump(calibration, file, indent=4)
            file.truncate()

    def close(self):
        self.slm.disconnect()
        for _ in self.__open_cams:
            _.disconnect()
