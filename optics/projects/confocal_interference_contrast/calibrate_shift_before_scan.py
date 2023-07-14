import numpy as np

from projects.confocal_interference_contrast.complex_phase_image import registration_calibration, save_registration
from optics.instruments.cam.ids_cam import IDSCam

cams = [IDSCam(exposure_time=5e-3) for _ in range(1)]

try:
    for idx, cam in enumerate(cams):
        snapshot = cam.acquire()
        registration = registration_calibration(snapshot, registration_name=f"scan pre-calibration for cam{idx}")
        # save_registration(registration, registration_name=f"scan pre-calibration for cam{idx}")
finally:
    for cam in cams:
        cam.power_down()



