import numpy as np
from pathlib import Path
import time

from optics.instruments.slm import PhaseSLM
from optics.calc import correction_from_pupil
from ..polarization_memory_effect.polarization_memory_effect_functions import create_2_spots

separation = -0.5e-6
input_config_file_path = Path(r'C:\Users\lab\OneDrive - University of Dundee\Documents\lab\code\python\optics\results/aberration_correction_2022-05-12_15-54-37.439768.npz').resolve()
aberration_correction_settings = np.load(input_config_file_path.as_posix())
pupil_aberration = aberration_correction_settings['pupil_aberration']


def open_slm():
    try:
        slm = PhaseSLM(display_target=0, deflection_frequency=(0, -1/4), two_pi_equivalent=0.3)
        slm.correction = correction_from_pupil(pupil_aberration, attenuation_limit=4)
        slm.post_modulation_callback = lambda _: time.sleep(100e-3)

        return slm
    except Exception() as error:
        print(error)
        slm.disconnect()
        raise
