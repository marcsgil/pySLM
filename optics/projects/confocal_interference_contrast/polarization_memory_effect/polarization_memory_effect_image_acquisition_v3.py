import numpy as np
import time
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from projects.confocal_interference_contrast.polarization_memory_effect import log
from optics.instruments.cam.ids_cam import IDSCam, IDSCamError
from optics.instruments.slm import PhaseSLM
from optics.utils.display import grid2extent
from optics.calc import correction_from_pupil
from optics.utils.ft import Grid
from process_polarization_memory_effect import display_polarization_memory_effect
# from optics.instruments.stage.nanostage import NanostageLT3
from optics.instruments.objective import Mitutoyo

input_config_file_path = Path('../../../results/aberration_correction_2022-05-12_15-54-37.439768.npz').resolve()


def create_spots(relative_amplitude: complex = 1.0, separation_in_sample: float = 0.0, oy =0.01):
    """Function that creates two focal spots with a relative complex amplitude of the second spot
    :param relative_amplitude: The amplitude of the second spot with respect to the first.
    """

    y_offset = [oy, 0, 0]
    separation_in_sample = np.array([0, separation_in_sample])

    objective = Mitutoyo(10, 0.55)
    wavelength = 488e-9
    k0 = 2 * np.pi / wavelength
    slm_pixel_pitch = 20e-6
    lens_f_between_slm_and_waveplate = 500e-3
    magnification_from_waveplate_to_pbd = 75/150  # 150, 75mm
    pbd_polarization_separation = np.array([3.115e-3, 0])  # Polarizing Beam Displacer
    magnification_from_pbd_to_sample = 150/100 * 2/400  # 100, 150, 400, 2 (200 / 100x)

    magnification_from_slm_to_backaperture = objective.focal_length / (lens_f_between_slm_and_waveplate * magnification_from_waveplate_to_pbd * magnification_from_pbd_to_sample)  # 500, 150, 75, 100, 150, 400

    effective_f_to_pbd = magnification_from_waveplate_to_pbd * lens_f_between_slm_and_waveplate  # m
    separation_right_after_pbd = separation_in_sample / magnification_from_pbd_to_sample  # m
    separation_right_before_pbd = pbd_polarization_separation + separation_right_after_pbd   # m
    dircos = separation_right_before_pbd / effective_f_to_pbd  # relative wavenumber (kx/k0), or cosine of deflection angle

    radius_at_slm = objective.pupil_diameter / 2.0 / magnification_from_slm_to_backaperture
    dk = dircos * k0  # wavenumber shift in rad/m

    amplitudes = np.array([1.0, relative_amplitude])
    amplitudes /= np.linalg.norm(amplitudes) * np.sqrt(2)
    reference_beam_amplitude_fraction = 1/3
    amplitudes = [*(amplitudes * (1 - reference_beam_amplitude_fraction)), reference_beam_amplitude_fraction]
    wavevectors = np.array([-dk / 2, dk / 2, 2*np.pi * np.array([0.3874, 0.155]) / slm_pixel_pitch])  # wavenumbers in rad/m
    print(f'slm pixel = {slm_pixel_pitch}')
    print(f'kx = {wavevectors[:, 0]*slm_pixel_pitch}')
    print(f'ky = {wavevectors[:, 1]*slm_pixel_pitch}')
    slm.modulate(lambda x, y: (np.sqrt(x**2 + y**2) * slm_pixel_pitch < 2 * radius_at_slm) *
                              sum(a * np.exp(1j * (x*slm_pixel_pitch * kx*(1+oy) + y*slm_pixel_pitch * ky))
                                  for a, kx, ky, oy in zip(amplitudes[0:3], wavevectors[:, 0], wavevectors[:, 1], y_offset))
                 )


if __name__ == '__main__':
    log.info(f'Loading aberration correction from {input_config_file_path}...')
    aberration_correction_settings = np.load(input_config_file_path.as_posix())
    pupil_aberration = aberration_correction_settings['pupil_aberration']
    log.info('Opening the spatial light modulator...')
    with PhaseSLM(display_target=0, deflection_frequency=(0, -1/4), two_pi_equivalent=0.3) as slm:
        slm.correction = correction_from_pupil(pupil_aberration, attenuation_limit=4)
        slm.post_modulation_callback = lambda _: time.sleep(100e-3)
        log.info('Opening the cameras...')
        #create_spots(relative_amplitude=1.0)

        control = '1'
        freq2 = 0
        while control != '0':
            control = input('0 - exit, 1 - adjustment, 2 - diameter adjustment, 3- relative amplitude')
            if control == '1':
                try:
                    separation_in_sample = float(input('beam separation (in meters, can be negative):'))
                except ValueError:
                    separation_in_sample = 0.0
                    log.warning(f'Could not understand separation value, using {separation_in_sample}m instead')
                create_spots(1.0, separation_in_sample)  # 1um = 0.485
            if control == '4':
                oy = float(input('oy:'))*(-1e-6)
                create_spots(0.95, oy, 0.01)

