import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from projects.confocal_interference_contrast.polarization_memory_effect import log
from optics.instruments.slm import PhaseSLM
from optics.utils.display import grid2extent
from optics.calc import correction_from_pupil
from optics.utils.ft import Grid
from optics.instruments.objective import Mitutoyo


x_center = [820, 1443, 682, 677]
y_center = [1065, 527, 442]

input_config_file_path = r'D:\scan_pol_memory\utils\aberration_correction_2022-05-12_15-54-37.439768.npz'


def create_3_spots(slm,relative_amplitude: complex = 1.05, separation_in_sample: float = -1.5e-6, adjust_var =0.01):
    """Function that creates two focal spots with a relative complex amplitude of the second spot
    :param relative_amplitude: The amplitude of the second spot with respect to the first.
    """

    x_offset = [-0.021, 0.04, 0]
    y_offset = [0, 0, 0]
    z_offset = [0, 0.0001, 0]
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
    amplitudes /= np.linalg.norm(amplitudes) * np.sqrt(2) ##todo check the * np.sqrt(2)
    # This is to make sure that circular or diagonal polarization do not require SLM amplitudes larger than 1. This is not needed for pure H or V polarization.
    reference_beam_amplitude_fraction = 1/3
    amplitudes = [*(amplitudes * (1 - reference_beam_amplitude_fraction)), reference_beam_amplitude_fraction]
    wavevectors = np.array([-dk / 2, dk / 2, 2*np.pi * np.array([0.3874, 0.155]) / slm_pixel_pitch])  # wavenumbers in rad/m

    slm.modulate(lambda x, y: (np.sqrt(x**2 + y**2) * slm_pixel_pitch < 2 * radius_at_slm) *
                              sum(a * np.exp(1j * (x*slm_pixel_pitch * kx*(1+ox) + y*slm_pixel_pitch * ky*(1+oy)
                                                   + oz *(x**2 + y**2)))
                                  for a, kx, ky, ox, oy, oz in zip(amplitudes[0:3], wavevectors[:, 0], wavevectors[:, 1],
                                                                   x_offset, y_offset, z_offset))
                 )

def create_2_spots(slm,relative_amplitude: float = 1.0, separation_in_sample: float = -0.5e-6, theta: float = 0.0, x_offset = [-0.0204, 0.04, 0]):
    """Function that creates two focal spots with a relative complex amplitude of the second spot
    :param relative_amplitude: The amplitude of the second spot with respect to the first.
    :param theta: The relative phase of the second spot with respect to the first. (rad)
    """
    #x_offset = [-0.021, 0.04, 0]
    y_offset = [0, 0, 0]
    z_offset = [0, 0.0001, 0]
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

    if relative_amplitude == np.infty:
        amplitudes = np.array([0, 1.0])
    else:
        amplitudes = np.array([1.0, relative_amplitude])
        amplitudes /= np.linalg.norm(amplitudes)
    wavevectors = np.array([-dk / 2, dk / 2, 2*np.pi * np.array([0.3874, 0.155]) / slm_pixel_pitch])  # wavenumbers in rad/m
    relative_phase = [1.0, np.exp(1j*theta)]

    slm.modulate(lambda x, y: (np.sqrt(x**2 + y**2) * slm_pixel_pitch < 2 * radius_at_slm) *
                              sum(a * rp * np.exp(1j * (x*slm_pixel_pitch * kx*(1+ox) + y*slm_pixel_pitch * ky*(1+oy)
                                                   + oz *(x**2 + y**2)))
                                  for a, kx, ky, ox, oy, oz, rp in zip(amplitudes[0:2], wavevectors[:, 0], wavevectors[:, 1],
                                                                   x_offset, y_offset, z_offset, relative_phase))
                 )

def set_exposure_time(cam,intensity_target: float = 0.85, initial_exposure_time: float = 50 * 1e-3):
    c = 0
    cam.exposure_time = initial_exposure_time
    max_intensity = 0
    while (max_intensity > intensity_target+0.05 or max_intensity < intensity_target-0.05) and c < 20:
        max_intensity = np.amax(cam.acquire()) / 255
        if max_intensity == 1:
            adjusting_constant = 0.5
        else:
            adjusting_constant = intensity_target/np.maximum(max_intensity, 0.1)
        initial_exposure_time *= adjusting_constant
        cam.exposure_time = initial_exposure_time
        img = cam.acquire()
        max_intensity = np.amax(img) / 255
        c += 1
    print(f'Cam {cam.model}:Max intensity after {c} iteration = {max_intensity:0.3f}, target intensity value is {intensity_target}, exposure time = {cam.exposure_time*1000} ms')
    return cam.exposure_time

def creates_display(img1, img2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharex='all', sharey='all')
    cam_axs_0 = axs[:]
    cam_axs_0[0].imshow(img1)
    cam_axs_0[1].imshow(img2)
    cam_axs_0[0].set_title('img1')
    cam_axs_0[1].set_title('img2')
    plt.show()

def display_imgs(imgs, pol_names = ['D','A', 'L', 'R']):
    fig, axs = plt.subplots(2, 2)
    c = 0
    for _ in range(2):
        for a in range(2):
            axs[_,a].imshow(imgs[c])
            axs[_,a].set_title(pol_names[c])
            c += 1
    plt.show()

def cam_multiple_acquirements(cam, number_acquirements):
    img = np.zeros(cam.shape)
    for _ in range(number_acquirements):
        img += cam.acquire()
    return img/number_acquirements


def spots_intensity_calibration(slm, cam, number_measurements: int = 10):
    #Will return the relative amplitude to have two spots with equal intensities
    create_2_spots(slm, 0, -0.5e-6)
    set_exposure_time(cam)
    relative_amplitude = np.linspace(0.7, 1.3, number_measurements) #
    measured_relative_intensity = np.zeros(number_measurements)
    dist = 1
    count =0
    while dist >= 0.01 and count < (number_measurements - 2):
        for _ in range(number_measurements):
            create_2_spots(slm, relative_amplitude[_], -0.5e-6)
            spot_1, spot_2 = cam_aux2p3p4(cam.acquire(), 50)
            measured_relative_intensity[_] = np.sum(spot_2, dtype=np.float64) / np.sum(spot_1, dtype=np.float64)
            print(measured_relative_intensity[_])
        idx = np.abs(measured_relative_intensity - 1).argmin()
        closest_value = measured_relative_intensity.flat[idx]
        dist = np.abs(1-closest_value)
        best_relative_amplitude = relative_amplitude[idx]
        relative_amplitude = np.linspace(relative_amplitude[idx-2], relative_amplitude[idx+2], number_measurements)
        count +=1
    print(f'Input relative amplitude = {best_relative_amplitude}; Relative intensity measured = {closest_value}')
    return best_relative_amplitude




if __name__ == '__main__':
    log.info(f'Loading aberration correction from {input_config_file_path}...')
    aberration_correction_settings = np.load(input_config_file_path)
    pupil_aberration = aberration_correction_settings['pupil_aberration']
    log.info('Opening the spatial light modulator...')
    with PhaseSLM(display_target=1, deflection_frequency=(0, -1/4), two_pi_equivalent=0.3) as slm:
        slm.correction = correction_from_pupil(pupil_aberration, attenuation_limit=4)
        slm.post_modulation_callback = lambda _: time.sleep(100e-3)
        create_2_spots(slm, 0, -0.5e-6, 0.0)

        input('enter')
        exit()
