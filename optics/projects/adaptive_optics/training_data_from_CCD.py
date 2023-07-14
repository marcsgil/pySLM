
# add aberration with SLM or DM?
# add bias with DM
# collect images after bias added
# calculate the pseudo-psf with biased images

import time

from optics.instruments.dm.alpao_dm import AlpaoDM
from optics.calc import zernike
from optics import log
import numpy as np
# from scipy.optimize import differential_evolution
from matplotlib import pyplot as plt
from pathlib import Path
import time
from optics.instruments.cam.ids_cam import IDSCam
from optics.utils import ft
from optics.instruments.cam.ids_peak_cam import IDSPeakCam
from optics.utils import Roi
from optics.instruments.slm import PhaseSLM
from projects.adaptive_optics.imaging_system import ImagingSystem
from datetime import datetime, timezone
from pathlib import Path


rng = np.random.Generator(np.random.PCG64(seed=1))  # For the noise generation
nb_modes = 18
number_of_aberrations = 1  # how many groups of implementations

corrected_modes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
uncorrected_modes = [17]
bias_modes = [3]
bias_depths = [0.5, 1]
std_zernike_coeffs = np.zeros(nb_modes)
std_zernike_coeffs[corrected_modes] = 1 / np.sqrt(3)
std_zernike_coeffs[uncorrected_modes] = 0.01 / np.sqrt(3)
training_data = []
train_label = []
coefficient_array = rng.normal(size=[number_of_aberrations, std_zernike_coeffs.size]) * std_zernike_coeffs
aberration_0 = zernike.Polynomial([]).cartesian
imaging_system = ImagingSystem(ft.Grid(np.full(2, 128), 0.2e-6), simulate_noise=True)
img_grid = ft.Grid(np.full(2, 64), center=imaging_system.grid.shape // 2)

if __name__ == '__main__':
    output_folder = Path.home() / Path('E:/Adaptive Optics/MLAO npz files from CCD/')
    with IDSCam(index=1, normalize=True, exposure_time=50e-3, gain=1, black_level=150) as cam:
        with AlpaoDM(serial_name='BAX240') as dm:
            dm.wavelength = 532e-9

            def modulate_and_capture(coefficients) -> float:
                aberration = zernike.Polynomial(coefficients).cartesian  # defocus, j_index = 4
                dm.modulate(lambda u, v: aberration(u, v))
                # actuator_positions = dm.actuator_position
                # log.info(f'Displaying {coefficient_bias}...')
                time.sleep(0.02)
                img = cam.acquire()
                return img  # pick the maximum (might be noisy)
            for superpos_idx in range(number_of_aberrations):
                pseudo_psf_all_modes = []
                for bias_mode in bias_modes:
                    for bias_depth in bias_depths:
                        coefficient = coefficient_array[superpos_idx]
                        bias_coeffs = np.zeros_like(coefficient)
                        bias_coeffs[bias_mode] = bias_depth
                        # coefficient_bias = coefficient + bias_coeffs
                        images = [modulate_and_capture(coefficient + _) for _ in [-bias_coeffs, bias_coeffs]]
                        pseudo_psf = imaging_system.pseudo_psf_from_images(images)
                        cropped_pseudo_psf = pseudo_psf[:, img_grid[0], img_grid[1]]
                        pseudo_psf_all_modes.append(cropped_pseudo_psf)
                        print(np.asarray(images).shape)

                if not np.any(np.isnan(pseudo_psf_all_modes)):
                    training_data.append(pseudo_psf_all_modes)  # shape [len(bias_modes), len(bias_depths), 2, *img_ranges.shape]
                    print(np.asarray(training_data).shape)
                    train_label.append(coefficient_array[superpos_idx, :])
                else:  # Remove invalid data
                    log.warning(f'Skipping a pseudo psf due to {np.sum(pseudo_psf_all_modes)} NaN.')
                # dm.modulate(lambda u, v: aberration_0(u, v)*0.5)
                # # actuator_positions = dm.actuator_position
        training_data = np.asarray(training_data)
        training_data = training_data.reshape([training_data.shape[0], -1, *training_data.shape[-2:]])
        train_label = np.asarray(train_label)

    log.info('Done!')
    # save the data to npz file
    settings = dict(nb_modes=nb_modes,
                    number_of_aberrations=number_of_aberrations,
                    corrected_modes=corrected_modes,
                    bias_modes=bias_modes,
                    bias_depths=bias_depths,
                    background_noise_level=imaging_system.background_noise_level
                    )

    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
    full_file_name = output_folder / f'PseudoPSF_from_CCD_{timestamp}_patchsize{number_of_aberrations}_biasmode{bias_modes}_biasdepths{bias_depths}.npz'

    log.info(f'Saving results to {full_file_name}...')
    np.savez(full_file_name, training_data=training_data, train_label=train_label, settings=settings)
    log.info('Saved everything!')

    print(np.array(training_data).shape)
    log.info('Displaying...')
    fig, axs = plt.subplots(2, number_of_aberrations, sharex='all', sharey='all')
    if number_of_aberrations < 2:
        axs = axs[:, np.newaxis]
    for _, ax_bias in enumerate(axs.transpose()):
        ax_bias[0].imshow(training_data[_, 0])
        ax_bias[1].imshow(training_data[_, 1])
    log.info('Done, close window to exit.')
    plt.show()

