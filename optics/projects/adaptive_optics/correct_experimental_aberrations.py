
# add aberration with SLM or DM?
# add bias with DM
# collect images after bias added
# calculate the pseudo-psf with biased images

from os.path import dirname, join as pjoin
import scipy.io as sio

import torch
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

# 1. load trained model, 2. generate pseudo-psf with CCD images, 3. make predictions with trained model and experimental pseudo-psf, 4-collect corrected images
model_path = Path.home() / Path('E:/Adaptive Optics/Test of model from multiple batches/Trained Model/')
trained_model = []
# load trained model
# model-4N
# with new distribution
trained_model.append(model_path / f'CNN_Model_New_Distribution_name_ast2_M_2_N_10_Num_epochs_3000_Num_batches_50_Batch_Size_10000_Corrected_modes_[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_CV_kernel_size_3_learningrate_0.0005_in_features250_timestamp_01-14-10.099_magnitude_1.155_noiselevel_0.0.pt')
trained_model.append(model_path / f'CNN_Model_New_Distribution_name_ast2_M_2_N_10_Num_epochs_300_Num_batches_50_Batch_Size_10000_Corrected_modes_[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_CV_kernel_size_3_learningrate_0.001_in_features250_timestamp_18-32-34.840_magnitude_1.155_noiselevel_0.02.pt')
trained_model.append(model_path / f'CNN_Model_New_Distribution_name_ast2_M_2_N_10_Num_epochs_300_Num_batches_60_Batch_Size_10000_Corrected_modes_[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_CV_kernel_size_3_learningrate_0.001_in_features250_timestamp_22-21-02.332_magnitude_1.155_noiselevel_0.05.pt')
trained_model.append(model_path / f'CNN_Model_New_Distribution_name_ast2_M_2_N_10_Num_epochs_3000_Num_batches_50_Batch_Size_10000_Corrected_modes_[3, 5, 6, 7, 8, 9, 11, 12, 13, 24]_CV_kernel_size_3_learningrate_0.0005_in_features250_timestamp_11-49-59.460_magnitude_1.155_noiselevel_0.0.pt')
trained_model.append(model_path / f'CNN_Model_New_Distribution_name_ast2_M_2_N_10_Num_epochs_300_Num_batches_50_Batch_Size_10000_Corrected_modes_[3, 5, 6, 7, 8, 9, 11, 12, 13, 24]_CV_kernel_size_3_learningrate_0.001_in_features250_timestamp_20-21-57.875_magnitude_1.155_noiselevel_0.02.pt')
trained_model.append(model_path / f'CNN_Model_New_Distribution_name_ast2_M_2_N_10_Num_epochs_3000_Num_batches_20_Batch_Size_10000_Corrected_modes_[3, 5, 6, 7, 8, 9, 11, 12, 13, 24]_CV_kernel_size_3_learningrate_0.0008_in_features250_timestamp_05-12-21.179_magnitude_1.155_noiselevel_0.05.pt')
trained_model.append(model_path / f'CNN_Model_Resolution_name_ast2_M_2_N_10_Num_epochs_1000_Num_batches_10_Batch_Size_10000_Corrected_modes_[3, 5, 6, 7, 8, 9, 11, 12, 13, 22]_CV_kernel_size_3_learningrate_0.001_in_features250_timestamp_20_49_37.144_magnitude_1.155_noiselevel_0.02.pt')

# model-ast2
# with new distribution
# model_save_name = model_path / f'CNN_Model_name_ast2_M_2_N_10_Num_epochs_2000_CV_kernel_size_3_learningrate_0.0005_in_features250_timestamp_2023-03-29_14-55-45.907_magnitude_0.577_NewDistribution.pt'
# 1 load trained model
model_idx = 2
model = torch.jit.load(trained_model[model_idx])  # model.eval()
model = model.cpu()
#
rng = np.random.Generator(np.random.PCG64(seed=1))  # For the noise generation
if model_idx < 3:
    nb_modes = 18
    corrected_modes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
elif 3 <= model_idx < 6:
    nb_modes = 25
    corrected_modes = [3, 5, 6, 7, 8, 9, 11, 12, 13, 22]
else:
    nb_modes = 18
    corrected_modes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

uncorrected_modes = [17]
zernike_index = np.arange(0, nb_modes)
zernike_index = zernike_index.reshape((1, nb_modes))

number_of_aberrations = 1  # how many groups of implementations
magnitude = 0.5 / (1*np.sqrt(3))
std_zernike_coeffs = np.zeros(nb_modes)
std_zernike_coeffs[corrected_modes] = magnitude
std_zernike_coeffs[uncorrected_modes] = 0.01 / np.sqrt(3)
iteration_num = 10

bias_modes = [3]
bias_depths = [1]

# train_label = []
# we can set aberrations here. If it is zero array then there is no pre-set aberrations
coefficient_array = rng.normal(size=[number_of_aberrations, std_zernike_coeffs.size]) * std_zernike_coeffs

# coefficient_array = np.zeros((number_of_aberrations, nb_modes))
# aberration_0 = zernike.Polynomial(coefficient_array[0]).cartesian

imaging_system = ImagingSystem(ft.Grid(np.full(2, 128)), simulate_noise=True)
img_grid = ft.Grid(np.full(2, 32), center=imaging_system.grid.shape // 2)

predicted_all = []

data_dir = pjoin(dirname('E:/Expt/DM/Original USB/Config/'))
matname = pjoin(data_dir, 'BAX240-Z2C.mat')
mat_contents = sio.loadmat(matname)
Z2CMatrix = mat_contents['Z2C']

if __name__ == '__main__':
    folder_path = Path.home() / Path(f'E:/Adaptive Optics/Experimental MLAO/USAF/correction with trained model/After calibration of the scale/')
    with IDSCam(index=1, normalize=True, exposure_time=10e-3, gain=1, black_level=110) as cam:
        with AlpaoDM(serial_name='BAX240') as dm:
            dm.wavelength = 500e-9

            def modulate_and_capture(coefficients) -> float:
                aberration = zernike.Polynomial(coefficients).cartesian  # defocus, j_index = 4
                # dm.modulate(lambda u, v: aberration(u, v) * dm.wavelength / (2 * np.pi))
                dm.modulate(lambda u, v: aberration(v, -u) / 17)
                # actuator_positions = dm.actuator_position
                log.info(f'Displaying {aberration}...')
                img = cam.acquire()
                time.sleep(0.1)
                return img  # pick the maximum (might be noisy)
            # 2. generate pseudo-psf with CCD images,
            # coefficient = np.zeros((1, nb_modes))

            for superpos_idx in range(number_of_aberrations):
                coefficient0 = coefficient_array[superpos_idx]
                coefficient0 = coefficient0.reshape(1, nb_modes)
                # coefficient0[0, 4] = -2.5
                image_no_correction = modulate_and_capture(coefficient0)

                for iters in range(iteration_num):
                    pseudo_psf_all_modes = []
                    training_data = []
                    print(iters)
                    if iters == 0:
                        coefficient = coefficient0
                        # print(coefficient)
                    corrected_image_last = modulate_and_capture(coefficient)

                    for bias_mode in bias_modes:
                        for bias_depth in bias_depths:
                            bias_coeffs = np.zeros_like(coefficient)
                            # print(bias_depth)
                            # exit()
                            bias_coeffs[:, bias_mode] = bias_depth
                            images = [modulate_and_capture(coefficient + _) for _ in [-bias_coeffs, bias_coeffs]]
                            pseudo_psf = imaging_system.pseudo_psf_from_images(images)
                            cropped_pseudo_psf = pseudo_psf[:, img_grid[0], img_grid[1]]
                            pseudo_psf_all_modes.append(cropped_pseudo_psf)
                            # print(np.asarray(images).shape)

                    if not np.any(np.isnan(pseudo_psf_all_modes)):
                        training_data.append(pseudo_psf_all_modes)  # shape [len(bias_modes), len(bias_depths), 2, *img_ranges.shape]
                    else:  # Remove invalid data
                        log.warning(f'Skipping a pseudo psf due to {np.sum(pseudo_psf_all_modes)} NaN.')
                    # dm.modulate(lambda u, v: aberration_0(u, v)*0.5)
                    # # actuator_positions = dm.actuator_position
                    training_data = np.asarray(training_data)
                    training_data = training_data.reshape([training_data.shape[0], -1, *training_data.shape[-2:]])
                    # print(np.asarray(training_data).shape)
                    training_data = torch.from_numpy(np.float32(training_data))

                    # 3. make predictions with trained model and the experimental pseudo-psf
                    predicted_label = model(training_data)
                    predicted_zernike = np.zeros((1, nb_modes))
                    predicted_zernike[:, corrected_modes] = predicted_label.detach().numpy()

                    # 4-collect corrected images
                    corrected_image = modulate_and_capture(coefficient - predicted_zernike)
                    # coefficient_array[superpos_idx] = predicted_zernike
                    coefficient = coefficient - predicted_zernike
                    log.info(f'predicted zernike {predicted_zernike}')
                    print(predicted_zernike)
                    predicted_all.append(predicted_zernike)

                    # 5-figures
                    fig_M, axs_M = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(18, 8))
                    im0 = axs_M[0].imshow(image_no_correction)
                    axs_M[0].set(title=f"image without correction")
                    im1 = axs_M[1].imshow(corrected_image_last)
                    axs_M[1].set(title=f"corrected image last step")
                    im2 = axs_M[2].imshow(corrected_image)
                    axs_M[2].set(title=f"image after correction interation index = {iters}")

                    # log.info(f'Results will be saved in {figure_path}.')
                    folder_path.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-3]
                    full_file_name_figures = folder_path / f'Time_{timestamp}_model_index_{model_idx}_corrected_modes_{corrected_modes}_bias_modes_{bias_modes}_bias_depths_{bias_depths}_iteration_idx_{iters}.png'
                    plt.savefig(full_file_name_figures)
                    # plt.show()

        predicted_all = np.asarray(predicted_all)
        predicted_all = predicted_all.reshape([iteration_num, nb_modes])
        print(predicted_all.shape)
        title_str = f' model_index_{model_idx}_interation_index{iteration_num}'
        plt.figure(11)
        plt.plot(predicted_all.T, '.-', linewidth=2, label="interation index")
        plt.plot(coefficient0.T, 'ok--', linewidth=4, label="aberration")
        plt.title(title_str)
        plt.xlabel("Zernike index")
        plt.ylabel("coefficient")
        # plt.legend(loc="upper right")
        plt.show()

