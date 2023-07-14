
import torch

# import time
import numpy as np
import random
# from datetime import datetime, timezone
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime, timezone
# from optics import log
from optics.utils import ft
# from optics.utils.display import complex2rgb, grid2extent

from projects.adaptive_optics.imaging_system import ImagingSystem
from sklearn.metrics import mean_squared_error
from optics.utils.display import complex2rgb, grid2extent
from projects.adaptive_optics import log


model_path = Path.home() / Path('E:/Adaptive Optics/Test of model from multiple batches/Trained Model/')
model_path_test = Path.home() / Path('E:/Adaptive Optics/Test of model from multiple batches/Test data/Seed 1/')

# with NEW distribution-ast2
model_save_name = model_path / f'CNN_Model_New_Distribution_name_ast2_M_2_N_10_Num_epochs_300_Num_batches_50_Batch_Size_10000_Corrected_modes_[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_CV_kernel_size_3_learningrate_0.001_in_features250_timestamp_18-32-34.840_magnitude_1.155_noiselevel_0.02.pt'

data_save_name_smaller = model_path_test / f'PseudoPSF_batchidx0_batchsize5000_corrected_modes[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasmode[3]_biasdepths[1]_magnitude_0.577_simulate_noise_True_background_noise_level_0.02_seed_1.npz'
data_save_name_same = model_path_test / f'PseudoPSF_batchidx0_batchsize5000_corrected_modes[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasmode[3]_biasdepths[1]_magnitude_1.155_simulate_noise_True_background_noise_level_0.02_seed_1.npz'
data_save_name_bigger = model_path_test / f'PseudoPSF_batchidx0_batchsize5000_corrected_modes[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasmode[3]_biasdepths[1]_magnitude_1.732_simulate_noise_True_background_noise_level_0.02_seed_1.npz'

# 1 load trained model
model = torch.jit.load(model_save_name)  # model.eval()
model = model.cpu()
# 2 load data
data_name = data_save_name_smaller
magnitude = 1 / (1*np.sqrt(3))
noise_level = 0.05
data_after_training = np.load(str(data_name), allow_pickle=True)

test_label = np.float32(data_after_training['train_label'])
test_data = np.float32(data_after_training['train_data'])
settings = data_after_training['settings']

# predicted_labels_epoch = data_after_training['predicted_labels_epoch']
# settings = data_after_training['settings']

# convert label to zernike modes
model_name = 'ast2'
num_zernike = 18
corrected_modes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
bias_modes = [3]
bias_depths = [1]

num_samples = 10
num_iteration = 10
# for original data
# zernike_modes = np.zeros([test_label.shape[0], num_zernike])
# zernike_modes[:, corrected_modes] = test_label

# for new generated test data
zernike_modes = test_label

imaging_system = ImagingSystem(ft.Grid(np.full(2, 128), 0.2e-6), simulate_noise=True, background_noise_level=noise_level)
grid_2d = ft.Grid(np.full(2, 128), 0.2e-6)
img_grid = ft.Grid(np.full(2, 32), center=imaging_system.grid.shape // 2)
original_image = imaging_system.image_from_zernike_coefficients()
sharpness_original_image = imaging_system.calc_sharpness_of_image_fast(original_image)

blurred_image = []
corrected_images = []
sharpness_blurred_image = []
sharpness_corrected_images = []

predicted_aberration_all_idx = []
predicted_modes = []
predicted_aberrations = []
rms_original_aberration = []

rms_2d_matrix = np.zeros([num_samples, num_iteration])
sharpness_2d_matrix = np.zeros([num_samples, num_iteration])
predicted_modes_3d_matrix = np.zeros([num_samples, num_zernike, num_iteration])
corrected_modes_3d_matrix = np.zeros([num_samples, num_zernike, num_iteration])

loop_num = 0
row_indexes = random.sample(range(0, zernike_modes.shape[0]), num_samples)
# row_idx = [1, 10, 100, 200, 400, 800, 1000, 1200]

folder_txt = f'E:/Adaptive Optics/Test of model from multiple batches/Predicted/Trained Model Test Multiply Iteration {model_name} MLAO/Boat_Test Data Magnitude_{magnitude:0.3f}_interation_{num_iteration}_corrected_modes_{corrected_modes}_biasmode_{bias_modes}_biasmode_{bias_depths}_noiselevel_{noise_level}'

for idx_check in row_indexes:
    #  images: original, blurred images and corrected images
    test_mode_1 = zernike_modes[idx_check, :]
    original_aberration_row = imaging_system.phase_from_zernike_polynomial(test_mode_1)
    mean_original_aberration_row = np.mean(original_aberration_row)*np.ones(original_aberration_row.shape)
    rms_original_aberration_row = mean_squared_error(mean_original_aberration_row, original_aberration_row)
    blurred_image_row = imaging_system.image_from_zernike_coefficients(test_mode_1)
    sharpness_blurred_image_row = imaging_system.calc_sharpness_of_image_fast(blurred_image_row)
    sharpness_blurred_image.append(sharpness_blurred_image_row) #sharpness of the blured image with original aberration

    blurred_image.append(blurred_image_row)
    test_data_row = torch.from_numpy(test_data[idx_check])

    predicted_modes_row = []
    predicted_aberration_row = []
    corrected_image_row = []
    corrected_coefficient_row = []

    rms_predicted_aberration_row = []
    rms_aberration_difference_row = []
    sharpness_corrected_image_row = []
    aberration_difference_best = []

    for iteration_idx in range(num_iteration):
        predicted_coefficient_row_iter = np.zeros([1, num_zernike])
        pseudo_psf_row_iter = []
        predicted_label = model(test_data_row)

        # output = predicted_modes
        predicted_coefficient_row_iter[:, corrected_modes] = predicted_label.detach().numpy()
        predicted_aberration_row_iter = imaging_system.phase_from_zernike_polynomial(predicted_coefficient_row_iter)
        # rms
        mean_predicted_aberration_row_iter = np.mean(predicted_aberration_row_iter)*np.ones(predicted_aberration_row_iter.shape)
        rms_predicted_aberration_row_iter = mean_squared_error(mean_predicted_aberration_row_iter, predicted_aberration_row_iter)

        # print(imaging_system.phase_from_zernike_polynomial(predicted_modes).shape)
        # print(predicted_aberration.shape)
        # exit()
        corrected_coefficient = test_mode_1 - predicted_coefficient_row_iter  # corrected_coefficient will serve as the start point of next round corrction
        corrected_coefficient_row.append(corrected_coefficient) # to store the corrected_coefficient after each correction
        aberration_difference_row_iter = imaging_system.phase_from_zernike_polynomial(corrected_coefficient) # aberration left after num_iteration times corrections
        # rms
        mean_aberration_difference_best_row_iter = np.mean(aberration_difference_row_iter) * np.ones(aberration_difference_row_iter.shape)
        rms_aberration_difference_best_row_iter = mean_squared_error(mean_aberration_difference_best_row_iter, aberration_difference_row_iter)

        # print(corrected_coefficient)
        corrected_image_row_iter = imaging_system.image_from_zernike_coefficients(corrected_coefficient)

        predicted_modes_row.append(predicted_coefficient_row_iter)
        predicted_aberration_row.append(predicted_aberration_row_iter)
        corrected_image_row.append(corrected_image_row_iter)
        sharpness_corrected_image_row_iter = imaging_system.calc_sharpness_of_image_slow(corrected_image_row_iter)
        sharpness_corrected_image_row.append(sharpness_corrected_image_row_iter) # sharpness of corrected images with corrected phase after each iteration
        # to store the rms, sharpness, predicted, corrected_coefficient
        rms_2d_matrix[loop_num, iteration_idx] = rms_aberration_difference_best_row_iter
        sharpness_2d_matrix[loop_num, iteration_idx] = sharpness_corrected_image_row_iter
        predicted_modes_3d_matrix[loop_num, :, iteration_idx] = predicted_coefficient_row_iter
        corrected_modes_3d_matrix[loop_num, :, iteration_idx] = corrected_coefficient

        # plot of each row and each iteration
        if loop_num % 3 == 0:

            fig_M, axs_M = plt.subplots(2, 3, sharex='all', sharey='all', figsize=(18, 8))
            im00 = axs_M[0, 0].imshow(complex2rgb(np.exp(1j*original_aberration_row)), origin='lower')
            axs_M[0, 0].set(title=f"Row{idx_check}_num_iteration_{num_iteration}_original_aberration_RMS_({rms_original_aberration_row:0.5f})")
            # fig_M.colorbar(im00, ax=axs_M[0, 0])

            axs_M[0, 1].imshow(complex2rgb(np.exp(1j * predicted_aberration_row_iter)), origin='lower')
            axs_M[0, 1].set(title=f"predicted_aberration_Averaged_RMS_({rms_predicted_aberration_row_iter:0.5f})")

            axs_M[0, 2].imshow(complex2rgb(np.exp(1j * aberration_difference_row_iter)), origin='lower')
            axs_M[0, 2].set(title=f"aberration_difference_Averaged_RMS_({rms_aberration_difference_best_row_iter:0.5f})")

            axs_M[1, 0].imshow(original_image)
            axs_M[1, 0].set(title=f"iteration_idx_{iteration_idx}_original_image_sharpness_({sharpness_original_image:0.5f})")

            axs_M[1, 1].imshow(blurred_image_row)
            axs_M[1, 1].set(title=f"blurred_image_sharpness_({sharpness_blurred_image_row:0.5f})")

            im12 = axs_M[1, 2].imshow(corrected_image_row_iter)
            axs_M[1, 2].set(title=f"corrected_image_sharpness_({sharpness_corrected_image_row_iter:0.5f})")
            # fig_M.colorbar(im12, ax=axs_M[1, 2])
            # plt.colorbar()

            folder_path = Path.home() / Path(f'{folder_txt}/row_{idx_check}')
            # log.info(f'Results will be saved in {figure_path}.')
            folder_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-3]
            full_file_name_figures = folder_path / f'Time_{timestamp}_Test_trained_model_MLAO_{model_name}_NewDistribution_rows_checked_{len(row_indexes)}_num_iteration_{num_iteration}_current_row_{idx_check}_current_iteration_{iteration_idx}.png'

            # plt.savefig(full_file_name_figures)
            plt.close()
            # plt.show()

        # to generate pseudo psf for next round correction with specified bias method
        if iteration_idx < (num_iteration - 1):# will not generate new pseudo-psf images if num_iteration = 1
            for bias_mode in bias_modes:
                for bias_depth in bias_depths:
                    pseudo_psf = imaging_system.pseudo_psf(corrected_coefficient[0], bias_mode, bias_depth)
                    cropped_pseudo_psf = pseudo_psf[:, img_grid[0], img_grid[1]]
                    pseudo_psf_row_iter.append(cropped_pseudo_psf)
                # Store valid data
            pseudo_psf_row_iter = np.array(pseudo_psf_row_iter)

            if not np.any(np.isnan(pseudo_psf_row_iter)):
                pseudo_psf_row_iter = pseudo_psf_row_iter.reshape([pseudo_psf_row_iter.shape[0] * pseudo_psf_row_iter.shape[1], *pseudo_psf_row_iter.shape[-2:]])
                test_data_row = torch.from_numpy(np.float32(pseudo_psf_row_iter.real))  # convert double to float32 with np.float32, shape [len(bias_modes), len(bias_depths), 2, *img_ranges.shape]
                test_mode_1 = corrected_coefficient
                # print(corrected_coefficient.shape)
            else:  # Remove invalid data
                log.warning(f'Skipping a pseudo psf due to {np.sum(pseudo_psf_row_iter)} NaN.')
    loop_num += 1

settings = dict(row_idx=row_indexes,
                model_name=model_name,
                num_zernike=num_zernike,
                corrected_modes=corrected_modes,
                bias_modes=bias_modes,
                bias_depths=bias_depths,
                num_samples=num_samples,
                num_iteration=num_iteration
                )
folder_path_data = Path.home() / Path(f'{folder_txt}')
# log.info(f'Results will be saved in {figure_path}.')
folder_path_data.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-3]
full_data_name = folder_path_data / f'Time_{timestamp}_rms_and_sharpness_Test_trained_multiply_iteration_model_MLAO_{model_name}_NewDistribution_rows_checked_{len(row_indexes)}_iteration_for_each_row_{num_iteration}.npz'
log.info(f'Saving results to {full_data_name}...')
# np.savez(full_data_name, rms_2d_matrix=rms_2d_matrix, sharpness_2d_matrix=sharpness_2d_matrix, predicted_modes_3d_matrix=predicted_modes_3d_matrix, corrected_modes_3d_matrix=corrected_modes_3d_matrix, test_data=test_data, test_label=test_label, settings=settings)
#

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 18,
        }

fig_M, axs_M = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(18, 8))
im0 = axs_M[0].imshow(rms_2d_matrix, extent=[0, num_iteration, 0, num_samples], aspect='auto')
axs_M[0].set(title=f"rms of corrected phase rows checked {len(row_indexes)}_iteration {num_iteration})")
axs_M[0].set_xlabel("interation", fontdict=font)
axs_M[0].set_ylabel("row of aberration", fontdict=font)
fig_M.colorbar(im0, ax=axs_M[0])

im1 = axs_M[1].imshow(sharpness_2d_matrix, extent=[0, num_iteration, 0, num_samples], aspect='auto')
axs_M[1].set(title=f"sharpness of corrected images rows checked {len(row_indexes)}_iteration {num_iteration})")
axs_M[1].set_xlabel("interation", fontdict=font)
axs_M[1].set_ylabel("row of aberration", fontdict=font)
fig_M.colorbar(im1, ax=axs_M[1])

rms_figure_name = folder_path_data / f'Time_{timestamp}_Test_trained_multiply_iteration_model_MLAO_{model_name}_NewDistribution_rows_checked_{len(row_indexes)}_iteration_for_each_row_{num_iteration}.png'
# plt.savefig(rms_figure_name)
plt.show()

