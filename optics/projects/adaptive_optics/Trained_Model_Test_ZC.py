# importing the libraries
# import pandas as pd
# import numpy as np
#
# # for reading and displaying images
# from skimage.io import imread
# import matplotlib.pyplot as plt
# # % matplotlib inline
#
# # for creating validation set
# from sklearn.model_selection import train_test_split
#
# # for evaluating the model
# from sklearn.metrics import accuracy_score
# from tqdm import tqdm
#
# # PyTorch libraries and modules
import torch
import torch.nn as nn
# import torchvision.models as models
# from torch.autograd import Variable
# from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
#
# import torchvision
# import torchvision.transforms as transforms
# import torch.optim as optim
#
# import torch.nn.functional as F
# import torch.optim as optim
# from optics import log
# from pathlib import Path
#
# import time
import numpy as np
import random
# from datetime import datetime, timezone
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
from optics import log
from optics.utils import ft
# from optics.utils.display import complex2rgb, grid2extent

from projects.adaptive_optics.imaging_system import ImagingSystem
from sklearn.metrics import mean_squared_error
# from optics.utils import mse as mean_squared_error
from optics.utils.display import complex2rgb, grid2extent
from projects.adaptive_optics import log
import math

#  load: trained CNN model and data for test

M = 40
N = 10
num_cv1 = 8
num_cv2 = 16
num_cv3 = 32
num_cv4 = 64
num_cv5 = 128
cv_kernel_size = 3

num_epochs = 2000
learning_rate = 0.0008
in_features = 288
model_path = Path.home() / Path('E:/Adaptive Optics/MLAO/')
model_path_test = Path.home() / Path('E:/Adaptive Optics/Test the Trained Model with the Dada of Different Magnitude/')
Test_figure_path = Path.home() / Path('E:/Adaptive Optics/Test trained model figures/')

model_name = 'ast2'
loss_fn = nn.MSELoss()
# model and data for 4N
# model_save_name = model_path / f'CNN_Model_name_4N_M_{40}_N_{N}_Num_epochs_{num_epochs}_CV_kernel_size_{cv_kernel_size}_learningrate_{learning_rate}_in_features{in_features}_timestamp_2023-03-19_21-38-07.002.pt'
# model_save_name = model_path / f'CNN_Model_name_4N_M_40_N_10_Num_epochs_5000_CV_kernel_size_3_learningrate_0.0008_in_features288_timestamp_2023-03-26_22-47-40.792.pt'
# model_save_name = model_path / f'CNN_Model_name_4N_M_40_N_10_Num_epochs_2000_CV_kernel_size_3_learningrate_0.0008_in_features288_timestamp_2023-03-27_17-09-06.693.pt'


# data
# data_save_name = model_path / f'CNN_training_Data_Model_name_4N_M_{40}_N_{N}_Num_epochs_{num_epochs}_CV_kernel_size_{cv_kernel_size}_learningrate_{learning_rate}_in_features{in_features}_timestamp_2023-03-19_21-38-07.002.npz'
# data_save_name = model_path / f'CNN_training_Data_Model_name_4N_M_40_N_10_Num_epochs_2000_CV_kernel_size_3_learningrate_0.0008_in_features288_timestamp_2023-03-27_17-09-06.693.npz'
# data_save_name_bigger = model_path_test / f'PseudoPSF_2023-03-23_17-35-46.947_patchidx0_patchsize1000_biasmode[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasdepths[0.5, 1]_magnitude_1.732_simulate_noise_True.npz'
# data_save_name_smaller = model_path_test / f'PseudoPSF_2023-03-23_17-21-00.880_patchidx0_patchsize1000_biasmode[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasdepths[0.5, 1]_magnitude_0.192_simulate_noise_True.npz'
# data_save_name_same = model_path_test / f'PseudoPSF_2023-03-23_17-08-05.537_patchidx0_patchsize1000_biasmode[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasdepths[0.5, 1]_magnitude_0.577_simulate_noise_True.npz'
# data_save_name_same = model_path_test / f'PseudoPSF_2023_03_26_12_32_07.276_patchidx0_patchsize1000_biasmode[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasdepths[0.5, 1]_magnitude_0.577_simulate_noise_True.npz'

# with new distribution
model_save_name = model_path / f'CNN_Model_New_Distribution_name_4N_M_40_N_10_Num_epochs_5000_CV_kernel_size_3_learningrate_0.0006_in_features288_timestamp_2023-03-30_01-14-09.469_magnitude_0.5773502691896258.pt'
#
# data_save_name = model_path / f'CNN_training_Data_New_Distribution_Model_name_4N_M_40_N_10_Num_epochs_5000_CV_kernel_size_3_learningrate_0.0006_in_features288_timestamp_2023-03-30_01-14-09.469_magnitude_0.5773502691896258.npz'
# data_save_name_bigger = model_path_test / f'PseudoPSF_2023_03_27_20_58_11.273_patchidx0_patchsize3333_biasmode[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasdepths[0.5, 1]_magnitude_1.732_simulate_noise_True.npz'
# data_save_name_smaller = model_path_test / f'PseudoPSF_2023_03_27_19_49_21.077_patchidx0_patchsize3333_biasmode[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasdepths[0.5, 1]_magnitude_0.192_simulate_noise_True.npz'
# data_save_name_same = model_path_test / f'PseudoPSF_2023_03_27_18_09_16.059_patchidx0_patchsize3333_biasmode[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasdepths[0.5, 1]_magnitude_0.577_simulate_noise_True.npz'
data_save_name_same = model_path_test / f'PseudoPSF_2023_04_11_14_14_45.730_patchidx0_patchsize3000_biasmode[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasdepths[0.5, 1]_magnitude_1.732_simulate_noise_True_background_noise_level_0.05.npz'
data_save_name_same = model_path_test / f'PseudoPSF_2023_04_11_15_35_47.927_patchidx0_patchsize3000_biasmode[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasdepths[0.5, 1]_magnitude_1.732_simulate_noise_True_noise0.1.npz'

# model and data for ast2
# model_save_name = model_path / f'CNN_Model_name_{model_name}_M_{2}_N_{N}_Num_epochs_{1000}_CV_kernel_size_{3}_learningrate_{0.002}_in_features{250}_timestamp_2023-03-21_12-57-38.087.pt'
# model_save_name = model_path / f'CNN_Model_name_ast2_M_2_N_10_Num_epochs_2000_CV_kernel_size_3_learningrate_0.0008_in_features250_timestamp_2023-03-28_13-09-12.352_magnitude_0.577.pt'
# data_save_name = model_path / f'CNN_training_Data_Model_name_ast2_M_2_N_10_Num_epochs_2000_CV_kernel_size_3_learningrate_0.0008_in_features250_timestamp_2023-03-28_13-09-12.352_magnitude_0.577.npz'
# with new distribution
# model_save_name = model_path / f'CNN_Model_name_ast2_M_2_N_10_Num_epochs_2000_CV_kernel_size_3_learningrate_0.0005_in_features250_timestamp_2023-03-29_14-55-45.907_magnitude_0.577_NewDistribution.pt'
# data_save_name = model_path / f'CNN_training_Data_Model_name_ast2_M_2_N_10_Num_epochs_2000_CV_kernel_size_3_learningrate_0.0005_in_features250_timestamp_2023-03-29_14-55-45.907_magnitude_0.577_NewDistribution.npz'
#
# data_save_name_bigger = model_path_test / f'PseudoPSF_2023_03_28_10_28_45.946_patchidx0_patchsize25000_biasmode[3]_biasdepths[1]_magnitude_1.732_simulate_noise_True.npz'
# data_save_name_smaller = model_path_test / f'PseudoPSF_2023_03_28_10_14_41.497_patchidx0_patchsize25000_biasmode[3]_biasdepths[1]_magnitude_0.192_simulate_noise_True.npz'
# data_save_name_same = model_path_test / f'PseudoPSF_2023_03_28_15_03_51.208_patchidx0_patchsize3333_biasmode[3]_biasdepths[1]_magnitude_0.577_simulate_noise_True.npz'

# log.info(f'Saving trained model to {model_path}...')
# torch.save(best_model.state_dict(), model_save_name)

# 1 load trained model
model = torch.jit.load(model_save_name)  # model.eval()

# 2 load data
data_name = data_save_name_same
magnitude = 1 / (1*np.sqrt(3))
original_loss = 0.1236
data_after_training = np.load(str(data_name))
test_label = np.float32(data_after_training['train_label'])
test_data = np.float32(data_after_training['train_data'])

log.info(f'load_data_from_{data_name}')
# predicted_labels_epoch = data_after_training['predicted_labels_epoch']
# settings = data_after_training['settings']
# the statistics
# test_loss

# print()
# exit()

# convert label to zernike modes
num_zernike = 18
corrected_modes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# convert label to zernike modes
# with new generated data
# zernike_modes = np.zeros([test_label.shape[0], num_zernike])
# predicted_modes = np.zeros([test_label.shape[0], num_zernike])
# zernike_modes[:, corrected_modes] = test_label
# with original test data
zernike_modes = test_label

imaging_system = ImagingSystem(ft.Grid(np.full(2, 128), 0.2e-6), simulate_noise=True)
grid_2d = ft.Grid(np.full(2, 128), 0.2e-6)
img_grid = ft.Grid(np.full(2, 32), center=imaging_system.grid.shape // 2)
original_image = imaging_system.image_from_zernike_coefficients()
sharpness_original_image = imaging_system.calc_sharpness_of_image_fast(original_image)

num_samples = 3000
num_iteration = 1


blurred_image = []
corrected_image = []
sharpness_blurred_image = []
sharpness_corrected_image = []
rms_original_aberration = []
rms_predicted_aberration = []
rms_aberration_difference = []
test_loss = []
predicted_aberration_all = []

row_idx = random.sample(range(0, zernike_modes.shape[0]), num_samples)
# row_idx = [365]

for idx_check in row_idx:
    #  images: original, blurred images and corrected images
    # blured image
    # predicted_aberration = []
    predicted_modes = np.zeros([1, num_zernike])
    test_mode_1_row = zernike_modes[idx_check, :]
    blurred_image_idx = imaging_system.image_from_zernike_coefficients(test_mode_1_row)
    blurred_image.append(blurred_image_idx)
    # to make prediction with trained model
    test_data_1 = torch.from_numpy(test_data[idx_check])
    output = model(test_data_1)
    # To calculate the loss function
    # test_loss = loss_fn(test_predicted_label, torch.from_numpy(test_label))
    test_loss_1_row = loss_fn(output, torch.reshape(torch.from_numpy(test_label[idx_check, corrected_modes]), (1, 10))).detach().numpy()
    test_loss.append(test_loss_1_row)
    # log.info(f'test_loss_{test_loss_1_row:0.3f}_for_row_{idx_check}')
    # predicted aberration for current row
    predicted_modes[:, corrected_modes] = output.detach().numpy()
    predicted_aberration_row = imaging_system.phase_from_zernike_polynomial(predicted_modes)
    predicted_aberration_all.append(predicted_aberration_row)

    # to calculate the corrected images
    corrected_coefficient = test_mode_1_row - predicted_modes  # corrected_coefficient will serve as the start point of next round corrction
    corrected_image_idx = imaging_system.image_from_zernike_coefficients(corrected_coefficient)
    corrected_image.append(corrected_image_idx)
    # to calculate sharpness of images
    sharpness_blurred_image.append(imaging_system.calc_sharpness_of_image_fast(blurred_image_idx))
    sharpness_corrected_image.append(imaging_system.calc_sharpness_of_image_fast(corrected_image_idx))

    # to calculate the RMS of phase
    original_aberration = imaging_system.phase_from_zernike_polynomial(zernike_modes[idx_check, :])
    aberration_difference = imaging_system.phase_from_zernike_polynomial(corrected_coefficient)
    # RMS
    mean_original_aberration = np.mean(original_aberration)*np.ones(original_aberration.shape)
    mean_predicted_aberration = np.mean(predicted_aberration_row) * np.ones(predicted_aberration_row.shape)
    mean_aberration_difference = np.mean(aberration_difference)*np.ones(aberration_difference.shape)

    rms_original_aberration.append(mean_squared_error(mean_original_aberration, original_aberration))
    rms_predicted_aberration.append(mean_squared_error(mean_predicted_aberration, predicted_aberration_row))
    rms_aberration_difference.append(mean_squared_error(mean_aberration_difference, aberration_difference))

    # fig_M, axs_M = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(18, 8))
    # axs_M[0].imshow(complex2rgb(mean_predicted_aberration))
    # axs_M[1].imshow(complex2rgb(original_aberration), origin='lower')
    # plt.show()
    # exit()

# to calculate the statistic parameters
# images-averaged
blurred_image_average = np.mean(np.array(blurred_image), axis=0)
corrected_image_average = np.mean(np.array(corrected_image), axis=0)
# sharpness-averaged
sharpness_blurred_image_average = np.mean(np.array(sharpness_blurred_image))
sharpness_corrected_image_average = np.mean(np.array(sharpness_corrected_image))
# rms of phase
rms_original_aberration_average = np.mean(np.array(rms_original_aberration))
rms_predicted_aberration_average = np.mean(np.array(rms_predicted_aberration))
rms_aberration_difference_average = np.mean(np.array(rms_aberration_difference))
# test loss
test_loss_mean = np.mean(np.array(test_loss))

# to calculate the std
std_sharpness_before = np.std(np.array(sharpness_blurred_image))
std_sharpness_after = np.std(np.array(sharpness_corrected_image))

std_rms_before = np.std(np.array(rms_original_aberration))
std_rms_predict = np.std(np.array(rms_predicted_aberration))
std_rms_after = np.std(np.array(rms_aberration_difference))

std_test_loss = np.std(test_loss)

# to calculate the improved rate
test_loss_better_than_average = np.array(test_loss) < test_loss_mean
test_loss_better_than_original = np.array(test_loss) < original_loss
sharpness_better_than_average = np.array(sharpness_corrected_image) > sharpness_corrected_image_average
sharpness_better_than_original = np.array(sharpness_corrected_image) > np.array(sharpness_blurred_image)
rms_better_than_average = np.array(rms_aberration_difference) < rms_original_aberration_average
rms_better_than_original = np.array(rms_aberration_difference) < np.array(rms_original_aberration)

# log information

log.info(f'test_loss_after_{test_loss_mean}')
log.info(f'better_than_original_{sum(test_loss_better_than_original)/num_samples}')
log.info(f'better_than_average_{sum(test_loss_better_than_average)/num_samples}')

log.info(f'sharpness_before_{sharpness_blurred_image_average}')
log.info(f'sharpness_after_{sharpness_corrected_image_average}')
log.info(f'sharpness_better_than_original_{sum(sharpness_better_than_original)/num_samples}')
log.info(f'sharpness_better_than_average_{sum(sharpness_better_than_average)/num_samples}')

log.info(f'rms_before_{rms_original_aberration_average}')
log.info(f'rms_after_{rms_aberration_difference_average}')
log.info(f'rms_better_than_original_{sum(rms_better_than_original)/num_samples}')
log.info(f'rms_better_than_average_{sum(rms_better_than_average)/num_samples}')

log.info(f'std_sharpness_before_{std_sharpness_before}')
log.info(f'std_sharpness_after_{std_sharpness_after}')

log.info(f'std_phase_rms_before_{std_rms_before}')
log.info(f'std_phase_rms_after_{std_rms_after}')

log.info(f'std_train_loss_{std_test_loss}')

timestamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-3]
full_file_name_figures = Test_figure_path / f'Test_trained_model_MLAO_{model_name}_idx_check_{idx_check}_rows_checked_{len(row_idx)}_magnitude_{magnitude:0.3f}_time_{timestamp}_NewDistribution.png'

fig_M, axs_M = plt.subplots(2, 3, sharex='all', sharey='all', figsize=(18, 8))
axs_M[0, 0].imshow(complex2rgb(np.exp(1j*original_aberration)), origin='lower')
axs_M[0, 0].set(title=f"Row{idx_check}_num_iteration_{num_iteration}_original_aberration_RMS_({rms_original_aberration_average:0.3f})")
axs_M[0, 1].imshow(complex2rgb(np.exp(1j * predicted_aberration_row)), origin='lower')
axs_M[0, 1].set(title=f"predicted_aberration_Averaged_RMS_({rms_predicted_aberration_average:0.3f})")
axs_M[0, 2].imshow(complex2rgb(np.exp(1j*aberration_difference)), origin='lower')
axs_M[0, 2].set(title=f"aberration_difference_Averaged_RMS_({rms_aberration_difference_average:0.3f})")

axs_M[1, 0].imshow(original_image)
axs_M[1, 0].set(title=f"Averaged_{num_samples}_original_image_sharpness_({sharpness_original_image:0.3f})")
axs_M[1, 1].imshow(blurred_image_average)
axs_M[1, 1].set(title=f"blurred_image_sharpness_({sharpness_blurred_image_average:0.3f})")
axs_M[1, 2].imshow(corrected_image_average)
axs_M[1, 2].set(title=f"corrected_image_sharpness_({sharpness_corrected_image_average:0.3f})")

plt.savefig(full_file_name_figures)

# std_sharpness_before = np.std(np.array(sharpness_before_for_std))
# std_sharpness_after = np.std(np.array(sharpness_after_for_std))
# std_rms_before = np.std(np.array(phase_rms_before_for_std))
# std_rms_after = np.std(np.array(phase_rms_after_for_std))
# std_train_loss = np.std(np.array(test_loss_for_std))
#
# log.info(f'std_sharpness_before_{std_sharpness_before}')
# log.info(f'std_sharpness_after_{std_sharpness_after}')
#
# log.info(f'std_phase_rms_before_{std_rms_before}')
# log.info(f'std_phase_rms_after_{std_rms_after}')
#
# log.info(f'std_train_loss_{std_train_loss}')
# #  metric: RMS of phase after correction, sharpness of images

