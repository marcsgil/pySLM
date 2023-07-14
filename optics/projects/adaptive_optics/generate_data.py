#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# for creating validation set
# from sklearn.model_selection import train_test_split
import time
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from pathlib import Path

from projects.adaptive_optics import log
from optics.utils import ft
# from optics.utils.display import complex2rgb, grid2extent

from projects.adaptive_optics.imaging_system import ImagingSystem


if __name__ == "__main__":
    # Generate .npz

    output_folder = Path.home() / Path('E:/Adaptive Optics/Test of model from multiple batches/Matched resolution/')
    log.info(f'Results will be saved in {output_folder}.')
    output_folder.mkdir(parents=True, exist_ok=True)

    simulate_noise = True
    background_noise_level = 0.02

    # Define the imaging system
    # todo: Is 164e-6 actually used? If not, it can be removed.
    imaging_system = ImagingSystem(ft.Grid(np.full(2, 128), 164e-6), simulate_noise=simulate_noise, background_noise_level=background_noise_level)
    # from optics.utils import reference_object
    # imaging_system.object = np.asarray(reference_object.usaf1951(imaging_system.grid.shape[-2:], scale=1.0)) / 255.0  # Maximum value = 1

    # Define the aberrations
    nb_modes = 24
    nb_batches = 50
    magnitude = 2 / (1*np.sqrt(3))
    batch_size = 10000  # how many groups of implementations
    seed = 'none'
    rng = np.random.Generator(np.random.PCG64(seed=1))  # For the noise generation

    corrected_modes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    bias_modes = [3]
    uncorrected_modes = [17]

    # corrected_modes = [3, 5, 6, 7, 8, 9, 11, 12, 13, 22]
    # bias_modes = [3]
    # uncorrected_modes = [23]

    bias_depths = [1]
    input_channel = len(bias_modes)*len(bias_depths)*2

    start_time = time.perf_counter()
    img_grid = ft.Grid(np.full(2, 32), center=imaging_system.grid.shape // 2)

    std_zernike_coeffs = np.zeros(nb_modes)
    std_zernike_coeffs[corrected_modes] = magnitude
    std_zernike_coeffs[uncorrected_modes] = 0.01 / np.sqrt(3)
    # 0.01 / np.sqrt(3)
    # std_zernike_coeffs /= 2

    for batch_idx in range(nb_batches):
        training_data = []
        train_label = []
        log.info(f'Generating patch {batch_idx}...')
        amplitudes = rng.uniform(size=batch_size)[..., np.newaxis] * std_zernike_coeffs
        coefficient_array = rng.normal(size=[batch_size, std_zernike_coeffs.size]) * amplitudes
        # coeff_rms = np.linalg.norm(coefficient_array, axis=-1) / coefficient_array.shape[-1]
        # plt.hist(coeff_rms, 100)
        # plt.show()
        # print(np.sum((coefficient_array), axis=1))
        # exit()
        # coefficient_array = rng.uniform(-1, 1, size=[number_of_zernike_superpositions_per_patch, std_zernike_coeffs.size]) * std_zernike_coeffs

        train_index = 0
        for superpos_idx in range(batch_size):
            pseudo_psf_all_modes = []
            for bias_mode in bias_modes:
                for bias_depth in bias_depths:
                    pseudo_psf = imaging_system.pseudo_psf(coefficient_array[superpos_idx], bias_mode, bias_depth)
                    cropped_pseudo_psf = pseudo_psf[:, img_grid[0], img_grid[1]]
                    # fig, axs = plt.subplots(1, 2)
                    # for _, img in enumerate(cropped_pseudo_psf):
                    #     axs[_].imshow(np.abs(img))
                    #     axs[_].set_title(f'{coefficient_array[superpos_idx]}')
                    # plt.show()

                    pseudo_psf_all_modes.append(cropped_pseudo_psf)
                    # print(np.array(pseudo_psf_all_modes).shape)

            # Store valid data
            if not np.any(np.isnan(pseudo_psf_all_modes)):
                training_data.append(pseudo_psf_all_modes)  # shape [len(bias_modes), len(bias_depths), 2, *img_ranges.shape]
                # print(np.array(training_data).shape)
                train_label.append(coefficient_array[superpos_idx, :])
            else:  # Remove invalid data
                log.warning(f'Skipping a pseudo psf due to {np.sum(pseudo_psf_all_modes)} NaN.')

            # training_data[superpos_idx] = images

        total_time = time.perf_counter() - start_time
        log.info(f'Total time {total_time:0.3f}s for {batch_size}x{len(bias_modes)}x{len(bias_depths)}x2. Time per superposition: {total_time / batch_size / 1e-3:0.3f}ms.')

        training_data = np.asarray(training_data)
        training_data = training_data.reshape([training_data.shape[0], -1, *training_data.shape[-2:]])
        train_label = np.asarray(train_label)

        settings = dict(nb_batches=nb_batches,
                        batch_size=batch_size,
                        corrected_modes=corrected_modes,
                        bias_modes=bias_modes,
                        train_index=train_index,
                        bias_depths=bias_depths,
                        background_noise_level=background_noise_level,
                        magnitude=magnitude,
                        simulate_noise=simulate_noise,
                        seed=seed
                        )

        timestamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-3]
        full_file_name = output_folder / f'PseudoPSF_Match_resolution_batchidx{batch_idx}_batchsize{batch_size}_nb_batches{nb_batches}_corrected_modes{corrected_modes}_biasmode{bias_modes}_biasdepths{bias_depths}_magnitude_{magnitude:0.3f}_simulate_noise_{simulate_noise}_background_noise_level_{background_noise_level}_seed_{seed}.npz'

        log.info(f'Saving results to {full_file_name}...')
        np.savez(full_file_name, train_data=training_data, train_label=train_label, settings=settings)
    log.info('Saved everything!')
    print(np.array(training_data).shape)
    log.info('Displaying...')
    fig, axs = plt.subplots(2, 6, sharex='all', sharey='all')
    for _, ax_bias in enumerate(axs.transpose()):
        ax_bias[0].imshow(training_data[_, 0])
        ax_bias[1].imshow(training_data[_, 1])
    log.info('Done, close window to exit.')
    plt.show()
