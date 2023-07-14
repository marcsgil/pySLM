import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from projects.adaptive_optics.imaging_system import ImagingSystem

from optics.utils import ft
from optics.utils.display import complex2rgb

from projects.adaptive_optics import log
from network import AdaptiveOpticsNet


def load_model(M=2, N=10, model_name='ast2'):
    model_save_name = Path('E:/Adaptive Optics/MLAO/') / f'CNN_Model_name_{model_name}_M_{M}_N_{N}_Num_epochs_{1000}_CV_kernel_size_{3}_learningrate_{0.002}_in_features{250}_timestamp_2023-03-21_12-57-38.087.pt'

    return torch.jit.load(model_save_name)


def rms(pupil):
    return np.linalg.norm(pupil) / np.sqrt(pupil.size * np.pi / 4 / 4)


if __name__ == '__main__':
    total_nb_zernike = 18

    corrected_modes_indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    bias_modes = [3]
    bias_depths = [1]

    nb_iterations = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    torch_dtype = torch.float32
    log.info(f'Using PyTorch device {device}...')

    # Define an imaging system with an 'unknown' aberration
    imaging_system = ImagingSystem(ft.Grid(np.full(2, 128), 0.2e-6), simulate_noise=True)

    def get_pseudo_psfs(estimated_coefficients):
        """Calculate the pseudo_psfs to feed to the neural network model."""
        img_grid = ft.Grid(np.full(2, 32), center=imaging_system.grid.shape // 2)
        pseudo_psf_all_modes = []
        for bias_mode in bias_modes:
            for bias_depth in bias_depths:
                pseudo_psf = imaging_system.pseudo_psf(-estimated_coefficients, bias_mode, bias_depth)
                cropped_pseudo_psf = pseudo_psf[:, img_grid[0], img_grid[1]]
                pseudo_psf_all_modes.append(cropped_pseudo_psf)
            # Store valid data
        pseudo_psf_all_modes = np.asarray(pseudo_psf_all_modes)
        pseudo_psf_all_modes = pseudo_psf_all_modes.reshape([pseudo_psf_all_modes.shape[0]*pseudo_psf_all_modes.shape[1], *pseudo_psf_all_modes.shape[-2:]])
        return np.float32(pseudo_psf_all_modes.real)

    log.info(f'Loading the model...')
    model = load_model(M=2, N=len(corrected_modes_indices), model_name='ast2').to(device=device, dtype=torch_dtype)

    log.info(f'Testing...')
    imaging_system.aberration_coefficients = np.zeros(total_nb_zernike)
    imaging_system.aberration_coefficients[4] = 0.66
    log.info(f'Initialized an imaging system with aberration coefficients: {imaging_system.aberration_coefficients}.')

    predicted_coefficients = np.zeros(total_nb_zernike)
    for iteration_idx in range(nb_iterations):
        # Simulate image formation and pseudo-psfs
        pseudo_psf_all_modes = get_pseudo_psfs()
        test_data_iter = torch.from_numpy(pseudo_psf_all_modes).to(device=device, dtype=torch_dtype)  # convert double to float32 with np.float32, shape [len(bias_modes), len(bias_depths), 2, *img_ranges.shape]
        # Predict the coefficients of this imaging system
        output = model(test_data_iter)
        predicted_coefficients[corrected_modes_indices] += output.detach().cpu().numpy().ravel()
        log.info(f'predicted coefficients = {predicted_coefficients}')

    log.info('Dislaying results...')
    fig_M, axs = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(18, 8))
    for ax, ab_phase in zip(axs, [imaging_system.phase_from_zernike_polynomial(),
                                  imaging_system.phase_from_zernike_polynomial(predicted_coefficients, include_system_aberration=False),
                                  imaging_system.phase_from_zernike_polynomial(-predicted_coefficients)
                                  ]):
        ax.imshow(complex2rgb(np.exp(1j * ab_phase)), origin='lower')
        msg = f'rms = {rms(ab_phase):0.3f})'
        ax.set(title=msg)
        log.info(msg)

    log.info('Done! Close window to exit.')
    plt.show()
