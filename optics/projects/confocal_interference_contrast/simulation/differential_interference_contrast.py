import numpy as np
import matplotlib.pyplot as plt

import torch  # Only for the restore function

from projects.confocal_interference_contrast.simulation import log
from optics.utils import ft
from optics.instruments.objective import Mitutoyo
from optics.utils.display import complex2rgb, grid2extent, format_image_axes
from optics.calc.psf import PSF


def restore(measured_intensities, initial_guess, grid, spot_positions, spot_amplitudes, psf_field_array, polarization_phasors, ground_truth) -> np.ndarray:
    dtype = torch.complex64
    if torch.cuda.is_available():
        try:
            log.info('Pytorch with CUDA installed, testing GPU.')
            device = torch.device('cuda')
            torch.fft.fft(torch.zeros(1).to(dtype=dtype, device=device))
            log.info('Using GPU')
        except RuntimeError as re:
            device = torch.device('cpu')
            log.warning(f'{re}\nCuda installed but cannot use GPU, falling back to CPU!')
    else:
        device = torch.device('cpu')

    def roll(subject, shift):
        axes = range(-shift.size, 0)
        translated_ft = torch.fft.fft2(subject)

        grid_k_step = np.zeros_like(translated_ft.shape, dtype=np.float64)
        grid_k_step[axes] = 2 * np.pi * shift / np.asarray(translated_ft.shape)[axes]

        grid_k = ft.Grid(translated_ft.shape, step=grid_k_step, origin_at_center=False)
        for phase_range, step in zip(grid_k, grid_k.step):
            translated_ft *= torch.exp(-1j * torch.from_numpy(phase_range))

        return torch.fft.ifft2(translated_ft)

    # Precalculate some values
    distance_to_edge = np.minimum(grid[1] - grid[1].ravel()[0], grid[1].ravel()[-1] - grid[1])
    edge_mask = torch.from_numpy(np.maximum(0.0, 1.0 - distance_to_edge / (1 * grid.step[1])))
    edge_mask_sum = torch.sum(edge_mask)

    class PhaseOnlyDifferentialPhaseContrastNet(torch.nn.Module):
        def __init__(self, scanned_dpc):
            super().__init__()
            scanned_dpc = torch.from_numpy(scanned_dpc)
            self.__weights = torch.abs(scanned_dpc)
            self.target = torch.stack([
                self.__weights,
                torch.angle(scanned_dpc),
                self.__weights
                ])

        def forward(self, phase):
            """
            :param phase: The phase of the reconstructed object.
            :return: The weighted gradient in dimension 0 and 1, weighed by the amplitudes.
            """
            return self.__weights * torch.stack([
                torch.diff(phase, dim=0, prepend=torch.from_numpy(np.zeros([1, phase.shape[1]]))) * 0,
                torch.diff(phase, dim=1, prepend=torch.from_numpy(np.zeros([phase.shape[0], 1]))),
                edge_mask * phase * 0
            ])

    class DifferentialPhaseContrastNet(torch.nn.Module):
        def __init__(self, psf_field_array, polarization_phasors):
            super().__init__()
            self.__otf = torch.fft.fft2(torch.from_numpy(ft.ifftshift(psf_field_array)).to(dtype=dtype, device=device))
            self.__polarization_phasors_t = torch.from_numpy(polarization_phasors)[:, np.newaxis, np.newaxis]

        def forward(self, _):
            scanned_ground_truth_t = torch.fft.ifft2(self.__otf * torch.fft.fft2(_))
            transmitted_field_offsets_t = [amp * roll(scanned_ground_truth_t, -sep/2)
                                           for amp, sep in zip(spot_amplitudes, spot_positions / grid.step[0])]
            interference_fields_t = transmitted_field_offsets_t[0] + transmitted_field_offsets_t[1] * self.__polarization_phasors_t
            return torch.abs(interference_fields_t) ** 2  # without the noise

    # The measurement
    target = torch.from_numpy(measured_intensities).to(dtype=dtype, device=device)   # with Gaussian error of noise_level
    # For the simple version
    interference_intensities_ft = ft.fft(measured_intensities, axis=0)
    scanned_dpc = interference_intensities_ft[1] / interference_intensities_ft[0]  # The relative contrast and phase

    def loss_fn_old(object_t, output):
        """
        A simple loss function. This seems to work well in the first iterations, but it diverges from the ground truth in later iterations.

        :param object_t: The reconstructed object.
        :param output: The measured data according to the model.
        :return: A floating point value indicating the error of this object.
        """
        output_diff = torch.linalg.norm(output - target) ** 2 / torch.linalg.norm(target) ** 2
        grad_error_0 = torch.linalg.norm(torch.diff(object_t, dim=0) / grid.step[0]) ** 2
        return output_diff + 1e-20 * grad_error_0

    def loss_fn(object_t, output):
        """
        :param object_t: The reconstructed object.
        :param output: The measured data according to the model.
        :return: A floating point value indicating the error of this object.
        """
        output_diff = (torch.linalg.norm(output**0.5 - target**0.5) / np.prod(target.shape))
        grad_0 = torch.diff(object_t, dim=0)
        grad_error_0 = (torch.linalg.norm(grad_0) / grid.size) ** 2
        # abs_grad_error_0 = (torch.linalg.norm(torch.abs(grad_0)) / grid.size) ** 2
        # grad_1 = torch.diff(object_t, dim=1)
        # grad_error_1 = (torch.linalg.norm(grad_1) / grid.size) ** 2
        edge_error = (torch.linalg.norm(torch.angle(object_t) * edge_mask) / edge_mask_sum) ** 2
        # print(f'diff:{output_diff:0.3e}  grad0:{grad_error_0:0.3e}  grad1:{grad_error_1:0.3e}  abs_grad:{abs_grad_error_0:0.3e}  edge: {edge_error:0.3e}')
        return output_diff + 1e-2 * grad_error_0 + edge_error   #+ 1e-18 * grad_error_1

    model = DifferentialPhaseContrastNet(psf_field_array, polarization_phasors)
    # model = PhaseOnlyDifferentialPhaseContrastNet(scanned_dpc)
    model.to(device)

    def phase_only_loss_fn(object_t, output):
        return torch.linalg.norm(torch.abs(object_t) - torch.abs(torch.from_numpy(scanned_dpc))**0.5) ** 2 \
               + torch.linalg.norm(output - model.target) ** 2

    # phase_shifter = np.exp(2j * np.pi * np.random.rand(grid.shape[0], 1) / 16)
    object_t = torch.from_numpy(ground_truth).to(dtype=dtype)  # for testing only
    # object_t = torch.from_numpy(ground_truth).to(dtype=dtype) + torch.randn(ground_truth.shape) + 1j * torch.randn(ground_truth.shape)
    # object_t = torch.from_numpy(initial_guess).to(dtype=dtype, device=device)
    # object_t.requires_grad = True
    object_t_phase = torch.angle(object_t)
    object_t.requires_grad = True
    optimizer = torch.optim.AdamW([object_t], lr=1e-2)
    max_iter = 5000
    for _ in range(max_iter):
        optimizer.zero_grad()
        # Calculate error
        output = model(object_t)
        loss = loss_fn(object_t, output)
        # loss = phase_only_loss_fn(object_t_phase, output)
        # Back propagation
        loss.backward()
        optimizer.step()
        # Display progress
        if _ % 100 == 0:
            with torch.no_grad():
                # object_t = torch.abs(object_t) * torch.exp(1j * object_t_phase)  # convert phase only to complex
                actual_error = torch.linalg.norm(object_t - ground_truth)
                log.info(f'Restoration iteration {_}/{max_iter}: loss = {np.sqrt(loss.detach().numpy()):0.6f}, actual error = {actual_error.detach().numpy():0.6f}')

    # with torch.no_grad():
    #     object_t = torch.abs(object_t) * torch.exp(1j * object_t_phase)  # convert phase only to complex
    reconstructed_dpc = object_t.detach().numpy()

    return reconstructed_dpc


def main():
    include_noise = False
    restore_object = True
    
    wavelength = 488e-9
    refractive_index_microsphere = 1.463  # Are the spheres Silica? https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
    # Johnson's baby oil contains Liquid paraffin (n = 1.467) and Isopropyl palmitate (n = 1.438) (where Silica is listed as n = 1.544 (https://www.chembk.com/))
    # Mineral oil could be between 1.46 and 1.48 (unsure about wavelength): https://opticsmag.com/what-is-the-refractive-index-of-oil/
    refractive_index_medium = 1.43
    diameter_sphere = 10e-6

    illumination_objective = Mitutoyo(100, 0.55)
    detection_objective = Mitutoyo(50, 0.55)  # Ignored at the moment, this would block some light and thus change the intensity within the image
    # spot_separation = 0.5e-6
    noise_level = 0.05 * include_noise  # todo: improve noise model
    spot_separation = wavelength / (2 * illumination_objective.numerical_aperture)
    spot_amplitudes = np.array([0.5, 0.5])  # Can be complex
    spot_positions = np.array([-0.5, 0.5]) * spot_separation  # Assume on axis 0 for now
    stage_step_size = spot_separation

    display_upsampling_factor = 4

    grid_scan = ft.Grid(step=stage_step_size, extent=2*diameter_sphere * np.array([1, 4/3]))
    grid_display_hr = ft.Grid(step=grid_scan.step / display_upsampling_factor, extent=grid_scan.extent)  # High resolution for a clearer view
    log.info(f'Scanning spots separated by {spot_separation / 1e-6:0.3f}um, in steps of {stage_step_size / 1e-6:0.3f}um.')
    polarization_phasors = np.exp(2j * np.pi / 4 * np.arange(4))

    log.info('Creating simulated object in sample...')
    radius_in_plane_squared_hr = sum(_ ** 2 for _ in grid_display_hr)
    thickness_hr = 2 * np.sqrt(np.maximum(0, (diameter_sphere / 2) ** 2 - radius_in_plane_squared_hr))
    optical_path_in_rad_hr = 2 * np.pi * (refractive_index_microsphere - refractive_index_medium) / wavelength * thickness_hr
    ground_truth_hr = 0.75 * np.exp(1j * optical_path_in_rad_hr)
    ground_truth = ft.interp(ground_truth_hr, 1.0 / display_upsampling_factor)

    log.info('Simulating scanned differential interference contrast...')
    log.info('Calculating PSF...')
    psf = PSF(objective=illumination_objective, vacuum_wavelength=wavelength)
    psf_field_array = psf(*ft.Grid(shape=[1, *grid_scan.shape], step=[0, *grid_scan.step], center=0.0))[0, 0]  # drop the polarization (for now) and z
    log.info('Calculating the field as the field scans the ground truth...')
    # # todo: model the physics of the attenuation properly
    # gradient = np.sqrt(sum((np.diff(np.exp(1j * np.angle(ground_truth)), axis=_, prepend=1.0)
    #                         / (s * 2 * np.pi / wavelength * detection_objective.numerical_aperture)) ** 2 for _, s in enumerate(grid_scan.step)))
    # gradient[[0, -1], [0, -1]] = 0.0
    # gradient_attenuation = np.sqrt(np.maximum(0.0, 1.0 - (np.abs(gradient) * 2) ** 2))
    scanned_ground_truth = ft.cyclic_conv(ft.ifftshift(psf_field_array), ground_truth)

    log.info('Simulating scanning...')
    transmitted_field_offsets = [amp * ft.roll(scanned_ground_truth, -sep/2)
                                 for amp, sep in zip(spot_amplitudes, spot_positions / grid_scan.step[0])]
    interference_fields = [transmitted_field_offsets[0] + _ * transmitted_field_offsets[1] for _ in polarization_phasors]
    interference_intensities = np.abs(interference_fields) ** 2
    measured_intensities = interference_intensities + noise_level * np.random.randn(*interference_intensities.shape)  # todo: This is a quite crude noise model
    log.info('Reconstructing differential phase contrast.')
    # For regular separation of the polarization phase differences:
    interference_intensities_ft = ft.fft(measured_intensities, axis=0)
    scanned_dpc = interference_intensities_ft[1] / interference_intensities_ft[0]  # The relative contrast and phase
    # todo: Irregular separation would be more something like this.
    # total_intensity = np.sum(measured_intensities, axis=0)
    # scanned_dpc = sum(_ * mi for _, mi in zip(polarization_phasors.conj(), measured_intensities)) / total_intensity

    log.info('Naively integrating phase in scanned differential interference contrast image.')  # todo: use prior such as gradient to fix the striping.
    integrated_phase = np.cumsum(np.angle(scanned_dpc), axis=1) * grid_scan.step[1] / (np.diff(spot_positions) / 2)  # todo: Why is the / 2 necessary?
    integrated_dpc = np.abs(scanned_dpc) * np.exp(1j * integrated_phase)
    reverse_integrated_phase = np.cumsum(-np.angle(scanned_dpc)[::-1, :], axis=1)[::-1, :] * grid_scan.step[1] / (np.diff(spot_positions) / 2)  # todo: Why is the / 2 necessary?
    reverse_integrated_dpc = np.abs(scanned_dpc) * np.exp(1j * reverse_integrated_phase)

    if restore_object:
        log.info('Digitally restoring the object from the differential phase contrast...')
        # initial_guess = ground_truth  # Check whether this at least works
        initial_guess = (integrated_dpc + reverse_integrated_dpc) / 2.0
        reconstructed_dpc = restore(measured_intensities, initial_guess, grid_scan, spot_positions, spot_amplitudes, psf_field_array, polarization_phasors, ground_truth)

    log.info('Interpolating for display...')
    scanned_ground_truth_hr = ft.interp(scanned_ground_truth, display_upsampling_factor)
    scanned_dpc_hr = ft.interp(scanned_dpc, display_upsampling_factor)
    integrated_dpc_hr = ft.interp(integrated_dpc, display_upsampling_factor)
    if restore_object:
        reconstructed_dpc_hr = ft.interp(reconstructed_dpc, display_upsampling_factor)

    log.info('Displaying results...')
    fig, axs = plt.subplots(2, 4 + restore_object)
    axs[0, 0].imshow(complex2rgb(ground_truth_hr), extent=grid2extent(grid_display_hr))
    format_image_axes(axs[0, 0], 'Ground Truth', scale=grid_scan.extent[1] / 5, unit='m', white_background=True)
    axs[0, 1].imshow(complex2rgb(scanned_ground_truth_hr), extent=grid2extent(grid_display_hr))
    format_image_axes(axs[0, 1], 'Scanned GT', white_background=True)
    axs[0, 2].imshow(complex2rgb(scanned_dpc_hr), extent=grid2extent(grid_display_hr))
    format_image_axes(axs[0, 2], 'Scanned DPC')
    axs[0, 3].imshow(complex2rgb(integrated_dpc_hr), extent=grid2extent(grid_display_hr))
    format_image_axes(axs[0, 3], 'Integrated')
    if restore_object:
        axs[0, 4].imshow(complex2rgb(reconstructed_dpc_hr), extent=grid2extent(grid_display_hr))
        format_image_axes(axs[0, 4], 'Reconstructed')
    axs[1, 0].plot(grid_display_hr[1].ravel() / 1e-6, np.unwrap(np.angle(ground_truth_hr[grid_display_hr.shape[0]//2, :])))
    axs[1, 0].plot(grid_display_hr[1].ravel() / 1e-6, np.abs(ground_truth_hr[grid_display_hr.shape[0]//2, :]))
    axs[1, 0].set(xlabel='y  [$\mu$m]', ylabel='$\phi$  [rad],  |A|   [a.u.]')
    axs[1, 1].plot(grid_display_hr[1].ravel() / 1e-6, np.unwrap(np.angle(scanned_ground_truth_hr[grid_display_hr.shape[0]//2, :])))
    axs[1, 1].plot(grid_display_hr[1].ravel() / 1e-6, np.abs(scanned_ground_truth_hr[grid_display_hr.shape[0]//2, :]))
    # axs[1, 1].plot(grid_display_hr[0].ravel() / 1e-6, np.unwrap(np.angle(scanned_ground_truth[:, grid_display_hr.shape[1]//2])))
    # axs[1, 1].set(xlabel='y  [$\mu$m]', ylabel='$\phi$  [rad],  |A|   [a.u.]')
    axs[1, 2].plot(grid_display_hr[1].ravel() / 1e-6, np.unwrap(np.angle(scanned_dpc_hr[grid_display_hr.shape[0]//2, :])))
    axs[1, 2].plot(grid_display_hr[1].ravel() / 1e-6, np.abs(scanned_dpc_hr[grid_display_hr.shape[0]//2, :]))
    # axs[1, 2].plot(grid_display_hr[0].ravel() / 1e-6, np.unwrap(np.angle(scanned_dpc_hr[:, grid_display_hr.shape[1]//2])))
    axs[1, 2].set(xlabel='y  [$\mu$m]', ylabel='$\Delta\phi$  [rad],  |A|   [a.u.]')
    axs[1, 3].plot(grid_display_hr[1].ravel() / 1e-6, np.unwrap(np.angle(integrated_dpc_hr[grid_display_hr.shape[0]//2, :])))
    axs[1, 3].plot(grid_display_hr[1].ravel() / 1e-6, np.abs(integrated_dpc_hr[grid_display_hr.shape[0]//2, :]))
    # axs[1, 3].plot(grid_display_hr[0].ravel() / 1e-6, np.unwrap(np.angle(integrated_dpc[:, grid_display_hr.shape[1]//2])))
    # axs[1, 3].set(xlabel='y  [$\mu$m]', ylabel='$\phi$  [rad],  |A|   [a.u.]')
    if restore_object:
        axs[1, 4].plot(grid_display_hr[1].ravel() / 1e-6, np.unwrap(np.angle(reconstructed_dpc_hr[grid_display_hr.shape[0]//2, :])))
        axs[1, 4].plot(grid_display_hr[1].ravel() / 1e-6, np.abs(reconstructed_dpc_hr[grid_display_hr.shape[0]//2, :]))
        # axs[1, 4].plot(grid_display_hr[0].ravel() / 1e-6, np.unwrap(np.angle(reconstructed_dpc_hr[:, grid_display_hr.shape[1]//2])))
        # axs[1, 4].set(xlabel='y  [$\mu$m]', ylabel='$\phi$  [rad],  |A|   [a.u.]')

    plt.draw_all()
    log.info('Done!')
    plt.show()


if __name__ == '__main__':
    main()
