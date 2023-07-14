import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import gc

from optics.utils.ft import Grid
from optics.utils.display import complex2rgb
from projects.confocal_interference_contrast.complex_phase_image import registration_calibration  #  get_registration
from optics.utils.ft import register

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def roll_ft_gpu(images_ft, shift):
    shift = torch.tensor(shift).flatten()
    axes = torch.arange(-torch.numel(shift), 0)

    grid_k_step = torch.tensor([2 * torch.pi * shift[idx] / dim for idx, dim in enumerate(images_ft.size()[axes[0]:])])
    grid_k = Grid(tuple(images_ft.size()[axes[0]:]), step=grid_k_step.numpy(), origin_at_center=False)
    for phase_range in [torch.from_numpy(_).to(device=device) for _ in grid_k]:
        images_ft = images_ft.type(torch.complex64)
        images_ft *= torch.exp(-1j * phase_range)
    return images_ft


def get_complex_image(interference_image, registration):
    grid = Grid(shape=interference_image.shape[-2:])
    f = torch.sqrt(sum([torch.from_numpy(_) ** 2 for _ in grid.f]))

    frequency_separation = np.linalg.norm(registration.shift * grid.f.step).item()
    low_pass = (f < frequency_separation / 2).to(device=device)

    interference_image[:] = torch.fft.ifftshift(torch.fft.ifftn(
        torch.fft.fftn(roll_ft_gpu(torch.fft.fftshift(interference_image) / registration.factor, registration.shift
                                   ), dim=(-1, -2)) * low_pass, dim=(-1, -2)))
    return interference_image


def scan2dic(scan_data, fringe_registrations: list = None):
    """Converts DIC scan fringe data to DIC phase image."""
    print(f'The device used is {device}...')
    scan_data[1] = scan_data[1, ..., ::-1]
    nb_cams_used = scan_data.shape[0]
    scan_shape = np.array(scan_data.shape[1:3])

    if fringe_registrations is None:
        fringe_registrations = []
        # Upload registration_lock from json file
        for cam_idx in range(nb_cams_used):
            registration_lock_image = scan_data[cam_idx, scan_shape[0] // 2, scan_shape[1] // 2]
            # registration_lock_image = scan_data[cam_idx, 0, 0]
            # registration_lock = get_registration(f"HV_DIC_registration_CAM{idx}")  # loads a pre-calculated registration_lock
            fringe_registrations.append(
                registration_calibration(registration_lock_image, f"HV_DIC_registration_CAM{cam_idx}"))

    cam_shift_registration = register(scan_data[0, 0, 0], scan_data[1, 0, 0])
    shift = cam_shift_registration.shift

    dic_image = np.zeros(scan_shape, dtype=np.complex64)
    start_time = time.perf_counter()
    for images_line_idx in range(scan_shape[0]):
        images_line_ft = torch.fft.fftn(torch.tensor(scan_data[0][images_line_idx], dtype=torch.complex64), dim=(-1, -2)).to(device=device)
        images_line = torch.abs(torch.fft.ifftn(roll_ft_gpu(images_line_ft, -shift), dim=(-1, -2))).type(dtype=torch.complex64)

        left_arg = get_complex_image(images_line, fringe_registrations[0])
        right_arg = torch.tensor(scan_data[1][images_line_idx], dtype=torch.complex64, device=device)
        right_arg[:] = get_complex_image(right_arg, fringe_registrations[1])

        # dic_image[images_line_idx] = torch.exp(1j * torch.angle(torch.sum(  # Dot product with matrix product
        #     torch.conj(torch.swapaxes(left_arg, -1, -2)) @ right_arg, dim=(-1, -2)))).cpu().numpy()
        dic_image[images_line_idx] = torch.sum((torch.abs(left_arg) + torch.abs(right_arg)) * torch.exp(1j * torch.angle(  # Dot product with matrix product
            torch.conj(left_arg) * right_arg)), dim=(-1, -2)).cpu().numpy()
        print(f'{images_line_idx+1}/{scan_shape[0]}')
    del left_arg, right_arg
    torch.cuda.empty_cache()
    gc.collect()
    print(f'DIC image calculated in {time.perf_counter() - start_time:.3g} seconds...')

    return dic_image


if __name__ == "__main__":
    folder = r"C:\Users\lab\OneDrive - University of Dundee\Documents\lab\code\python\optics\projects\confocal_interference_contrast/"
    name = '0.8um_separated_2022-11-10_17-00-31.npz'
    scan_data = np.load(folder + name)["camera_data"]
    dic_image = scan2dic(scan_data)

    while True:
        saturation = input("Set saturation: ")
        fig, axs = plt.subplots()
        axs.imshow(complex2rgb(dic_image, float(saturation)))
        plt.show(block=True)
