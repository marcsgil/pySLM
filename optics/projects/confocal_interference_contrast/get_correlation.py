import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import re
import os
from os.path import isfile, join
from typing import Union

from optics.utils.display import complex2rgb
from optics.utils.ft import Grid, Registration, register
from projects.confocal_interference_contrast.complex_phase_image import Interferogram, registration_calibration, \
    get_registration


def get_file(serial_nb: str):
    path = r'C:\\Users\\Laurynas\\OneDrive - University of Dundee\\Work\\Data\\Scans\\Interference_scans\\'
    files = [file for file in os.listdir(path) if isfile(join(path, file))]
    search = re.compile(fr'{serial_nb}')
    file_type = re.compile(r'\.npy')
    for file in files:
        target = (search.search(file) is not None) * (file_type.search(file) is not None)
        if target:
            target_file = file
            return np.array(np.load(path + target_file))
    raise FileNotFoundError


def gaussian_func(shape):
    """ symmetrical 2D Gaussian function"""
    grid = Grid(shape=shape)
    sigma = np.mean(shape) / 4
    z = (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -(grid[0] ** 2 + grid[1] ** 2) / (2 * sigma ** 2))
    z /= z.max()
    return z


def apply_mask(image):
    shape = image.shape
    spot_radius = 500  # pixels
    grid = Grid(shape=shape)
    pixel_grid = np.sqrt(grid[0] ** 2 + grid[1] ** 2)
    mask = pixel_grid < spot_radius
    centered_image = register(image, gaussian_func(shape)).image
    masked_image = centered_image * mask
    return masked_image.astype(float)


def get_correlation(images: Union[np.ndarray, list]):
    """getting correlations from a list of images"""
    # Aligning the two images and applying mask
    processed_images = np.array(images).copy()
    correlation = []
    # TODO the following 2 lines are used for fft registering the image
    # ndim = len(images[0].shape)
    # ref_img_ft = np.fft.fftn(np.fft.ifftshift(processed_images[0], axes=np.arange(ndim)), axes=np.arange(ndim))
    for idx, image in enumerate(images):
        # processed_images[idx] = apply_mask(register(image, images[0]).image)
        registration = register(images[idx], images[0])
        processed_images[idx] = registration.image * registration.factor
        # TODO repeating registration in Fourier image in the next 3 lines
        # iter_img_ft = np.fft.fftn(np.fft.ifftshift(processed_images[idx], axes=np.arange(ndim)), axes=np.arange(ndim))
        # registration_ft = register(iter_img_ft, ref_img_ft)
        # processed_images[idx] = np.fft.ifftn(np.fft.fftshift(registration_ft.image * registration_ft.factor,
        #                                                      axes=np.arange(ndim)), axes=np.arange(ndim))
        processed_images[idx] /= np.linalg.norm(processed_images[idx])
        correlation.append(np.vdot(processed_images[0].ravel(), processed_images[idx].ravel()))

    return correlation


def get_phase_correlation(images: list):
    # Calculating registrations
    registrations = []  # np.zeros(len(images), dtype=object)
    for image in images:
        registrations.append(registration_calibration(image))

    # Getting the phase image
    complex_phase_images = np.empty(np.shape(images), dtype=np.complex)
    for idx in range(len(images)):
        complex_phase_images[idx, ...] = np.array(Interferogram(images[idx], registration=registrations[idx]))
    # complex_phase_images = [*complex_phase_images, -complex_phase_images[0] * 0.5j]  # Testing correlation

    return get_correlation(complex_phase_images)


if __name__ == "__main__":
    images = [get_file(f"FS{idx}") for idx in range(1, 3)]
    print([np.abs(correlation) for correlation in get_phase_correlation(images)])
    # print([np.abs(correlation) for correlation in get_correlation(np.array(images)[:, 0, :])])
    gaussian = gaussian_func(images[0].shape)
    # plt.imshow(apply_mask(images[0]))
    # plt.colorbar()
    # plt.show(block=True)
