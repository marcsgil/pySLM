import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from optics.calc.interferogram import Interferogram
from optics.utils.display import complex2rgb, grid2extent
from optics.utils import ft
from optics.utils.ft import Grid
from optics.utils.ft.subpixel import Registration


log = logging.getLogger(__file__)


def get_registration(registration_name: str = "registration", registration_path: str = "settings.json") -> Registration:
    """
    Loads saved registration from JSON file.

    :param: Name of the saved registration
    :return: Loaded registration
    """
    with open(
            registration_path,
            'r') as file:
        calibration = json.load(file)[registration_name]
    shift = calibration["shift"]
    factor = calibration["factor"]["real"] + calibration["factor"]["imag"] * 1j
    return Registration(shift=shift, factor=factor)


def save_registration(registration: Registration, registration_name: str = "registration", registration_path: str = "settings.json"):
    """
    Saves the given registration into a JSON file.
    :param: Name under which the registration will be save in a JSON file.
    """
    with open(
            registration_path,
            'r+') as file:
        settings = json.load(file)
        registration_settings = {}
        registration_settings["shift"] = list(registration.shift)
        registration_settings["factor"] = {"real": np.real(registration.factor), "imag": np.imag(registration.factor)}
        settings[registration_name] = registration_settings
        file.seek(0)
        json.dump(settings, file, indent=4)
        file.truncate()


def registration_calibration(interference_image: np.ndarray, registration_name: str = "registration",
                             registration_path: str = "settings.json",
                             registration_guess: Registration = None) -> Registration:
    """
    Gets interference registration from the given image and iteratively corrects it. Then proceeds to save it for later
    use, in addition to returning it.

    :param interference_image: Interference image, which will be evaluated for registration.
    :param registration_name: Name under which the registration will be save in a JSON file.
    :param registration_guess: Registration guess to better triangulate registration
    :return: Interference image registration
    """
    registration = registration_guess
    for _ in range(5):
        interferogram = Interferogram(interference_image, approximate_registration=registration)
        registration = interferogram.registration
    save_registration(registration, registration_name, registration_path=registration_path)
    return registration


def remove_aberration(image: np.ndarray, calibration_name: str = "aberrations calibration"):
    def defocus(rho):
        return 3 ** 0.5 * (2 * rho ** 2 - 1)

    def spherical(rho):
        return 5 ** 0.5 * (6 * rho ** 4 - 6 * rho ** 2 + 1)

    with open(
            'C:\\Users\\Laurynas\\Desktop\\Python apps\\Shared_Lab\\lab\\code\\python\\optics\\projects\\confocal_interference_contrast\\settings.json',
            'r') as file:
        calibration = json.load(file)[calibration_name]
    centre = calibration["centre pixel coordinates"]
    aberration_coefficients = calibration["aberrations calibration"]

    grid = Grid(shape=image.shape, center=centre)
    radial_grid = (grid[0] ** 2 + grid[1] ** 2) ** 0.5


if __name__ == "__main__":
    simulate = True  # Real testing requires the data from Laurynas

    pixel_pitch = 5.3e-6
    approximate_period = 3.25
    if not simulate:
        interference_image = Image.open(
            r'C:\Users\Laurynas\OneDrive - University of Dundee\Work\Pictures\Phase Contrast Era\michealson setup - 2 telescopes\polarisation play\No_sample_0_degrees_pol_change2020-10-28_16-03-28.807.png')
        reference_image = np.array(interference_image, dtype=float)[:, :, 0] / 256.0

        interference_image = Image.open(
            r'C:\Users\Laurynas\OneDrive - University of Dundee\Work\Pictures\Phase Contrast Era\michealson setup - 2 telescopes\polarisation play\glass-marker_sample_0_degrees_pol_change2020-10-28_16-05-47.520.png')
        test_image = np.array(interference_image, dtype=float)[:, :, 0] / 256.0

    else:
        actual_period_ref = approximate_period * pixel_pitch * 1.10
        actual_period_test = actual_period_ref * 1.10
        test_grid = Grid(shape=(234, 345), step=pixel_pitch)
        r = np.sqrt(test_grid[0] ** 2 + test_grid[1] ** 2)
        interference_field = (np.exp(2j * np.pi / (actual_period_ref * np.sqrt(2)) * (test_grid[0] + test_grid[1])) +
                              1.0) / 2
        reference_image = np.abs(interference_field) ** 2
        interference_field = (np.exp(2j * np.pi / (actual_period_test * np.sqrt(2)) * (test_grid[0] + test_grid[1])) +
                              np.exp(5j * np.arctan2(test_grid[1], test_grid[0]) * (r < np.amax(r) / 4))) / 2 \
                            * np.exp(-0.5 * (r / np.amax(r)) ** 2)
        test_image = np.abs(interference_field) ** 2

    grid = Grid(shape=reference_image.shape, step=pixel_pitch)
    comp_img_ref = Interferogram(raw_interference_image=reference_image)
    comp_img_test = Interferogram(raw_interference_image=test_image,
                                  approximate_registration=comp_img_ref.registration)

    comp_img_ref = np.array(comp_img_ref)
    comp_img_test = np.array(comp_img_test)

    complex_image = np.exp(1j * np.angle(comp_img_test)) / np.exp(1j * np.angle(comp_img_ref))

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(reference_image, extent=grid2extent(grid) / 1e-6)
    axs[0, 0].set(xlabel='x [$\mu m$]', ylabel='y [$\mu m$]', title='recorded reference')
    axs[1, 0].imshow(complex2rgb(ft.fftshift(ft.fft2(comp_img_ref)), 10), extent=grid2extent(grid.f) * 1e-3)
    axs[1, 0].set(xlabel='$f_x$ [cycles/mm]', ylabel='$f_y$ [cycles/mm]', title='spectrum reference')
    axs[0, 1].imshow(test_image, extent=grid2extent(grid) / 1e-6)
    axs[0, 1].set(xlabel='x [$\mu m$]', ylabel='y [$\mu m$]', title='recorded test')
    axs[1, 1].imshow(complex2rgb(ft.fftshift(ft.fft2(comp_img_test)), 10), extent=grid2extent(grid.f) * 1e-3)
    axs[1, 1].set(xlabel='$f_x$ [cycles/mm]', ylabel='$f_y$ [cycles/mm]', title='spectrum test')
    axs[0, 2].imshow(complex2rgb(comp_img_test, 1), extent=grid2extent(grid) / 1e-6)
    axs[0, 2].set(xlabel='x [$\mu m$]', ylabel='y [$\mu m]$', title='complex')
    axs[1, 2].imshow(complex2rgb(np.exp(1j * np.angle(complex_image)), 1), extent=grid2extent(grid) / 1e-6)
    axs[1, 2].set(xlabel='x [$\mu m$]', ylabel='y [$\mu m]$', title='phase only')

    plt.show()
