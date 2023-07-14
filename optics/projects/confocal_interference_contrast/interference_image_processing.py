import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import isfile, join
import re
from typing import Union

from optics.utils.ft import Grid, ft
from projects.confocal_interference_contrast.complex_phase_image import Interferogram, get_registration
from optics.utils.ft.subpixel import Registration, register
from optics.utils.display import complex2rgb, grid2extent
from optics.utils.ft.subpixel import roll, roll_ft
from projects.confocal_interference_contrast.get_correlation import get_correlation


data_locations = {'experimental': r'C:\\Users\\Laurynas\\Desktop\\Python '
                                  r'apps\\Shared_Lab\\lab\\code\\python\\optics\\optics\\experimental\\',
                  'bead scans': r'C:\Users\dshamoon001\OneDrive - University of Dundee\LS PhC scans of beads\\',

                  'cheek cell scans': r'C:\\Users\\Laurynas\\OneDrive - University of '
                                      r'Dundee\\Work\\Data\\Scans\\LS PhC scans of cheek cells\\'}


def get_file(serial_nb: str, file_folder: str, file_format: str = 'npy'):
    path = file_folder
    files = [file for file in os.listdir(path) if isfile(join(path, file))]
    search = re.compile(fr'{serial_nb}')
    file_type = re.compile(fr'\.{file_format}')
    for file in files:
        target = (search.search(file) is not None) * (file_type.search(file) is not None)
        if target:
            target_file = file
            return np.array(np.load(path + target_file))
    raise FileNotFoundError


def reduce_roi_circularly(file):
    # file_shape = file.shape
    # center = np.array(file_shape[-2:], dtype=int) / 2
    # row_offset = 0
    # col_offset = 0

    grid = Grid(shape=file.shape[-2:])
    radial_grid = (grid[0] ** 2 + grid[1] ** 2) ** 0.5

    file[..., radial_grid > 100] = 0
    return file


def avg4image(file):
    image = np.zeros(file.shape[:2])
    for idx_row, _ in enumerate(file):
        for idx_col, snapshot in enumerate(_):
            image[idx_row, idx_col] = np.mean(snapshot)
    return image


def prod(iterable):
    """Like sum but with multiplication."""
    iterable = np.asarray(iterable).ravel()
    if np.size(iterable) == 1:
        return iterable[0]
    else:
        result = iterable[0]
    for item in iterable[1:]:
        result *= item
    return result


def complex_phase_operator(registration_shift, shape):
    """
    Calculating operators 2 convert intensity image to complex phase image.
    """

    image_grid = Grid(shape=shape)
    # Calculates phase operator using a single point Fourier transform
    # cp_operator = np.exp(np.array(1j * sum(
    #     [r * k for r, k in zip(image_grid, np.array(image_grid.k.step) * np.array(registration.shift))])))

    component_range = 1
    cp_operators = []
    for shift_x in np.arange(component_range) - int(component_range / 2):
        for shift_y in np.arange(component_range) - int(component_range / 2):
            current_shift = [registration_shift[0] + shift_x, registration_shift[1] + shift_y]
            cp_operator = np.exp(np.array(1j * sum(
                [r * k for r, k in zip(image_grid, np.array(image_grid.k.step) * np.array(current_shift))])))
            cp_operators.append(cp_operator)
    cp_operators = sum(cp_operators)

    return cp_operators


def reduce_pixel_resolution(array: np.ndarray, reduction_idx: Union[list, np.ndarray, int] = 2, out=None):
    ndim = np.ndim(array)
    if np.ndim(reduction_idx) != ndim:
        reduction_idx = [reduction_idx for _ in range(ndim)]

    # slcs = [slice(None)] * ndim
    # for idx in range(ndim):
    #     slcs[idx] = slice(reduction_idx[idx] - 1, None, reduction_idx[idx])
    # return array[tuple(slcs)]  # TODO: this output will return decreased camera sampling rather than resolution

    if out is None:
        out = array.copy()[tuple([slice(0, None, reduction_idx[dim]) for dim in range(ndim)])]
    for idx_row in range(out.shape[0]):
        for idx_col in range(out.shape[1]):
            out[idx_row, idx_col] = np.mean(array[int(idx_row * reduction_idx[0]):int((idx_row + 1) * reduction_idx[0]),
                                            int(idx_col * reduction_idx[1]):int((idx_col + 1) * reduction_idx[1])])
    return out


def simulate_dic(complex_image: np.ndarray, px_separation: int = 1) -> np.ndarray:
    # Simulate two-spot detection
    nb_cols = complex_image.shape[1]
    rng_b = np.clip(np.arange(nb_cols) + px_separation, 0, nb_cols - 1)
    a = complex_image
    b = complex_image[:, rng_b]

    # Measurements of different interferences
    def add_noise(signal, noise_level=0.0):
        return signal + np.random.randn(*signal.shape) * noise_level * np.amax(signal)

    interference_1 = add_noise(np.abs(a + b) ** 2)
    interference_i = add_noise(np.abs(a + 1j * b) ** 2)
    interference_m1 = add_noise(np.abs(a - b) ** 2)
    interference_mi = add_noise(np.abs(a - 1j * b) ** 2)
    # Interference term
    interference_term = interference_1 - 1j * interference_i - interference_m1 + 1j * interference_mi
    # Integrate to get absolute phase measurements again
    interference_term /= np.abs(interference_term)
    integral_result = np.cumprod(interference_term, axis=1) * np.abs(a)  # TODO use B as well
    # integral_result_right_to_left = np.cumprod(interference_term[:, ::-1], axis=1) * np.abs(a)[:, ::-1]  # TODO use B as well
    # integral_result[1::2] = integral_result_right_to_left[1::2, ::-1]
    return integral_result


if __name__ == "__main__":
    def use_one_px_instead_of_mean():
        # scan_file = get_file('0045FULL', data_locations['cheek cell scans'])
        # registration = get_registration()
        # complex_scan_image = np.zeros(scan_file.shape[:2], dtype=np.complex)
        # for idx_row in range(complex_scan_image.shape[0]):
        #     for idx_col in range(complex_scan_image.shape[1]):
        #         complex_image = np.array(Interferogram(scan_file[idx_row, idx_col].astype(np.complex) / 255, registration=registration))
        #         complex_scan_image[idx_row, idx_col] = complex_image[complex_image.shape[0] // 2,
        #                                                              complex_image.shape[1] // 2]
        #         print(f'{(int(idx_row + 1) * (idx_col + 1))} / {int(prod(complex_scan_image.shape))}')
        # np.save('complex_scan_image.npy', complex_scan_image)
        complex_scan_image = np.load('complex_scan_image.npy')
        plt.imshow(complex2rgb(complex_scan_image, 1))
        plt.show()
    use_one_px_instead_of_mean()

    def exact_dic():
        file = get_file('0026FULL', data_locations['bead scans'], file_format='npy')
        file_cp = get_file('0026CP', data_locations['bead scans'], file_format='npy')
        # converting the interference image into a complex phase image
        registration = Registration(shift=[42.0546875, -478.2734375],
                                    factor=-443398.7939103801 - 1437184.2432062984j)  # registration is picked for 0026

        separation_px = 1
        dic_image = np.zeros((file.shape[0], file.shape[1] - separation_px), dtype=np.complex64)
        interference_vals = np.zeros((4, *file.shape[:2]), dtype=np.complex64)
        phase_shifts = np.array([1, 1j, -1, -1j], dtype=np.complex64)
        interference_measurements = {}
        counter_switch = 1
        for idx_row in np.arange(file.shape[0]):
            for idx_col in np.arange(file.shape[1] - separation_px):
                # Calculating complex images from iterference patterns
                if idx_col != 0 and separation_px == 1:
                    interference_image1 = interference_image2  # executed in 2nd iteration at the earliest
                    cp_image1 = cp_image2
                else:
                    interference_image1 = np.array(file[idx_row, idx_col], dtype=np.complex64) / 255
                    cp_image1 = np.array(Interferogram(interference_image1, registration=registration))
                    if counter_switch:
                        norm_factor = np.linalg.norm(cp_image1)
                        counter_switch = 0
                    cp_image1 /= norm_factor

                interference_image2 = np.array(file[idx_row, idx_col + separation_px], dtype=np.complex64) / 255
                cp_image2 = np.array(Interferogram(interference_image2, registration=registration)) / norm_factor

                # Simulating the method with which we would measure the phase
                interference_term = 0
                for phase_shift in phase_shifts:
                    interference_measurements[str(phase_shift)] = np.abs(cp_image1 + cp_image2 * phase_shift) ** 2
                    interference_term += interference_measurements[str(phase_shift)] * phase_shift.conj()


                dic_image[idx_row, idx_col] = np.mean(interference_term)

                complex_difference = (np.abs(cp_image1) - np.abs(cp_image2)) * \
                                      np.exp(1j * (np.angle(cp_image1) - np.angle(cp_image1)))
                dic_image[idx_row, idx_col] = np.mean(complex_difference)
                # interference_vals[:, idx_row, idx_col] =  # TODO: save these values
            # break
            print(f'Progress: {(idx_row + 1) / file.shape[0] * 100:.2f}%')
        np.save('DIC_sim_img.npy', dic_image)

        dic_image = np.load('DIC_sim_img.npy')

        def add_noise(signal, noise_level=0.0):
            return signal + np.random.randn(*signal.shape) * noise_level * np.amax(signal)
        dic_image = add_noise(dic_image, 0.05)
        phase_integral = np.ones(dic_image.shape[0], dtype=np.complex64)
        phase_integral_smooth = phase_integral.copy()
        phase_image = dic_image.copy()
        phase_image_smooth = phase_image.copy()
        smooth_dic_angle = np.angle(dic_image.copy())
        dic_angle = np.angle(dic_image.copy())
        for idx in np.arange(dic_image.shape[1]):
            if idx > 1 and idx < dic_image.shape[1] - 2:
                # smooth_dic_angle[:, idx] = ((dic_angle[:, idx - 1] + dic_angle[:, idx + 1]) / 2
                #                             + dic_angle[:, idx]) / 2
                first_degree_mean = (dic_angle[:, idx - 1] + dic_angle[:, idx + 1]) / 2
                second_degree_mean = (dic_angle[:, idx - 2] + dic_angle[:, idx + 2]) / 2
                cond = np.abs(first_degree_mean - dic_angle[:, idx]) > np.abs(first_degree_mean - second_degree_mean)

                smooth_dic_angle[:, idx][cond] = dic_angle[:, idx][cond]
                smooth_dic_angle[:, idx][np.logical_not(cond)] = second_degree_mean[np.logical_not(cond)]

            phase_integral_smooth *= np.exp(1j * smooth_dic_angle[:, idx])
            phase_image_smooth[:, idx] = phase_integral_smooth
            phase_integral *= np.exp(1j * np.angle(dic_image[:, idx]))
            phase_image[:, idx] = phase_integral

        # integral_result = np.cumprod(dic_image, axis=1)
        # for idx_col in range(1, dic_image.shape[1]):
        #     dic_image[:, idx_col - 1] *= dic_image[:, idx_col]

        fig, axs = plt.subplots(2)

        axs[0].imshow(complex2rgb(phase_image, 1))
        axs[0].set_title('Regular integration')
        axs[1].imshow(complex2rgb(phase_image_smooth, 1))
        axs[1].set_title('Smoothed out integration')
        plt.show()
    # exact_dic()


    def init_dic_simulation():
        # Confocal DIC simulation from phase contrast
        # complex_img = get_file('0013CP', data_locations['experimental'] + '0013\\')
        complex_img = get_file('0024CP', data_locations['bead scans'])
        noise_level = 0.0
        complex_img += (np.random.randn(*complex_img.shape) + 1j * np.random.randn(*complex_img.shape)) * noise_level * np.amax(complex_img) / np.sqrt(2)
        phase_img = np.exp(1j * np.angle(complex_img))  # converts the complex into phase only (sets intensity to 1 everywhere)
        dic_img = simulate_dic(complex_img, px_separation=1)
        fig, axs = plt.subplots(1, 3)

        img = axs[0].imshow(complex2rgb(complex_img[0], 1))
        axs[0].set_title('Complex phase image')

        axs[1].imshow(complex2rgb(dic_img[0], 1))
        axs[1].set_title('DIC phase')
        axs[2].imshow(np.abs(dic_img[0]))
        axs[2].set_title('DIC intensity')
        plt.show(block=True)
    init_dic_simulation()


    def init_cp_analysis():
        file = get_file('0040FULL', data_locations['cheek cell scans'], file_format='npy')
        img_shape = file.shape[-2:]

        registration = get_registration()
        # registration.shift = [79.140625, -395.4453125]
        nb_reg_shifts = 5
        reg_shift_step = 1
        reg_shift_shifts = (np.arange(nb_reg_shifts) - np.arange(nb_reg_shifts) // 2) * reg_shift_step

        cp_operators = []
        for idx_row, row_shift in enumerate(reg_shift_shifts):
            cp_operators.append([])
            for idx_col, col_shift in enumerate(reg_shift_shifts):
                registration_shift = registration.shift
                registration_shift[0] += row_shift
                registration_shift[1] += col_shift
                cp_operators[idx_row].append(complex_phase_operator(registration_shift, img_shape))
        cp_operators = np.asarray(cp_operators)
        # cp_operator = complex_phase_operator(registration.shift, file.shape[-2:])

        scan_img = np.zeros(shape=(*cp_operators.shape[:2], *file.shape[:2]), dtype=np.complex)
        for idx_row, _ in enumerate(file):
            for idx_col, img in enumerate(_):
                # scan_img[idx_row, idx_col] = np.mean(np.array(Interferogram(np.asarray(img, dtype=np.complex) / 255, registration=registration)))
                for idx_row_shift in range(nb_reg_shifts):
                    for idx_col_shift in range(nb_reg_shifts):
                        scan_img[idx_row_shift, idx_col_shift, idx_row, idx_col] \
                            = np.mean(img * cp_operators[idx_row_shift, idx_col_shift])
                # scan_img[idx_row, idx_col] = np.mean(img * cp_operator)

        fig, axs = plt.subplots(*cp_operators.shape[:2])
        for idx_row in range(nb_reg_shifts):
            for idx_col in range(nb_reg_shifts):
                axs[idx_row, idx_col].imshow(complex2rgb(scan_img[idx_row, idx_col], 1))
                axs[idx_row, idx_col].set_title(f'$\Delta$ Registration shift: '
                                                f'{reg_shift_shifts[idx_row]}, {reg_shift_shifts[idx_col]}')
        # plt.imshow(complex2rgb(scan_img, 10))
        plt.show(block=True)
    # init_cp_analysis()


    def init_spectrum_analysis():
        file = get_file('0026FULL', data_locations['bead scans'], file_format='npy')
        image = np.asarray(file[1, 10, ...], dtype=np.complex64) / 255
        registration = Registration(shift=[42.0546875, -478.2734375],
                                    factor=-443398.7939103801 - 1437184.2432062984j)
        complex_img = np.array(Interferogram(image, registration=registration))
        del file

        grid = Grid(shape=image.shape)
        # plt.imshow(complex2rgb(ft.fftshift(ft.fft2(image)), 1000), extent=grid2extent(grid))
        plt.imshow(complex2rgb(ft.fftshift(ft.fft2(ft.ifftshift(complex_img))), 1))
        plt.show()
    # init_spectrum_analysis()


    def init_tm_analysis():
        file = get_file('0026FULL', data_locations['bead scans'], file_format='npy')

        # converting the interference image into a complex phase image
        registration = Registration(shift=[42.0546875, -478.2734375],
                                    factor=-443398.7939103801 - 1437184.2432062984j)  # registration is picked for 0026

        # tm calculation starts here
        tm_shape = [np.prod(file.shape[:2]), np.prod(file.shape[-2:])]
        tm_shape[0] = file.shape[1]  # TODO: adjustment to only measure the first row
        # adjust the file shape to make the tm square (selects a roi) comment out to leave it be
        image_centre = np.array(file.shape[-2:], dtype=int) // 2
        roi_half_extent = int(np.sqrt(tm_shape[0])) // 2
        field_slc = tuple([slice(image_centre[0] - roi_half_extent, image_centre[0] + roi_half_extent),
                           slice(image_centre[1] - roi_half_extent, image_centre[1] + roi_half_extent)])
        field_slc = tuple([slice(image_centre[0], image_centre[0] + 1),  # TODO adjustment to only measure the first line
                           slice(image_centre[1] - tm_shape[0] // 2, image_centre[1] + tm_shape[0] // 2)])


        tm_shape[1] = tm_shape[0]

        tm = np.zeros(tm_shape, dtype=np.complex64)
        memory_effect = np.zeros(tm_shape[1], dtype=np.complex64)
        for idx_row, row in enumerate(file):
            for idx_col, interference_image in enumerate(row):
                tm_col_idx = (idx_row + 1) * (idx_col + 1) - 1
                # noinspection PyTypeChecker
                cp_image = np.array(Interferogram(interference_image.astype(dtype=np.complex64) / 255,
                                                  registration=registration))
                # field_slc = tuple([slice(image_centre[0] // 2, image_centre[0] // 2 + 1),  # TODO adjustment for resolution reduction
                #                                slice(image_centre[1] // 2 - tm_shape[0] // 2, image_centre[1] // 2 + tm_shape[0] // 2)])
                # cp_image = reduce_pixel_resolution(cp_image)
                cp_image = cp_image[field_slc]
                cp_image_ft = np.fft.ifftshift(np.fft.fft2(cp_image))
                tm[:, tm_col_idx] = roll(cp_image_ft.ravel(), (-tm_shape[1] // 2 + tm_col_idx) * 1)
                # tm[:, tm_col_idx] = cp_image_ft.ravel()

                # analysis of optical memory effect in terms of correlation
                if tm_col_idx == 0:
                    mem_effect_reference = cp_image
                memory_effect[idx_col] = get_correlation([mem_effect_reference, cp_image])[1]

                print(f'TM column {tm_col_idx + 1}/{tm_shape[1]}')
            break  # TODO adjustment to only measure the first line
        np.save('tm.npy', tm)
        fig, axs = plt.subplots(2)
        axs[0].imshow(complex2rgb(tm, 2))
        axs[1].plot(np.abs(memory_effect))
        plt.show(block=True)
    # init_tm_analysis()
