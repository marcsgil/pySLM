import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import re
from os.path import isfile, join
from typing import Union

from optics.utils.ft import Grid, ft, Registration
from optics.utils.display import complex2rgb, grid2extent
from projects.confocal_interference_contrast.complex_phase_image import Interferogram

data_locations = {'experimental': r'C:\\Users\\Laurynas\\Desktop\\Python '
                                  r'apps\\Shared_Lab\\lab\\code\\python\\optics\\optics\\experimental\\',
                  'bead scans': r'C:\\Users\\Laurynas\\OneDrive - University of '
                                r'Dundee\\Work\\Data\\Scans\\Tomographic scans of beads\\'}


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


file = get_file('0026FULL', data_locations['bead scans'])


def complex_phase_operator(registration_shift, shape):
    """
    Calculating operators 2 convert intensity image to complex phase image.
    """
    image_grid = Grid(shape=shape)
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


def upscale_vertical_array_size(array: np.ndarray, out: np.ndarray):
    proportions = out.shape[0] // array.shape[0]
    padding = [int((out.shape[idx] - array.shape[idx] * proportions) / 2) for idx in range(len(array.shape))]

    for idx_row in range(1, array.shape[0]):
        for idx_col in range(1, array.shape[1]):
            out[int((idx_row - 1) * proportions) + padding[0]:int(idx_row * proportions),
            int((idx_col - 1) * proportions) + padding[1]:int(idx_col * proportions)] \
                = array[idx_row, idx_col]
    return out


registration = Registration(shift=[42.0546875, -478.2734375], factor=-443398.7939103801 - 1437184.2432062984j)
cp_operator = complex_phase_operator(registration.shift, file.shape[-2:])

# def normalised_image(img, reference_img, transform=None):  # TODO past attempt to do one point Fourier for complex phase
#     img = np.asarray(img, dtype=np.complex) * transform / 255
#     reference_img = np.asarray(reference_img, dtype=np.complex) * transform / 255
#     return np.asarray(complex2rgb(subtract_phase_and_amplitude(img, reference_img), 3) * 255, dtype=np.uint8)

res_reduction = 4
shape = np.array(file.shape[-2:]) // res_reduction
vid_shape = [shape[0], shape[1] * 2][::-1]
video = cv2.VideoWriter('video_test_decreased_res.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, tuple(vid_shape))


def subtract_phase_and_amplitude(A, B):
    return np.abs(np.abs(A) - np.abs(B)) * np.exp(1j * (np.angle(A) - np.angle(B)))


cp_image = lambda image: np.array(Interferogram(np.array(image, dtype=np.complex) / 255, registration=registration))
cnv_rgb = lambda image: np.asarray(complex2rgb(image, 2) * 255, dtype=np.uint8)
ref_img_0 = cp_image(file[0, 0, ...])  # First reference img (first img in the set)

# fig, ax = plt.subplots() # TODO debugging
# im = ax.imshow(np.zeros(shape), extent=grid2extent(Grid(shape=shape)))

for idx_row, row_imgs in enumerate(file):
    # To display it in the scanning order every second scanning row must be flipped in order
    boustrophedon_flipping = int((idx_row % 2) * (-2) + 1)
    row_imgs = row_imgs[::boustrophedon_flipping, ...]

    ref_img_1 = cp_image(
        file[idx_row, int(row_imgs.shape[0] / 2), ...])  # Second reference img (the center of current row)
    for idx_col, img in enumerate(row_imgs):
        cp_img_current = cp_image(img)
        disp_img_0 = reduce_pixel_resolution(subtract_phase_and_amplitude(cp_img_current, ref_img_0), res_reduction)
        disp_img_1 = reduce_pixel_resolution(subtract_phase_and_amplitude(cp_img_current, ref_img_1), res_reduction)

        vid_frame = cnv_rgb(np.concatenate((disp_img_0, disp_img_1), axis=1))
        video.write(cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB))

        # if idx_col % 2 == 0:  # TODO: debugging
        #     test_A = cp_image(img)
        #
        #     plt.pause(0.001)
        #     im.set_data(complex2rgb(subtract_phase_and_amplitude(test_A, ref_img_0), 5))
        #     plt.show(block=False)

    break

video.release()
