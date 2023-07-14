import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
from pathlib import Path
from optics.calc.interferogram import Interferogram, InterferogramOld
# from optics.utils.array import add_dims_on_right
from optics.utils.display import complex2rgb, grid2extent, format_image_axes, complex_color_legend
# from optics.utils import ft
from optics.utils.ft.subpixel import Reference, roll, roll_ft
from projects.confocal_interference_contrast.polarization_memory_effect import log
from tqdm import tqdm
from datetime import datetime
# import cv2
# from scipy.optimize import curve_fit


def display(data1,data2=12):
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs[0].imshow(complex2rgb(data1))
    if data2 == 12:
        axs[0].imshow(complex2rgb(data1))
    else:
        axs[1].imshow(complex2rgb(data2))
    plt.show()


#todo list: fit the experimental data***, compare two scans with dif pol, test for different noise values ...
#todo linst tom: check the difference between the horizontal filed in dif positions
input_directories = [r'E:\polarization_memory_effect\experimental_data']

reference_path = fd.askopenfilenames(initialdir=input_directories[0], title='Reference file:')
log.info('Loading reference data...')
reference_data = np.load(reference_path[0])
scans_reference = reference_data['scan_images'][0]  # the [0] is because of the shape: (1, 10, 2, 1024, 1280) -> (10, 2, 1024, 1280)
positions_reference = reference_data['positions']
position = 0 + int(scans_reference.shape[0]/2)  # central position of the reference scan
log.info('Creating reference interferograms...')
reference_interferograms = [InterferogramOld(_) for _ in scans_reference[position]]
grid = reference_interferograms[0].grid
log.info('Detecting image translation between the two output polarization channels based on the amplitude only.')
registration_of_second = Reference(np.abs(reference_interferograms[0])).register(np.abs(reference_interferograms[1])) #registration of the images
inter_channel_shift = registration_of_second.shift
inter_channel_factor = registration_of_second.factor.real # using real values because our measurement are intensity images
log.info(f'Found a shift of {inter_channel_shift} px between the two channels, and the second is brighter by a factor of {inter_channel_factor:0.3f}.')
log.info('Calculating filter for per-pixel phase and amplitude correction filters.')
initial_interferogram_images = np.array([np.asarray(_) for _ in reference_interferograms])
log.info(f'Max contrast of initial interferograms: {np.amax(np.abs(initial_interferogram_images), axis=(-2, -1))}')
noise_factor = 0.30  #TODO: Check if this is a good value
#todo try use a circle aperture instead if wiener filters
noise_level = noise_factor * np.amax(np.abs(initial_interferogram_images)) # todo try to remove the amax
spatially_correct_using_wiener_filter = True #todo try False
log.info(f'Spatially correct using wiener filter: {spatially_correct_using_wiener_filter}')

if spatially_correct_using_wiener_filter:
    wiener_filters = np.conj(initial_interferogram_images) / (np.abs(initial_interferogram_images) ** 2 + noise_level ** 2)
else:
    wiener_filters = 1


paths = fd.askopenfilenames(initialdir=input_directories[0], title= 'Polarization scan files (horizontal, vertical, diagonal, anti, right, left):')
log.info('Loading polarization scan data...')
aux_sequence = np.zeros(6, dtype='int')
index = 0

for _ in ['horizontal', 'vertical', 'diagonal','anti_diagonal', 'right','left']:
    out = 'n'
    count = 0
    while out == 'n':
        if paths[count].find(_) >= 0:
            aux_sequence[index] = int(count)
            out = 'y'
            if _ == 'diagonal' and paths[count].find('anti') >= 0:
                out = 'n'
        count += 1
    index += 1

scans_polarizations = []
horizontal_pol_data = np.load(paths[0])
scans_polarizations.append(horizontal_pol_data['scan_images'][0])
positions = horizontal_pol_data['positions']
[scans_polarizations.append(np.load(paths[aux_sequence[_+1]])['scan_images'][0]) for _ in range(5)]


def hv_field(scans, reference_interferograms, inter_channel_shift, inter_channel_factor):
    #Calculates the horizal and vertical polarized field for each pair of images along the scan
    hv_field_pairs = []

    for count in tqdm(range(scans.shape[0]), desc='Calculating the horizontal and vertical polarized field...'):
        diagonal_interferograms = [InterferogramOld(img,
                                                    fringe_frequency=ref.fringe_frequency, fringe_amplitude=ref.fringe_amplitude,
                                                    ) for img, ref in zip(scans[count], reference_interferograms)]
        diagonal_interferogram_images = np.array([np.asarray(_) for _ in diagonal_interferograms])
        hv_field_pair = wiener_filters * diagonal_interferogram_images
        hv_field_pair[1] = roll(hv_field_pair[1], -inter_channel_shift) / inter_channel_factor
        hv_field_pairs.append(hv_field_pair)

    hv_field_pairs = np.asarray(hv_field_pairs)
    return hv_field_pairs


pol_coefficients = np.asarray([0, np.infty, 1, -1, 1j, -1j])  # todo check if is necessary the normalized coefficients
pol_names = ['Horizontal', 'Vertical', 'Diagonal', 'Anti diagonal', 'Right circular', 'Left circular']
save_control = input('Save the reconstructed fields of the scans ? (y/n)')
scan_fields = []
for _ in range(6):
    log.info(f'Reconstructing the {pol_names[_]} polarization scan fields...')
    hv_field_pairs = hv_field(scans_polarizations[_], reference_interferograms, inter_channel_shift, inter_channel_factor)
    if pol_coefficients[_] != np.infty:
        scan_fields.append(hv_field_pairs[:, 0] + pol_coefficients[_] * hv_field_pairs[:, 1])  # hv_field_pairs: stage position, output polarization, vertical pixel, horizontal pixel
    else:
        scan_fields.append(hv_field_pairs[:, 1])
    if save_control == 'y':
        log.info(f'Saving the {pol_names[_]} polarization scan fields...')
        output_file_path = Path(f'E:/polarization_memory_effect/processed_data/scan_field_{pol_names[_]}' # ../../confocal_interference_contrast/polarization_memory_effect/results
                                + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")).resolve()
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_file_path, scan_fields=scan_fields[_], polarization=pol_names[_],
                 positions=positions,
                 spatially_correct_using_wiener_filter=spatially_correct_using_wiener_filter,
                 noise_factor=noise_factor,
                 )
        log.info(f'Saved to {output_file_path}.')


def complex_field_comparison(scan_field_pol0, scan_field_pol1):
    """
    Compares the scattered/transmitted field at the same position in two independent scans by calculating the inner
    product of their respective complex fields. This can be used to evaluate the polarization or shift memory effect.

    :param scan_field_pol0: The complex images of scan 0. Array with axes: stage position, vertical pixel, horizontal pixel
    :param scan_field_pol1: The complex images of scan 1. Array with axes: stage position, vertical pixel, horizontal pixel
    :return: A 1D array with the inner products between different polarizations at the same position in their respective scans.
    """
    normalize = lambda _: _ / np.sqrt(np.mean(_**2, axis=(-3, -2, -1), keepdims=True))
    inp_b_vector = np.conj(normalize(scan_field_pol0)) * normalize(scan_field_pol1)  # TODO: check whether multiplication is the right operation (or need to do dot with @)
    return np.mean(inp_b_vector, axis=(-2, -1))  # the inner product as complex scalars, one for each position


def compare_complex_field_with_central_field(scan_fields_for_1_input_polarization):
    """
    Compares the scattered/transmitted field at different positions in the scan by calculating the inner product of the
    complex fields. It takes a scalar products of a complex (vectorized) images for each stage position, after L2
    normalization, with the normalized complex image at the center.

    :param scan_fields_for_1_input_polarization: complex array with 3 axes: stage position, vertical pixel, horizontal pixel
    :return: Inner products for each stage position.
    """
    center = scan_fields_for_1_input_polarization.shape[0] // 2
    return complex_field_comparison(scan_fields_for_1_input_polarization[center:center+1], scan_fields_for_1_input_polarization)


complex_shift_pols= []
for _ in range(6):
    log.info(f'Calculating shift memory effect of {pol_names[_]} polarization')
    complex_shift_pols.append(compare_complex_field_with_central_field(scan_fields[_]))

complex_pol_memory= []

for _ in range(5):
    log.info(f'Calculating polarization memory effect of {pol_names[_+1]} polarization in relation to {pol_names[0]}')
    complex_pol_memory.append(complex_field_comparison(scan_fields[0], scan_fields[_+1]))

log.info(f'Saving the complex_shift and pol memory effect...')
output_file_path = Path(f'E:/polarization_memory_effect/processed_data/complex_shift_pol_memory_' # ../../confocal_interference_contrast/polarization_memory_effect/results
                        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")).resolve()
output_file_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(output_file_path, complex_shift_pols=complex_shift_pols, complex_pol_memory = complex_pol_memory,
         polarization=pol_names, positions=positions,
         spatially_correct_using_wiener_filter=spatially_correct_using_wiener_filter,
         noise_factor=noise_factor, reference_path = reference_path[0]
         )
log.info(f'Saved to {output_file_path}.')


#
# colors = 'rkgbc'
# control = ''
# while control != 'exit':
#     control = input('1- Shift memory effect; 2- Polarization comparison; exit - Exit')
#     if control == '1':
#         complex_shift_pols = []
#         for _ in range(4):
#             complex_shift_pols.append(compare_complex_field_with_central_field(scan_fields[_]))
#         fig = plt.figure()



exit()

    # for _ in range(len(complex_shift_pols)):
    #     plt.plot(positions[:,0], abs_shift_pols_fit[count]/np.amax(abs_shift_pols_fit[count])), 'o'+ colors[count], label=_ + 'Fit')
    # plt.legend()
    # plt.show()


# utilities cuts

reference_path = fd.askopenfilenames(initialdir=input_directories[0], title='Reference file:')

polarization_scan_fields = []
pol_names = []
complex_shift_pols = []
scan_fields = []
for _ in reference_path:
    log.info(f'Loading data set {_[-40:-1]}')
    data_loaded = np.load(_)
    complex_shift_pols.append(compare_complex_field_with_central_field(data_loaded['scan_fields']))
    scan_fields.append(data_loaded['scan_fields'])
    pol_names.append(data_loaded['polarization'])
positions = data_loaded['positions']

complex_pol_shift = []
[complex_pol_shift.append(complex_field_comparison(scan_fields[0], scan_fields[_ + 1])) for _ in range(len(reference_path) - 1)]

# def Gauss(x, A, B):
#     y = A*np.exp(-1*B*x**2)
#     return y
#
# x_vector = np.linspace(positions[0, 0], positions[-1, 0], 1000)
# abs_shift_pols_fit = []
# for count in range(len(complex_shift_pols)):
#     parameters, covariance = curve_fit(Gauss, positions[:,0], complex_shift_pols[int(count)][0]/np.amax(complex_shift_pols[int(count)][0]))
#     abs_shift_pols_fit.append(Gauss(positions[:,0], parameters[0],parameters[1]))

fig = plt.figure()
colors = ['r', 'k', 'g', 'b', 'c']
#for count, _ in enumerate(pol_names):
for _ in range(len(complex_pol_shift)):
    name = 'test'#pol_names[_+1]
    plt.plot(positions[:,0], complex_pol_shift[_][0], 'o'+colors[_], label=name)
    #plt.plot(positions[:,0], abs_shift_pols_fit[count]/np.amax(abs_shift_pols_fit[count]))#, colors[count], label=_ + 'Fit')
plt.legend()
plt.show()



# import scipy.io
# output_file_path = Path(f'../../confocal_interference_contrast/polarization_memory_effect/results/scan_field_testt'
#                         + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.mat")).resolve()
# output_file_path.parent.mkdir(parents=True, exist_ok=True)
#
#
# scipy.io.savemat(output_file_path, {'datay': complex_shift_pols[0][0]/np.amax(complex_shift_pols[0][0]), 'datax': positions[:,0]})
#
# print('save')



hv_field_pairs = hv_field(scans_diagonal, reference_interferograms, inter_channel_shift, inter_channel_factor)

output_file_path = Path(f'../../confocal_interference_contrast/polarization_memory_effect/results/scan_field_test_'
                        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.npz")).resolve()
output_file_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(output_file_path,hv_field_pairs=hv_field_pairs)
log.info(f'Saved to {output_file_path}.')


#display
fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharex='all', sharey='all')
for _, pol in enumerate(['Reference transmitted','Reference reflected']):
    axs[_].imshow(complex2rgb(hv_field_pairs[0,_]))
    axs[_].set_title(pol)
plt.tight_layout()
plt.show()

exit()


# For complex fields:
normalize = lambda _: _ / np.sqrt(np.sum(_**2, axis=(-3, -2, -1), keepdims=True))
inp_b = np.sum(np.conj(normalize(scans)) * normalize(scans[0, 49:50]), axis=(-3, -2, -1))
results_b = np.abs(inp_b)

# for a in range(2):
#     for _ in range(11):
#         results_b[a, _] = np.mean(abs(scans[a+1,10-_,0]/np.amax(scans[a+1,10-_,0]) - scans[0,_,0]/np.amax(scans[0,_,0])))


#
# #display
# #titles = ['5626', '5706', '5751', 'Monday']
colors = ['r', 'k', 'g', 'b', 'c']
fig = plt.figure()
# count = 0
# for _ in range(scans.shape[0]):
#     for b in range(2):
#         plt.plot(results[_,b])
for _ in range(2):
    plt.plot(results_b[_], 'o', label=f'scan {_}')
#     plt.plot(x, results[_, 0],  '^' + colors[count], label='Linear ' + titles[_])
#     plt.plot(x, results[_, 2], 'o' + colors[count], label='Circular ' + titles[_])
#     plt.plot(X, curves_fit[_, 0],  '--' + colors[count], label='Fit Linear ' + titles[_])
#     plt.plot(X, curves_fit[_, 2], ':' + colors[count], label='Fit Circular ' + titles[_])
#     count += 1
#plt.imshow(test/test)
plt.legend()
plt.show()
# plt.legend()
# plt.show()
