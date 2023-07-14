import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
from pathlib import Path
from optics.calc.interferogram import Interferogram, InterferogramOld
# from optics.utils.array import add_dims_on_right
# from optics.utils.display import complex2rgb, grid2extent, format_image_axes, complex_color_legend
# from optics.utils import ft
from optics.utils.ft.subpixel import Reference, roll, roll_ft
from projects.confocal_interference_contrast.polarization_memory_effect import log
from tqdm import tqdm
# from datetime import datetime
# import cv2
# from scipy.optimize import curve_fit

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
#todo try use a circle aperture instead if Wiener filters
noise_level = noise_factor * np.amax(np.abs(initial_interferogram_images))  # todo try to remove the amax
spatially_correct_using_wiener_filter = True #todo try False
log.info(f'Spatially correct using wiener filter: {spatially_correct_using_wiener_filter}')

if spatially_correct_using_wiener_filter:
    wiener_filters = np.conj(initial_interferogram_images) / (np.abs(initial_interferogram_images) ** 2 + noise_level ** 2)
else:
    wiener_filters = 1

paths = fd.askopenfilenames(initialdir=input_directories[0], title='Polarization scan files (----------):')
log.info('Loading data...')
stability_data = np.load(paths[0])

for _ in range(6):
    stability_image_pairs = stability_data['stability_image_pairs'][_*100:(_+1)*100]
    hv_field_pairs = []

    for count in tqdm(range(stability_image_pairs.shape[0]), desc='Calculating the horizontal and vertical polarized field...'):
        diagonal_interferograms = [InterferogramOld(img,
                                                    fringe_frequency=ref.fringe_frequency, fringe_amplitude=ref.fringe_amplitude,
                                                    ) for img, ref in zip(stability_image_pairs[count], reference_interferograms)]
        diagonal_interferogram_images = np.array([np.asarray(_) for _ in diagonal_interferograms])
        hv_field_pair = wiener_filters * diagonal_interferogram_images
        hv_field_pair[1] = roll(hv_field_pair[1], -inter_channel_shift) / inter_channel_factor
        hv_field_pairs.append(hv_field_pair)

    hv_field_pairs = np.asarray(hv_field_pairs)   # hv_field_pairs[:, 0] + pol_coefficients[_] * hv_field_pairs[:, 1]
    log.info(f'Saving hv field pair...')
    output_file_path = Path(f'E:/polarization_memory_effect/processed_data/stability_part_{_}'
                            ).resolve()
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file_path, hv_field_pairs=hv_field_pairs)
    log.info(f'Saved to {output_file_path}.')

exit()
#
#


normalize = lambda _: _ / np.sqrt(np.mean(_**2, axis=(-3, -2, -1), keepdims=True))

ref_field = []
h_comparison = []
v_comparison = []
for part in tqdm(range(5)):
    file_path = Path(f'E:/polarization_memory_effect/processed_data/stability_part_{part}.npz').resolve()
    hv_field_pairs = np.load(file_path)['hv_field_pairs']
    complex_field_comparison = []
    for _ in range(2):
        field = hv_field_pairs[:, _]
        if part == 0:
            ref_field.append(np.conj(normalize(field[0:1])))

        inp_b_vector = ref_field[_] * normalize(field)
        complex_field_comparison.append(np.mean(inp_b_vector, axis=(-2, -1)))
    h_comparison.append(complex_field_comparison[0])
    v_comparison.append(complex_field_comparison[1])
print(len(h_comparison))
print(h_comparison[0].shape)
h_comparison = np.array(np.asarray(h_comparison))
v_comparison = np.array(np.asarray(v_comparison))


fig, axs = plt.subplots(1, 2, figsize=(12, 8))

amp_h = np.abs(h_comparison)
amp_v = np.abs(v_comparison)

axs[0].plot(amp_h/amp_h[0])
axs[1].plot(amp_v/amp_v[0])
plt.show()
exit()



# fig2, axs2 = plt.subplots(1, 2, figsize=(12, 8))
# [axs2[_].imshow(complex2rgb(hv_field_pairs[0,_])) for _ in range(2)]


#


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

