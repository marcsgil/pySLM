import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import isfile, join
import re

from optics.utils.ft.subpixel import register
from optics.utils.ft import ft
from optics.utils.display import complex2rgb
from projects.confocal_interference_contrast.complex_phase_image import Interferogram, registration_calibration
from projects.confocal_interference_contrast import log


def get_file(serial_nb: str):
    path = r'C:\\Users\\Laurynas\\OneDrive - University of Dundee\\Work\\Data\\Scans\\Interference_scans\\'
    files = [file for file in os.listdir(path) if isfile(join(path, file))]
    search = re.compile(fr'{serial_nb}')
    file_type = re.compile(r'\.npy')
    for file in files:
        target = (search.search(file) is not None) * (file_type.search(file) is not None)
        if target:
            target_file = file
            return np.array(np.load(path+target_file)[0])
    raise FileNotFoundError


images = []
for idx in range(3):
    images.append(get_file(f'004.{idx}'))
images = np.sum(images, axis=0) / 3
images_repeated = get_file('004.3')

reference_image_without_sample = get_file('OO2')[0]
# images = [reference_image_without_sample] * len(images)  # TODO can be used in debugging
# reference_registration = get_registration()  # Loads registration from JSON
reference_registration = registration_calibration(reference_image_without_sample)  # calculates registration
nb = 20  # specify the number of images that are to be compared
first_pic = 60
scan_positions = np.arange(0, nb) * 0.25
reference_interferogram = Interferogram(images_repeated[first_pic + int(nb / 2)], registration=reference_registration)
reference_image = np.array(reference_interferogram)
# reference_registration.shift[1] *= 1.01

errors = []  # first dimension is distance, second is (unregistered, shift, tilt, shift-tilt)
for idx, interference_image in enumerate(images[first_pic:nb+first_pic]):
    log.info(f'Converting interference image to complex image {idx + 1}/{nb}...')
    complex_image = np.array(Interferogram(interference_image, registration=reference_registration))

    log.info(f'Analysing complex image {idx + 1}/{nb}...')
    registered_image_shifted = register(complex_image, reference_image, precision=0.1).image
    registered_image_tilt = register(ft.ifft2(complex_image), ft.ifft2(reference_image), precision=0.1).image_ft
    registered_image_shifted_tilt = register(ft.ifft2(registered_image_shifted), ft.ifft2(reference_image), precision=0.1).image_ft

    images_to_compare = (complex_image, registered_image_shifted, registered_image_tilt, registered_image_shifted_tilt)

    errors.append([np.linalg.norm((_ - reference_image).ravel()) for _ in images_to_compare])

errors = np.asarray(errors) / np.linalg.norm(reference_image.ravel())

fig, axs = plt.subplots(4)
for ax, image in zip(axs, images_to_compare):
    ax.imshow(complex2rgb(image, 10))
plt.show(block=True)

# # print(np.max(errors[]))
plt.plot(scan_positions, errors)
plt.legend(['non-registered', 'shift', 'tilt', 'shift-tilt'])
plt.xlabel('shift distance, $\mu m$')
plt.ylabel('std')
plt.show(block=True)

# registered_image = register(images[nb], images[0], precision=0.2)
# fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
# for idx in range(2):
#     axs[idx].imshow(images[int(idx * nb), ...])
# axs[2].imshow(complex2rgb(registered_image.image)) #.astype(np.float))
# print(np.std(registered_image.image - images[0]))
# print(np.std(registered_image.image))
# plt.show()