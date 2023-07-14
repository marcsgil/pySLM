from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt

from optics.calc.interferogram import Interferogram
from optics.utils.display import complex2rgb
from skimage.restoration import unwrap_phase
# from scipy.signal import unwrap


def read_gray_images(folder_path):
    image_names = []
    images = []

    # Get all files in the folder
    files = os.listdir(folder_path)

    # Iterate over the files
    for file in files:
        if file.endswith(".bmp"):
            image_names.append(file)
            image_path = os.path.join(folder_path, file)
            # Read the image and convert it to grayscale
            img = Image.open(image_path).convert('L')
            img_array = np.array(img)
            images.append(img_array)

    return image_names, images

# Specify the folder path where the images are located
folder_path = "E:/Adaptive Optics/Experimental MLAO/Interference_profile_to_command/Loop 0/Zernike order 0.0/"
# Call the function to read the gray images
image_names, images = read_gray_images(folder_path)

fig, ax = plt.subplots(1, 1)
phase_im = ax.imshow(np.zeros([1024, 1280]))  #, extent=grid2extent(grid))
interferogram_registration = None
# Print the image names and display the images
scales = np.linspace(-2.5, 2.5, num=51)
scale_index = 0

fig_M, axs_M = plt.subplots(1, 1, sharex='all', sharey='all', figsize=(18, 8))
axs_M.set_xlabel("interation")
axs_M.set_ylabel("row of aberration")

for name, image in zip(image_names, images):

    interferogram = Interferogram(image, registration=interferogram_registration)
    print(interferogram)
    interferogram_registration = interferogram.registration  # So that the next interferograms use the same shift
    # interferogram = np.array(interferogram)
    # wrapped_phase = (interferogram - np.min(interferogram)) % (2 * np.pi)
    # Perform phase unwrapping
    wrapped_phase = np.angle(interferogram)
    unwrapped_phase = unwrap_phase(wrapped_phase)

    im0 = axs_M.imshow(unwrapped_phase)
    axs_M.set(title=f"Zernike Order {4} scale {scales[scale_index]})")
    plt.pause(0.1)
    plt.show(block=False)
    fig_M.colorbar(im0)
    scale_index += 1

    # print(np.min(unwrapped_phase))
    # print(np.max(unwrapped_phase))

