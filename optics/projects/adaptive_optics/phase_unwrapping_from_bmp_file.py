from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
from optics.calc.interferogram import Interferogram
from optics.utils.display import complex2rgb
from skimage.restoration import unwrap_phase
# from scipy.signal import unwrap
# from mpl_toolkits.mplot3d import Axes3D
from optics.utils import ft
from optics.calc import zernike

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


def create_aperture(px, py, x0, y0, r):
    # Create an array of x and y coordinates
    x = np.arange(px)[:, np.newaxis]
    y = np.arange(py)
    # Calculate the squared distance from each pixel to the central pixel
    distance_sq = (x - x0)**2 + (y - y0)**2
    # Create the aperture array
    aperture = np.where(distance_sq <= r**2, 1, 0)

    return aperture


def plot_curve(x, y, title, xlabel, ylabel, linewidth=1, linecolor='blue', marker='o', markersize=5):
    plt.plot(x, y, linewidth=linewidth, color=linecolor, marker=marker, markersize=markersize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)  # Add gridlines
    # plt.show()


def plot_2d_matrix_3d(matrix_2d):
    # Get the shape of the matrix
    rows, cols = matrix_2d.shape

    # Create a meshgrid for the X and Y coordinates
    x, y = np.meshgrid(range(cols), range(rows))

    # Flatten the matrix to get the Z coordinate
    z = matrix_2d.flatten()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface
    ax.plot_surface(x, y, z.reshape(rows, cols))

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of 2D Matrix')

    # Show the plot
    plt.show()


def plot_2d_matrix_wireframe(matrix_2d, order, scale):
    # Get the shape of the matrix
    rows, cols = matrix_2d.shape

    # Generate the x and y coordinates
    x = np.arange(cols)
    y = np.arange(rows)

    # Create a meshgrid for the x and y coordinates
    X, Y = np.meshgrid(x, y)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D wireframe
    ax.plot_wireframe(X, Y, matrix_2d, color='green')

    # # Turn off/on axis
    # plt.axis('off')
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Zernike Order {order} scale {scale}')

    # Show the plot
    # plt.show()


def crop_matrix(mb, x0, y0, l0):
    # Calculate the starting and ending indices for rows and columns
    start_row = x0 - (l0 // 2)
    end_row = start_row + l0
    start_col = y0 - (l0 // 2)
    end_col = start_col + l0

    # Extract the smaller matrix
    ms = mb[start_row:end_row, start_col:end_col]

    return ms


folder_loop = 0
mainfolder_txt = 'E:/Adaptive Optics/Experimental MLAO/Interference_profile_to_command_4fsystem_v2/'
# Specify the folder path where the images are located
# folder_path_reference = "E:/Adaptive Optics/Experimental MLAO/Interference_profile_to_command/Loop 0/Zernike order 0.0/"
folder_path_reference = f'{mainfolder_txt}/Loop {folder_loop}/Reference/'
# folder_path_reference = f'E:\Adaptive Optics\Experimental MLAO\Interference_profile_to_command_4fsystem/Reference/'

# Call the function to read the gray images
image_names_ref, images_ref = read_gray_images(folder_path_reference)

# To do the phase unwrapping for the reference order which is piston
interferogram_registration = None
reference_phase = []
image_count = 0
crop_side_length = 370
centre_x = 678
centre_y = 415
# To make an aperture
aperture_phase = create_aperture(1024, 1280, centre_y, centre_x, crop_side_length)
# crop_side_length = 320
# # To make an aperture
# aperture_phase = create_aperture(1024, 1280, 458, 635, crop_side_length)

for name_ref, image_ref in zip(image_names_ref, images_ref):

    interferogram = Interferogram(image_ref, registration=interferogram_registration)
    interferogram_registration = interferogram.registration  # So that the next interferograms use the same shift

    wrapped_phase = np.angle(interferogram)
    unwrapped_phase = unwrap_phase(wrapped_phase)
    if image_count == 0:
        reference_phase = unwrapped_phase
    elif image_count > 0:
        reference_phase = reference_phase + unwrapped_phase

    image_count += 1

reference_phase = reference_phase/image_count

fig_M, axs_M = plt.subplots(1, 1, sharex='all', sharey='all', figsize=(18, 8))
im0 = axs_M.imshow(reference_phase*aperture_phase)
axs_M.set(title=f" Reference phase Zernike Order {0}")
fig_M.colorbar(im0)
plt.show()

# To do the phase unwrapping for the other zernike orders
scales = np.linspace(-5, 5, num=41)
zernike_orders = np.linspace(4, 4, num=1)
zernike_orders = zernike_orders.astype(int)

# zernike_orders = np.linspace(1, 30, num=30)
# zernike_orders = zernike_orders.astype(int)
# scales = np.linspace(-15, 15, num=62)

zernike_order = 4
phase_grid = ft.Grid((crop_side_length, crop_side_length), extent=1.0)
coefficients = []
for OSA_order in zernike_orders:

    coefficients_order = []
    # folder_path = f'E:/Adaptive Optics/Experimental MLAO/Interference_profile_to_command/Loop {folder_loop}/Zernike order {OSA_order}.0'
    folder_path = f'{mainfolder_txt}/Loop {folder_loop}/Zernike order {OSA_order}'
    # folder_path = f'E:/Adaptive Optics/Experimental MLAO/Interference_Z2C/Loop {folder_loop}/Zernike order {OSA_order}'
    image_names, images = read_gray_images(folder_path)
    scale_index = 0
    interferogram_registration = None
    folder_path_save = Path.home() / Path(f'{mainfolder_txt}/Data Process/Folder loop {folder_loop}/Zernike OSA order {OSA_order}_Profile_to_Commands_with Aperture_Wireframe/')
    # folder_path_save = Path.home() / Path(f'E:/Adaptive Optics/Experimental MLAO/Interference_Z2C/Data Process/Folder loop {folder_loop}/Zernike OSA order {OSA_order}_Matrix_to_Commands_with Aperture/')
    Zernike_std = zernike.BasisPolynomial(OSA_order).cartesian(*phase_grid)
    com_std_zernike = np.exp(1j*Zernike_std)
    # unwrapped_phase_All = []
    for name, image in zip(image_names, images):

        interferogram = Interferogram(image, registration=interferogram_registration)
        interferogram_registration = interferogram.registration  # So that the next interferograms use the same shift

        wrapped_phase = np.angle(interferogram)
        unwrapped_phase = unwrap_phase(wrapped_phase)
        unwrapped_phase_ref = unwrapped_phase - reference_phase
        # unwrapped_phase_ref = unwrapped_phase
        unwrapped_phase_crop = crop_matrix(unwrapped_phase_ref, centre_y, centre_x, crop_side_length)

        # to perform the standard inner product
        com_phase = np.exp(1j*unwrapped_phase_crop)
        totalE = np.sum(com_phase**2)
        com_phase = com_phase/totalE
        inner_product = np.sum(np.conjugate(com_std_zernike) * com_phase)
        coefficients_order.append(inner_product)

        # unwrapped_phase_All.append(unwrapped_phase_ref*aperture_phase)
        # fig_z, axs_z = plt.subplots(1, 1, sharex='all', sharey='all', figsize=(18, 8))
        # imz = axs_z.imshow(unwrapped_phase_ref)
        # axs_z.set(title=f" Zernike Order {OSA_order} scale {scales[scale_index]}")
        # fig_z.colorbar(imz)
        # plt.show()

        # save the figure to png file
        # plot_2d_matrix_3d(Zernike_std)

        plot_2d_matrix_wireframe(unwrapped_phase_crop, OSA_order, scales[scale_index])
        timestamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-3]
        folder_path_save.mkdir(parents=True, exist_ok=True)
        figure_name = folder_path_save / f'Time_{timestamp}_Unwrapped_phase_Scale_{scales[scale_index]}_OSA_order_{OSA_order}.png'
        plt.savefig(figure_name)
        plt.close()

        scale_index += 1

    plot_curve(scales, np.array(coefficients_order), f'Inner product with standard Zernike order = {OSA_order}', 'Scale', 'inner product')
    inner_figure_name = folder_path_save / f'Time_{timestamp}_InnerProduct_with_standard_Zernike_OSA_order_{OSA_order}.png'
    plt.savefig(inner_figure_name)
    plt.close()
    coefficients.append(coefficients_order)
    # plot_2d_matrices_wireframe(unwrapped_phase_All)


