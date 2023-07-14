import numpy as np
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer

from optics.instruments.stage.nanostage import NanostageLT3
from optics.instruments.cam.ids_cam import IDSCam
from optics.utils.ft import Grid
from optics.utils.display import grid2extent, complex2rgb
from projects.confocal_interference_contrast.complex_phase_image import get_registration


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


# Axes are inverted in grid to match the display (they are switched in scanning)
grid = Grid(extent=(20e-6, 20e-6)[::-1], step=0.3e-6, center=(150e-6, 150e-6)[::-1])  # TODO bug: non-square side of the scan mimics the square side in every second line
depth = 150e-6
save_all_data = False
correct_aberration = False

if correct_aberration:
    aberration_reference = np.load(r'C:\Users\Laurynas\Desktop\Python apps\Shared_Lab\lab\code\pyt'
                                   r'hon\optics\projects\confocal_interference_contrast\aberration_reference.npy')
else:
    aberration_reference = 1


def scan_xy(grid: Grid, depth: float, scan_queue: bool = False):
    # Stage parameters
    stage = NanostageLT3()
    scan_axes = (1, 2)
    initial_position = [depth, *grid.center[::-1]]  # TODO remove flipping when it's fixed elsewhere
    for idx, coord in enumerate(initial_position):
        stage.move(idx, coord * 1e6, 'um')

    # Camera parameters
    nb_cams = 1
    cameras = [IDSCam(exposure_time=5e-3, normalize=False) for _ in range(nb_cams)]
    # camera1.roi = Roi(center=(524, 642), shape=(3, 3))  # Select region of interest
    try:

        # -----------------------------------------------------------------------------
        """ 
        Calculating operators 2 convert intensity image to complex phase image. These require registration
        of the interference pattern which can be either calculated from a new capture or loaded from settings
        JSON located in project folders. Uncomment the option that will be used.
        """
        phase_operators = []

        # Upload registration from json file
        for idx, cam in enumerate(cameras):
            test_shot = cam.acquire()
            registration = get_registration(f"HV_DIC_registration_CAM{idx}")  # loads a pre-calculated registration
            # registration = registration_calibration(test_shot, f"HV_DIC_registration_CAM{idx}")  # calculates registration from a new capture
            image_grid = Grid(shape=np.shape(test_shot))
            # Calculates phase operator using a single point Fourier transform
            phase_operator = np.exp(np.array(1j * np.sum(
                [r * k for r, k in zip(image_grid, np.array(image_grid.k.step) * np.array(registration.shift))])))
            phase_operators.append(phase_operator)
            del test_shot, registration, image_grid, phase_operator
        # -----------------------------------------------------------------------------

        # Responsible for Boustrophedon scan directions
        boustrophedon_order = (np.arange(1, np.size(grid[0]) + 1) % 2) * grid.shape[
            1]  # image insert order (max, 0, max, etc...) # TODO non-square bug
        boustrophedon_grid = np.repeat(grid[1].ravel()[np.newaxis, :], grid.shape[0], axis=0)
        boustrophedon_grid[1::2] = boustrophedon_grid[1::2, ::-1]
        start_time, nb_iterations = timer(), prod(grid.shape)

        if save_all_data:
            full_image_set = np.zeros((*grid.shape, *np.shape(phase_operators[0])),
                                      dtype=np.uint8)  # uncomment all instances 4 whole imgs
        images = [[] for _ in cameras]
        complex_phase_images = [[] for _ in cameras]
        for idx_row, coord_y in enumerate(grid[0].flat):
            stage.move(scan_axes[1], coord_y * 1e9, 'nm')
            for cam_idx in range(nb_cams):
                images[cam_idx].append([])
                complex_phase_images[cam_idx].append([])

            for idx_col, coord_x in enumerate(boustrophedon_grid[idx_row]):
                stage.move(scan_axes[0], coord_x * 1e9, 'nm')
                if idx_row == 0 and coord_x == boustrophedon_grid[idx_row][0]:
                    time.sleep(0.3)  # pauses for first frame
                else:
                    time.sleep(cameras[0].frame_time * 2)
                for cam_idx in range(nb_cams):
                    cam_frame = cameras[cam_idx].acquire()
                    # TODO new
                    boustrophedon_factor = ((idx_row + 1) % 2 - 0.5) * 2  # TODO write better code for boustrophedon fliping
                    boustrophedon_compensator = idx_row % 2  # TODO horrible
                    if save_all_data:
                        full_image_set[idx_row, int((idx_col + boustrophedon_compensator) * boustrophedon_factor), ...] = cam_frame  # Saving the whole image
                    cam_frame = cam_frame.astype(dtype=np.float32) / 255
                    images[cam_idx][idx_row].insert(boustrophedon_order[idx_row], np.mean(cam_frame))
                    complex_phase_images[cam_idx][idx_row].insert(boustrophedon_order[idx_row],
                                                                  np.mean(cam_frame * phase_operators[cam_idx] /
                                                                          aberration_reference))
                    # TODO old
                    # images[cam_idx][idx_y].append(np.mean(cam_frame))
                    # complex_phase_images[cam_idx][idx_y].append(np.mean(cam_frame * phase_operators[cam_idx]))

            # status monitoring (unimportant for operation) ------------------------------
            current_iteration = (idx_row + 1) * prod(grid.shape[1:])
            if idx_row == 0:
                iteration_time = timer() - start_time
            elif idx_row == 4:
                iteration_time = (timer() - start_time) / 5
            time_left = iteration_time * (grid.shape[0] - idx_row - 1)
            time_units = 's'
            if time_left > 3600:
                time_left, time_units = time_left / 3600, 'h'
            elif time_left > 60:
                time_left, time_units = time_left / 60, 'min'
            print(f'Progress is {current_iteration / nb_iterations * 100:.2f}%' +
                  f' with {time_left:.2f} {time_units} left.')
            # -----------------------------------------------------------------------------

        images = np.asarray(images, dtype=float)
        complex_phase_images = np.asarray(complex_phase_images, dtype=complex)

        scan_serial_nb = '0049'
        if save_all_data:
            np.save(
                f'{scan_serial_nb}FULL - Single cam - cheek cells behind 2 layers of lens tissue in water confocal scan - {int(grid.extent[0] * 1e6)}x{int(grid.extent[1] * 1e6)}um, {grid.step[0] * 1e6:.2f}um step.npy',
                full_image_set)
        np.save(
            f'{scan_serial_nb}I - Single cam - cheek cells behind 2 layers of lens tissue in water confocal scan - {int(grid.extent[0] * 1e6)}x{int(grid.extent[1] * 1e6)}um, {grid.step[0] * 1e6:.2f}um step.npy',
            images)
        np.save(
            f'{scan_serial_nb}CP - Single cam - cheek cells behind 2 layers of lens tissue in water confocal scan - {int(grid.extent[0] * 1e6)}x{int(grid.extent[1] * 1e6)}um, {grid.step[0] * 1e6:.2f}um step.npy',
            complex_phase_images)

        # recentering
        for idx, coord in enumerate(initial_position):
            stage.move(idx, coord * 1e6, 'um')

        imgs = np.empty((2, nb_cams), dtype=object)
        fig, axs = plt.subplots(*imgs.shape)

        ax_labels = {}
        for idx, letter in enumerate('xyz'):
            ax_labels[idx] = letter

        ax_intensity = axs[0] if nb_cams > 1 else [axs[0]]
        ax_complex = axs[1] if nb_cams > 1 else [axs[1]]
        extent = grid2extent(grid * 1e6)
        extent = list(extent[:2]) + list(extent[2:])
        for idx_col, axi, axc in zip(range(nb_cams), ax_intensity, ax_complex):
            imgs[0, idx_col] = axi.imshow(images[idx_col], extent=extent)
            imgs[1, idx_col] = axc.imshow(complex2rgb(complex_phase_images[idx_col], 1), extent=extent)

            axi.set(xlabel=f'{ax_labels[scan_axes[0]]} $\mu m$', ylabel=f'{ax_labels[scan_axes[1]]} $\mu m$')
            axi.set_title(depth)
            axc.set(xlabel=f'{ax_labels[scan_axes[0]]} $\mu m$', ylabel=f'{ax_labels[scan_axes[1]]} $\mu m$')

            colourbar(imgs[0, idx_col])
        plt.show(block=False)
    finally:
        for cam in cameras:
            cam.power_down()
            cam.disconnect()
        stage.close()


def scan_depths(centre, step, nb_measurements):
    depth_array = centre + (np.arange(nb_measurements) - nb_measurements // 2) * step
    for depth in depth_array:
        scan_xy(grid, depth)


# time.sleep(15)
try:
    scan_xy(grid, depth)
    # scan_depths(144e-6, step=2e-6, nb_measurements=9)
    # send_email('Scan finished', receiver_email='lau.valantinas@gmail.com')
except Exception as e:
    print(e)
    # send_email(str(e), receiver_email='lau.valantinas@gmail.com')

plt.show(block=True)
