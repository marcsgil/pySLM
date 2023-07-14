import numpy as np
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer
from datetime import datetime
from pathlib import Path

from optics.instruments.stage.nanostage import NanostageLT3
from optics.instruments.cam.ids_cam import IDSCam
from optics.utils.roi import Roi
from optics.utils.ft import Grid
from optics.utils.display import grid2extent, complex2rgb
from projects.confocal_interference_contrast.complex_phase_image import get_registration, registration_calibration
from projects.confocal_interference_contrast.scan_utils.scan2dic import scan2dic
from optics.instruments.slm import PhaseSLM
from optics.calc import correction_from_pupil
from polarization_memory_effect.polarization_memory_effect_functions import create_2_spots, calibration_protocol, tom_cam_acquisitions

cam_dict = {
    "cam_aux": 4103698194,
    "cam_transmission": 4103198121,
    "cam_reflection": 4103697390
}


#test part
#serial=4103697390 --- center of mass center=(516, 710)
#serial=4103198121 --- center of mass center=(530, 640)
# with IDSCam(serial=4103198121, exposure_time=50e-3, normalize=False) as cam:
#     from scipy import ndimage
#     time.sleep(0.5)
#     center = []
#     for _ in range(5):
#         img = cam.acquire()
#         center.append(ndimage.center_of_mass(img))
#     center = np.asarray(center)
#     center = np.mean(center, axis=0, dtype = np.int16)
#     print(f'Center of mass: {center}')
#     #cam.roi = Roi(center=(center[0], center[1]), shape=(870, 870))
#     cam.roi = Roi(center=(530, 640), shape=(870, 870))
#     img2= cam.acquire()
#     fig = plt.figure()
#     plt.imshow(img)
#     fig2 = plt.figure()
#     plt.imshow(img2)
#     plt.show()


# Axes are inverted in grid to match the display (they are switched in scanning)
grid = Grid(extent=(15e-6, 15e-6)[::-1], step=0.5e-6, center=(150e-6, 150e-6)[::-1])
depth = 150e-6


def scan_xy(grid: Grid, depth: float, beam_separation=0):
    # Stage parameters
    stage = NanostageLT3()
    scan_axes = (1, 2)
    initial_position = [depth, *grid.center[::-1]]  # TODO remove flipping when it's fixed elsewhere
    for idx, coord in enumerate(initial_position):
        stage.move(idx, coord * 1e6, 'um')
    # input("procceed?")

    # Camera parameters
    serial_nbs = [4103697390, 4103198121]
    roi = [(516, 710), (530, 640)]
    nb_cams = len(serial_nbs)
    exposures = [50e-3, 50e-3]
    try:
        cameras = [IDSCam(serial=_, exposure_time=et, normalize=False) for _, et in zip(serial_nbs, exposures)]
        for _, cam in enumerate(cameras):
            cam.roi = Roi(center=roi[_], shape=(870, 870))

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
            # registration = registration_calibration(test_shot, f"HV_DIC_registration_CAM{idx}")
            image_grid = Grid(shape=np.shape(test_shot))
            # Calculates phase operator using a single point Fourier transform
            phase_operator = np.exp(np.array(1j * sum(r * k for r, k in zip(image_grid, np.array(image_grid.k.step) * np.array(registration.shift)))))
            phase_operators.append(phase_operator)
            del test_shot, registration, image_grid, phase_operator
        phase_operators = np.array(phase_operators)
        # -----------------------------------------------------------------------------

        # Responsible for Boustrophedon scan directions
        boustrophedon_order = (np.arange(1, np.size(grid[0]) + 1) % 2) * grid.shape[
            1]  # image insert order (max, 0, max, etc...) # TODO non-square bug
        boustrophedon_grid = np.repeat(grid[1].ravel()[np.newaxis, :], grid.shape[0], axis=0)
        boustrophedon_grid[1::2] = boustrophedon_grid[1::2, ::-1]
        start_time, nb_iterations = timer(), np.prod(grid.shape)

        camera_data = [[] for _ in cameras]
        for idx_row, coord_y in enumerate(grid[0].flat):
            stage.move(scan_axes[1], coord_y * 1e9, 'nm')
            for cam_idx in range(nb_cams):
                camera_data[cam_idx].append([])

            for idx_col, coord_x in enumerate(boustrophedon_grid[idx_row]):
                stage.move(scan_axes[0], coord_x * 1e9, 'nm')
                if idx_row == 0 and coord_x == boustrophedon_grid[idx_row][0]:
                    time.sleep(0.3)  # pauses for first frame
                else:
                    time.sleep(cameras[0].frame_time * 2)
                for cam_idx in range(nb_cams):
                    cam_frame = cameras[cam_idx].acquire()
                    cam_frame = cam_frame.astype(dtype=np.uint8)
                    camera_data[cam_idx][idx_row].insert(boustrophedon_order[idx_row], cam_frame)

            # status monitoring (unimportant for operation) ------------------------------
            current_iteration = (idx_row + 1) * np.prod(grid.shape[1:])
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

        # Data processing and saving -------------------------------------------------
        print("Processing data...")
        camera_data = np.array(camera_data)  # dims: cam, scan_y, scan_x, img_y, img_x
        save_name = f'{beam_separation * 1e6:.3g}um_separated_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        np.savez(save_name + '.npz',  camera_data=camera_data)
        #         # phase_operators=phase_operators,
        #         # external_dic=external_dic, complex_images=complex_images)
        camera_images = np.mean(camera_data, axis=(-1, -2))
        external_dic = scan2dic(camera_data)

        del camera_data
        # -----------------------------------------------------------------------------

        # recentering
        for idx, coord in enumerate(initial_position):
            stage.move(idx, coord * 1e6, 'um')

        imgs = np.empty((2, nb_cams), dtype=object)
        fig, axs = plt.subplots(*imgs.shape)

        ax_labels = {}
        for idx, letter in enumerate('xyz'):
            ax_labels[idx] = letter

        extent = grid2extent(grid * 1e6)
        for idx_col in range(nb_cams):
            imgs[0, idx_col] = axs[0, idx_col].imshow(camera_images[idx_col], extent=extent)
            imgs[1, idx_col] = axs[1, idx_col].imshow(complex2rgb(external_dic, normalization=1.0), extent=extent)
            # imgs[1, idx_col] = axs[1, idx_col].imshow(complex2rgb(complex_images[idx_col], 1), extent=extent)

            axs[0, idx_col].set(xlabel=f'{ax_labels[scan_axes[0]]} $\mu m$', ylabel=f'{ax_labels[scan_axes[1]]} $\mu m$')
            axs[1, idx_col].set(xlabel=f'{ax_labels[scan_axes[0]]} $\mu m$', ylabel=f'{ax_labels[scan_axes[1]]} $\mu m$')
            # axs[1, idx_col].set(xlabel=f'{ax_labels[scan_axes[0]]} $\mu m$', ylabel=f'{ax_labels[scan_axes[1]]} $\mu m$')
            axs[0, idx_col].set_title(f"{beam_separation*1e6:.3g}$\mu m$ Beam separation")
            axs[1, idx_col].set_title(f"External DIC")
            # axs[1, idx_col].set_title(f"Phase contrast")

        plt.show(block=False)
        plt.savefig(save_name+".png")
    finally:
        for cam in cameras:
            cam.power_down()
            cam.disconnect()
        stage.close()


# if __name__ == "__main__":
#     for depth in [154.2e-6]:
#         scan_xy(grid, depth, 0.5e-6)
#     plt.show(block=True)
if __name__ == "__main__":
    input_config_file_path = Path('../../results/aberration_correction_2022-05-12_15-54-37.439768.npz').resolve()
    aberration_correction_settings = np.load(input_config_file_path.as_posix())
    pupil_aberration = aberration_correction_settings['pupil_aberration']

    # Calibration
    separation = -0.5e-6

    with PhaseSLM(display_target=0, deflection_frequency=(0, -1/4), two_pi_equivalent=0.3) as slm:
        slm.correction = correction_from_pupil(pupil_aberration, attenuation_limit=4)
        slm.post_modulation_callback = lambda _: time.sleep(100e-3)

        start_point = 0.5e-6
        finish_point = 0.5e-6
        nb_points = 1
        for separation in np.linspace(0, 1e-6, nb_points):
            create_2_spots(slm, 1.05, separation)
            scan_xy(grid, depth, -separation)
    plt.show(block=True)
