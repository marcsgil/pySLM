import numpy as np
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer
from datetime import datetime

from optics.instruments.stage.nanostage import NanostageLT3
from optics.utils.ft import Grid
from optics.utils.display import grid2extent, complex2rgb
# from projects.confocal_interference_contrast.scan_utils.scan2dic import scan2dic  # used with external reference
# from polarization_memory_effect.polarization_memory_effect_functions import parallel_cam_acquisitions, phase_amp_result
from polarization_memory_effect.polarization_memory_effect_functions import set_exposure_time, create_2_spots
from optics.instruments.cam.ids_cam import IDSCam, Roi
from scan_utils.dpc_calibration import DPC_calibration
from optics.instruments.slm import PhaseSLM
from optics.calc import correction_from_pupil
from projects.confocal_interference_contrast import log
from pathlib import Path
from tqdm import tqdm
# Axes are inverted in grid to match the display (they are switched in scanning)
# grid = Grid(extent=(22e-6, 22e-6)[::-1], step=1e-6, center=(103e-6, 73e-6)[::-1]) # (x,y) 187e-6, 163e-6
#
# depth = 195e-6

# # For calibration with IDSCam app
# dpc_cali = DPC_calibration(load=True, open_cams=False)
# input("SLM initalised... Proceed to open cams")
# dpc_cali.close()
# del dpc_cali

# dpc_cali = DPC_calibration(load=True, open_cams=False)  # Contains calibrated cameras

run_scan = True

extra_info = 'none'
def scan_xy(extra_info, exposure_factor, grid: Grid, depth: float, dpc_cali = None):
    polarization_phasors = np.exp(2j * np.pi / 4 * np.arange(4))
    try:
        # Stage parameters
        stage = NanostageLT3()
        scan_axes = (1, 2)
        initial_position = [depth, *grid.center[::-1]]  # TODO remove flipping when it's fixed elsewhere
        for idx, coord in enumerate(initial_position):
            stage.move(idx, coord * 1e6, 'um')

        # Responsible for Boustrophedon scan directions
        boustrophedon_order = (np.arange(1, np.size(grid[0]) + 1) % 2) * grid.shape[
            1]  # image insert order (max, 0, max, etc...) # TODO non-square bug
        boustrophedon_grid = np.repeat(grid[1].ravel()[np.newaxis, :], grid.shape[0], axis=0)
        boustrophedon_grid[1::2] = boustrophedon_grid[1::2, ::-1]
        start_time, nb_iterations = timer(), np.prod(grid.shape)

        # TODO: do this in DPC Cali once the ultimate solution reached
        cam = IDSCam(serial=4103698194, exposure_time=50 * 1e-3, gain=0.0, normalize=False)
        cam_saturation = 2**cam.bit_depth - 1

        set_exposure_time(cam, exposure_factor)# TODO extra_info)
        measurement_centre = np.array([1186, 1588])
        measurement_diameter = 1200
        cam.roi = Roi(center=measurement_centre, width=measurement_diameter, height=measurement_diameter)

        def check_saturation(name, img):
            # Check for underflow
            if np.amin(img) <= 0.0:
                log.warning(f'Complete darkness of {name} in {np.sum(img <= 0.0)} pixels.')
            # Check for overflow
            if np.amax(img) >= cam_saturation:
                log.warning(f'Saturation of {name} in {np.sum(img >= cam_saturation)} pixels.')

        nb_detectors = 1
        camera_data = [[] for _ in range(nb_detectors)]
        for idx_row, coord_y in enumerate(tqdm(grid[0].flat)):
            stage.move(scan_axes[1], coord_y * 1e9, 'nm')
            for append_row in range(nb_detectors):
                camera_data[append_row].append([])

            for idx_col, coord_x in enumerate(boustrophedon_grid[idx_row]):
                stage.move(scan_axes[0], coord_x * 1e9, 'nm')
                if idx_row == 0 and coord_x == boustrophedon_grid[idx_row][0]:
                    time.sleep(0.3)  # pauses for first frame
                else:
                    time.sleep(cam.frame_time * 2)

                # acquired = parallel_cam_acquisitions(dpc_cali.cam_reflection, dpc_cali.cam_transmission,
                #                                 dpc_cali.cam_aux, y_center=y_center, x_center=x_center,
                #                                 mask=1) #dpc_cali.mask)     # camera acquisition

                # fig, axs = plt.subplots(1, 4)  # TODO debugging code
                # for idx, _ in enumerate(acquired[0]):
                #     axs[idx].imshow(_)
                # plt.show(block=True)

                for cam_idx in range(nb_detectors):
                    img = cam.acquire()
                    centre = np.array(img.shape) // 2
                    intensity_d = np.mean(img[:centre[0], :centre[1]])
                    intensity_l = np.mean(img[centre[0]:, :centre[1]])
                    intensity_a = np.mean(img[:centre[0], centre[1]:])
                    intensity_r = np.mean(img[centre[0]:, centre[1]:])
                    intensities_img = np.array([intensity_d, intensity_l, intensity_a, intensity_r]).reshape(2, 2)
                    camera_data[cam_idx][idx_row].insert(boustrophedon_order[idx_row], intensities_img)
                    del img

                    # centre = np.array(img.shape[-2:]) // 2
                    # check_saturation('top-left', img[..., :centre[0], :centre[1]])
                    # check_saturation('bottom-left', img[..., centre[0]:, :centre[1]])
                    # check_saturation('top-right', img[..., :centre[0], centre[1]:])
                    # check_saturation('bottom-right', img[..., centre[0]:, centre[1]:])
                    # check_saturation('camera', img)

        # status monitoring (unimportant for operation) ------------------------------
        #     current_iteration = (idx_row + 1) * np.prod(grid.shape[1:])
        #     if idx_row == 0:
        #         iteration_time = timer() - start_time
        #     elif idx_row == 4:
        #         iteration_time = (timer() - start_time) / 5
        #     time_left = iteration_time * (grid.shape[0] - idx_row - 1)
        #     time_units = 's'
        #     if time_left > 3600:
        #         time_left, time_units = time_left / 3600, 'h'
        #     elif time_left > 60:
        #         time_left, time_units = time_left / 60, 'min'
        #     print(f'Progress is {current_iteration / nb_iterations * 100:.2f}%' +
        #           f' with {time_left:.2f} {time_units} left.')
            # -----------------------------------------------------------------------------

        # Data processing and saving -------------------------------------------------
        print("Processing data...")
        camera_data = np.array(camera_data)[0]  # dims: cam, scan_y, scan_x, img_y, img_x
        save_folder = r"results/"
        scan_name = "DPC_scan"
        save_parameters = f'_{grid.shape[0]}x{grid.shape[1]}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        centre = np.array(camera_data.shape[-2:]) // 2
        # intensity_d = np.mean(camera_data[..., :centre[0], :centre[1]], axis=(-1, -2))
        # intensity_l = np.mean(camera_data[..., centre[0]:, :centre[1]], axis=(-1, -2))
        # intensity_a = np.mean(camera_data[..., :centre[0], centre[1]:], axis=(-1, -2))
        # intensity_r = np.mean(camera_data[..., centre[0]:, centre[1]:], axis=(-1, -2))
        intensity_d = camera_data[..., 0, 0]
        intensity_l = camera_data[..., 1, 0]
        intensity_a = camera_data[..., 0, 1]
        intensity_r = camera_data[..., 1, 1]
        intensities = np.array([intensity_d, intensity_l, intensity_a, intensity_r])
        average_intensity = np.mean(intensities, axis=0)
        complex_amplitude = sum(i * _ for i, _ in zip(intensities, polarization_phasors)) / polarization_phasors.size / average_intensity
        complex_amplitude[np.isnan(complex_amplitude)] = 0.0

        np.savez(save_folder + scan_name + save_parameters + '.npz', camera_data=camera_data,
                 complex_amplitude=complex_amplitude, scan_size=grid.shape, step_size=grid.step)

        # external_dic = scan2dic(camera_data)  # TODO process data for DPC

        del camera_data
        # -----------------------------------------------------------------------------

        # recentering
        for idx, coord in enumerate(initial_position):
            stage.move(idx, coord * 1e6, 'um')

        fig, axs = plt.subplots(2, 4)

        ax_labels = {}
        for idx, letter in enumerate('xyz'):
            ax_labels[idx] = letter

        log.info('Calculating a phase factor to fix the background (edge) to 0.')
        distance_from_edge = np.amin(
            np.broadcast_arrays(*(np.minimum(_ - _.ravel()[0], _.ravel()[-1] - _) for _ in grid)),
            axis=0
        )
        background_edge_width = 3e-6  # The edge over which to average to get the background phase
        weight = np.maximum(0, background_edge_width - distance_from_edge) / background_edge_width
        edge_phasor = np.dot(complex_amplitude.ravel(), weight.ravel())
        edge_phasor /= np.abs(edge_phasor)
        log.info(f'Background phase is {np.angle(edge_phasor) * 180 / np.pi:0.1f}°.')

        extent = grid2extent(grid * 1e6)
        axs[0, 0].imshow(intensity_d, extent=extent)
        axs[0, 0].set_title(f'diagonal [{np.amin(intensity_d)/cam_saturation:0.6f}->{np.amax(intensity_d)/cam_saturation:0.6f}]')
        axs[0, 1].imshow(intensity_l, extent=extent)
        axs[0, 1].set_title(f'left [{np.amin(intensity_l)/cam_saturation:0.6f}->{np.amax(intensity_l)/cam_saturation:0.6f}]')
        axs[0, 2].imshow(intensity_a, extent=extent)
        axs[0, 2].set_title(f'anti-diagonal [{np.amin(intensity_a)/cam_saturation:0.6f}->{np.amax(intensity_a)/cam_saturation:0.6f}]')
        axs[0, 3].imshow(intensity_r, extent=extent)
        axs[0, 3].set_title(f'right [{np.amin(intensity_r)/cam_saturation:0.6f}->{np.amax(intensity_r)/cam_saturation:0.6f}]')
        axs[1, 0].imshow(np.abs(complex_amplitude), extent=extent)
        axs[1, 0].set_title(f'visibility [{np.amin(np.abs(complex_amplitude))*100:0.1f}%->{np.amax(np.abs(complex_amplitude))*100:0.1f}%]')
        axs[1, 1].imshow(complex2rgb(np.exp(1j*np.angle(complex_amplitude))), extent=extent)
        axs[1, 1].set_title(f'phase [{np.amin(np.abs(complex_amplitude))*100:0.1f}%->{np.amax(np.abs(complex_amplitude))*100:0.1f}%]')
        # axs[1, 2].imshow(complex2rgb(complex_amplitude), extent=extent)
        # axs[1, 2].set_title(f'complex amplitude [{np.amin(np.abs(complex_amplitude))*100:0.1f}%->{np.amax(np.abs(complex_amplitude))*100:0.1f}%]')
        axs[1, 2].imshow(complex2rgb(complex_amplitude, 1), extent=extent)
        axs[1, 2].set_title(f'norm. comp. amp. [{np.amin(np.abs(complex_amplitude))*100:0.1f}%->{np.amax(np.abs(complex_amplitude))*100:0.1f}%]')
        axs[1, 3].imshow(complex2rgb(complex_amplitude / edge_phasor, 1), extent=extent)
        axs[1, 3].set_title(f'fixed phase {np.angle(edge_phasor) * 180 / np.pi:0.1f}° [{np.amin(np.abs(complex_amplitude))*100:0.1f}%->{np.amax(np.abs(complex_amplitude))*100:0.1f}%]')
        for ax in axs.ravel():
            ax.set(xlabel=f'{ax_labels[scan_axes[0]]} $\mu m$', ylabel=f'{ax_labels[scan_axes[1]]} $\mu m$')

        plt.draw_all()
        plt.show(block=False)
        plt.get_current_fig_manager().window.showMaximized()
        plt.pause(0.01)
        output_file = save_folder + scan_name + save_parameters + 'z_' + str(depth*1e6) +'exp_factor_' + str(exposure_factor) + str(extra_info) +'.png'
        log.info(f'Saving to {output_file}...')
        fig.savefig(output_file)
    finally:
        stage.close()
        cam.disconnect()
        if dpc_cali is not None:
            dpc_cali.close()

def z_adjustment(z_initial: float = 150e-6,delta_z: float = 20e-6, steps: int = 10):
    depth = np.linspace(z_initial - delta_z, z_initial + delta_z, steps)
    for _ in range(steps):
        info = ''
        exposure_factor = 0.35
        scan_xy(info, exposure_factor, grid, depth[_], dpc_cali=None)

def exposure_adj(exp_f_initial: float = 0.5, delta: float = 0.4, steps: int = 10):
    exposure_factor = np.linspace(exp_f_initial - delta, exp_f_initial + delta, steps)
    for _ in range(steps):
        info = ''
        scan_xy(info, exposure_factor[_], grid, depth, dpc_cali=None)

def rel_amp_adj(rel_amp_initial: float = 1.0, delta: float = 0.05, steps: int = 10):
    rel_amp = np.linspace(rel_amp_initial - delta, rel_amp_initial + delta, steps)
    for _ in range(steps):
        create_2_spots(slm,  rel_amp[_], -0.5e-6)
        info = 'rel_amp_' + str(rel_amp[_])
        scan_xy(info, 0.35, grid, depth, dpc_cali=None)

def separation_adj(separation_initial: float = -0.5e-6, delta: float = 0.1e-6, steps: int = 10):
    separation = np.linspace(separation_initial - delta, separation_initial + delta, steps)
    for _ in range(steps):
        create_2_spots(slm,  1.0275, separation[_])
        info = 'separation_' + str(separation[_])
        scan_xy(info, 0.35, grid, depth, dpc_cali=None)

def test_illumination_offset(x_initial: float = 0, delta: float = 0.01, steps: int = 10):
    x_off = np.linspace(x_initial - delta, x_initial + delta, steps)
    for _ in range(steps):
        x_offset = [-0.021 + x_off[_], 0.04 + x_off[_] , 0]
        create_2_spots(slm, 1.0275, -0.5e-6, 0.0, x_offset)
        info = 'x_off_' + str(x_off[_])
        scan_xy(info, 0.35, grid, depth, dpc_cali=None)

def test_illumination_offset_2(x_initial: float = 0.00, delta: float = 0.05, steps: int = 10):
    x_off = np.linspace(x_initial - delta, x_initial + delta, steps)
    for _ in range(steps):
        x_offset = [0, x_off[_], 0]
        create_2_spots(slm, 1.0275, -0.5e-6, 0.0, x_offset)
        info = 'y_off2_' + str(x_off[_])
        scan_xy(info, 0.35, grid, depth, dpc_cali=None)




if __name__ == "__main__":
    with PhaseSLM(display_target=0, deflection_frequency=(0, -1/4), two_pi_equivalent=0.3) as slm:
        input_config_file_path = r'C:\Users\lab\OneDrive - University of Dundee\Documents\lab\code\python\optics\results\aberration_correction_2022-05-12_15-54-37.439768.npz'
        aberration_correction_settings = np.load(input_config_file_path)
        pupil_aberration = aberration_correction_settings['pupil_aberration']
        log.info('Opening the spatial light modulator...')
        slm.correction = correction_from_pupil(pupil_aberration, attenuation_limit=4)
        slm.post_modulation_callback = lambda _: time.sleep(100e-3)
        create_2_spots(slm,  1.0275, -0.5e-6)
        # grid = Grid(extent=(18e-6, 18e-6)[::-1], step=0.5e-6, center=(72e-6, 135e-6)[::-1]) # (x,y) 187e-6, 163e-6
        grid = Grid(extent=(38e-6, 62e-6)[::-1], step=0.25e-6, center=(75e-6, 130e-6)[::-1])
        depth = 147e-6
        scan_xy('sc_000', 0.35, grid, depth, dpc_cali=None)
        # grid = Grid(extent=(30e-6, 30e-6)[::-1], step=0.25e-6, center=(151e-6, 148e-6)[::-1]) # (x,y) 187e-6, 163e-6
        # scan_xy('sc_005', 0.35, grid, depth, dpc_cali=None)
        # grid = Grid(extent=(30e-6, 30e-6)[::-1], step=0.125e-6, center=(151e-6, 148e-6)[::-1]) # (x,y) 187e-6, 163e-6
        # scan_xy('sc_005', 0.35, grid, depth, dpc_cali=None)
        # scan_xy('sc_005', 0.35, grid, depth, dpc_cali=None)
        # grid = Grid(extent=(25e-6, 25e-6)[::-1], step=0.25e-6, center=(212e-6, 143e-6)[::-1]) # (x,y) 187e-6, 163e-6
        # scan_xy('sc_005', 0.35, grid, depth, dpc_cali=None)

        # rel_amp_initial = 1.0275
        # delta = 0.0005
        # steps = 4
        # rel_amp_adj(rel_amp_initial, delta, steps)

        # exp_f_initial = 0.3
        # delta = 0.1
        # steps= 3
        # exposure_adj(exp_f_initial, delta, steps)

        # extra_info = 'ref'
        # scan_xy('', 0.5, grid, depth, dpc_cali=None)

        # test_illumination_offset()
        # separation_adj()
        #test_illumination_offset_2()

        z_initial = 146e-6
        delta_z = 4e-6
        steps = int(5)
        z_adjustment(z_initial, delta_z, steps)


        # info = 'real_amp_1.027_cam_time_0.5'
        # scan_xy(info, grid, depth, dpc_cali=None)
        # rel_amp = np.linspace(1.02, 1.05, 10)
        # for _ in rel_amp:
        #     create_2_spots(slm,  _, -0.5e-6)
        #     time.sleep(0.2)
        #     info = f'relative_amp_{_:.3f}'
        #     scan_xy(info, grid, depth, dpc_cali=None)
        plt.show(block=True)