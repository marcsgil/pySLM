"""
Braian: this may be unused code. TODO
"""

import numpy as np

import matplotlib.pyplot as plt
import time
from datetime import datetime
import pathlib
from tqdm import tqdm
from projects.transverse_structures import log

from optics.instruments.slm.meadowlark_pci_slm import MeadowlarkPCISLM
from optics.instruments.cam.ids_cam import IDSCam
from optics.utils.display import complex2rgb, grid2extent
from tkinter import filedialog as fd
from optics.utils import Roi, ft
from optics.calc.special_beams import hermite_gaussian

if __name__ == '__main__':
    testing = False

    center_adjustment = 0  # offset where the Meadowlark SLM splits
    nb_datasets = 1 if testing else 2
    phase_change_deg = 20 if testing else 2
    nb_images_to_average = 1 if testing else 20

    phases = np.arange(-180, 720, phase_change_deg) * np.pi / 180

    output_path = pathlib.Path(__file__).parent.absolute() / 'output'
    pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)

    with MeadowlarkPCISLM(wavelength=633e-9, deflection_frequency=(0, -1/6)) as slm:
        def modulate_superposition():
            log.info('Placing superposition on SLM...')
            grid = ft.Grid(shape=slm.roi.grid.shape, step=1 / 4.0)  # Step = waist size as fraction of SLM
            super_position_ft = hermite_gaussian(0, 0)(*grid) - 2.7474747474747474 * hermite_gaussian(2, 0)(*grid)
            super_position = ft.fftshift(ft.ifftn(ft.ifftshift(super_position_ft)))
            super_position /= np.amax(np.abs(super_position))
            slm.modulate(super_position)
            time.sleep(0.1)

        with IDSCam(index=0, normalize=True) as cam:
            modulate_superposition()
            log.info(f'cam.shape={cam.shape}')
            cam.roi = None
            log.info(f'cam.roi={cam.roi}')
            log.info('Adjusting camera exposure and centering field-of-view...')
            cam.center_roi_around_peak_intensity(shape=(201, 151), target_graylevel_fraction=1.0)
            log.info(f'Camera exposure: {cam.exposure_time*1e3:0.3f}ms and region of interest: {cam.roi}.')
            cam.exposure_time = 0.063 * 1e-3
            log.info(f'Camera exposure: {cam.exposure_time*1e3:0.3f}ms and region of interest: {cam.roi}.')
            fig = plt.figure()
            test_acq = cam.acquire()
            print(np.amax(test_acq))
            plt.imshow(np.rot90(test_acq))
            plt.show()


            def modulate_acquire_superposition():
                fig, axs = plt.subplots(1, 2)
                measured_images = []
                log.info('Placing superposition on SLM...')
                grid = ft.Grid(shape=slm.roi.grid.shape, step=1 / 4.0)  # Step = waist size as fraction of SLM
                super_position_ft = hermite_gaussian(0, 0)(*grid) - 2.7474747474747474 * hermite_gaussian(2, 0)(*grid)
                super_position = ft.fftshift(ft.ifftn(ft.ifftshift(super_position_ft)))
                super_position /= np.amax(np.abs(super_position))
                ramp = np.ones(super_position.shape, dtype='complex')
                for phase in tqdm(phases):
                    ramp[576::, :] = np.exp(1j*phase)
                    super_position_after = ramp * super_position
                    slm.modulate(super_position_after)
                    time.sleep(0.1)
                    cam.acquire()
                    img = np.mean([cam.acquire() for _ in range(nb_images_to_average)], axis=0)
                    measured_images.append(np.rot90(img))
                output_full_file = output_path / ('photonic_island_nb_images_to_average_' + str(nb_images_to_average) +'_'+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.npz')
                log.info(f'Writing result to {output_full_file}...')
                np.savez(output_full_file, images=measured_images, phases=phases)

            control = ''
            while control != 'exit':
                time.sleep(0.1)
                control = input('1- live video; 2- deflection frequency; 3- Two pi equivalent; 4- modulate; 5- Sample measurement; 6- exposure time '  )
                if control == '1':
                    modulate_superposition()
                    fig1 = plt.figure()
                    ax = plt.imshow(np.zeros(cam.roi.shape))
                    ax.set_clim(0, 1)
                    plt.show(block=False)

                    while plt.fignum_exists(fig1.number) == True:
                        ax.set_data(cam.acquire())
                        plt.show(block=False)
                        plt.pause(0.05)

                elif control =='2':
                    xz = input('New xz deflection frequency (fraction form):')
                    if xz  == '0':
                        xz = 0
                    else:
                        xz = float(xz.rsplit('/')[0])/float(xz.rsplit('/')[1])
                    yz = input('New yz deflection frequency (fraction form):')
                    if yz == '0':
                        yz = 0
                    else:
                        yz= float(yz.rsplit('/')[0])/float(yz.rsplit('/')[1])
                    slm.deflection_frequency = (-yz, -xz)  #todo remove the minus signal and correct the algorithm2
                    log.info(f'Deflection frequency set to: {slm.deflection_frequency}')

                elif control =='3':
                    slm.two_pi_equivalent = float(input('Set two pi equivalent (float from 0 to 1): '))
                    log.info(f'Two pi equivalent set to: {slm.two_pi_equivalent}')
                elif control == '4':
                    for _ in range(2):
                        log.info(f'Recording data set...')
                        modulate_acquire_superposition()
                elif control == '5':
                    modulate_superposition()
                    measured_images = []
                    cam.acquire()
                    img = np.mean([cam.acquire() for _ in range(nb_images_to_average)], axis=0)
                    measured_images.append(np.rot90(img))
                    fig5 = plt.figure()
                    plt.imshow(img)
                    plt.show()
                    sample_name = input('Sample name:')
                    if sample_name != '':
                        output_full_file = output_path / ('photonic_island_nb_images_to_average_' + str(nb_images_to_average) +'_'+ sample_name + '.npz')
                        log.info(f'Writing result to {output_full_file}...')
                        np.savez(output_full_file, images=measured_images)

                elif control == '6':
                    exposure = float(input('Set exposure time in ms:')) * 1e-3
                    cam.exposure_time = exposure
                    log.info(f'Camera exposure: {cam.exposure_time*1e3:0.3f}ms.')
    log.info('Done.')
