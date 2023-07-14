import numpy as np

import matplotlib.pyplot as plt
import time
from datetime import datetime
import pathlib

from projects.transverse_structures import log

from optics.instruments.slm.meadowlark_pci_slm import MeadowlarkPCISLM
from optics.instruments.cam.ids_cam import IDSCam
from optics.utils.display import complex2rgb, grid2extent
from optics.utils import Roi

if __name__ == '__main__':
    testing = False

    center_adjustment = 0  # offset where the Meadowlark SLM splits
    nb_datasets = 1 if testing else 2
    phase_change_deg = 10 if testing else 1
    nb_images_to_average = 1 if testing else 49

    phases = np.arange(-180, 180, phase_change_deg) * np.pi / 180

    output_path = pathlib.Path(__file__).parent.absolute() / 'output'
    pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)

    with MeadowlarkPCISLM(wavelength=633e-9) as slm:
        with IDSCam(normalize=True) as cam:
            log.info(f'cam.shape={cam.shape}')
            cam.roi = Roi(shape=(501, 501), center=cam.shape//2)
            log.info(f'cam.roi={cam.roi}')
            log.info('Adjusting camera exposure and centering field-of-view...')
            cam.center_roi_around_peak_intensity(shape=(151, 201), target_graylevel_fraction=1.0)
            log.info(f'Camera exposure: {cam.exposure_time*1e3:0.3f}ms and region of interest: {cam.roi}.')

            # Display
            fig, axs = plt.subplots(1, 1, sharex='all', sharey='all')
            axs = np.atleast_1d(axs)
            ax_im = axs[0].imshow(np.zeros(cam.roi.shape), extent=grid2extent(cam.roi.grid))
            ax_im.set_clim(0, 1)
            fig.colorbar(ax_im, ax=axs[0])

            for dataset_idx in range(nb_datasets):
                log.info(f'Recording data set {dataset_idx}...')
                measured_images = []
                for phase in phases:
                    log.info(f'Measuring for phase {phase*180/np.pi:0.1f}...')

                    slm.modulate(np.exp(1j * phase * (
                            slm.roi.grid[1] - slm.roi.grid.shape[1]//2 + center_adjustment > 0))
                                 )
                    time.sleep(0.5)

                    cam.acquire()
                    img = np.mean([cam.acquire() for _ in range(nb_images_to_average)], axis=0)
                    measured_images.append(img)

                    # Display latest image
                    log.info(f'Showing $\\theta={phase*180/np.pi:0.1f}^o$, max={np.amax(img):0.3f}')
                    ax_im.set_data(measured_images[-1])
                    axs[0].set_title(f'$\\theta={phase*180/np.pi:0.1f}^o$, max={np.amax(img):0.3f}')
                    plt.show(block=False)
                    plt.pause(0.01)

                # Saving
                output_full_file = output_path / ('photonic_island_phase_images_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f%z") + '.npz')
                log.info(f'Writing result {(dataset_idx+1) / nb_datasets} to {output_full_file}...')
                np.savez(output_full_file, images=measured_images, phases=phases)

    log.info('Done.')
