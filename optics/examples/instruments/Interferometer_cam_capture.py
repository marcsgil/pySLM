import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime

from examples.instruments import log
from optics.instruments.cam.ids_cam import IDSCam
from optics.utils.ft.subpixel import register, roll, Reference, roll_ft, Registration
from optics.utils.display import complex2rgb, grid2extent
from optics.utils.ft.grid import Grid
from optics.utils import ft
from projects.confocal_interference_contrast.complex_phase_image import Interferogram, get_registration, save_registration, registration_calibration


if __name__ == '__main__':
    fig, axs = plt.subplots(2, 2)

    log.info(IDSCam.list())

    grid = None
    initial_interferograms = None
    registrations = [None, None]
    pyplot_images = None
    with IDSCam(exposure_time=20e-3, gain=0.0) as cam1, IDSCam(exposure_time=20e-3, gain=0.0) as cam2:
        output_path = pathlib.Path(__file__).parent.absolute() / 'results'
        output_path.mkdir(parents=True, exist_ok=True)

        while True:
            output_file_path = output_path / ('polarized_interference_field_' + datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3])

            log.info('Acquiring interference images...')
            interference_images = [cam.acquire() for cam in (cam1, cam2)]
            log.info('Mirroring the second image left-right.')
            interference_images[1] = interference_images[1][:, ::-1]
            if grid is None:
                grid = Grid(interference_images[0].shape[:2])   #, cam1.pixel_pitch)  # Assume both cameras have same pixel pitch
            log.info('Extracting fields from interference pattern...')
            interferograms = [Interferogram(img, reg, maximum_fringe_period=10) for img, reg in zip(interference_images, registrations)]  # exact reg
            if initial_interferograms is None:
                initial_interferograms = interferograms
                registrations = [_.registration for _ in initial_interferograms]
                init_filters = [_.__array__() / np.amax(np.abs(_)) for _ in initial_interferograms]
            log.info('Interference frequency')
            for _ in interferograms:
                log.info(_.registration.shift)
            log.info('Normalizing to the initial interference field.')
            noise_to_signal_ratio = 1 / 10
            interference_fields = [interferogram.__array__() * init_filter.conj() / (np.abs(init_filter) ** 2 + noise_to_signal_ratio ** 2)
                                   for init_filter, interferogram in zip(init_filters, interferograms)]
            interference_fields = np.array(interference_fields, dtype=np.complex64)
            log.info(f'Saving to {output_file_path}...')
            np.savez(str(output_file_path), interference_fields=interference_fields)
            log.info('Displaying...')
            if pyplot_images is None:
                pyplot_images = [[], []]
                for ax, img in zip(axs[0], interference_images):
                    pyplot_images[0].append(ax.imshow(img, extent=grid2extent(grid)))
                    ax.set(xlabel='x', ylabel='y', title='recording')
                for ax, fld in zip(axs[1], interference_fields):
                    pyplot_images[1].append(ax.imshow(complex2rgb(fld, normalization=1.0), extent=grid2extent(grid)))
                    ax.set(xlabel='x', ylabel='y', title='field')
            else:
                for im, img in zip(pyplot_images[0], interference_images):
                    im.set_data(img)
                for im, fld in zip(pyplot_images[1], interference_fields):
                    im.set_data(complex2rgb(fld, normalization=1.0))

            # for ax, img in zip(axs[1], interference_images):
            #     ax.imshow(complex2rgb(ft.fftshift(ft.fft2(img)), normalization=100.0), extent=grid2extent(grid.k.as_origin_at_center))
            #     ax.set(xlabel='k_x', ylabel='k_y', title='recording_ft')
            plt.pause(0.01)
            plt.show(block=False)
