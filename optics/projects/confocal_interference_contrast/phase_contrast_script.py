import numpy as np
import matplotlib.pyplot as plt
import json
import jsonpickle

from projects.confocal_interference_contrast import log
from optics.utils.display import complex2rgb, grid2extent
from optics.utils.ft.grid import Grid
from optics.utils.ft.subpixel import register, roll, Reference, roll_ft, Registration
from optics.utils import ft
from optics.instruments.cam.ids_cam import IDSCam
from projects.confocal_interference_contrast.complex_phase_image import Interferogram, get_registration, save_registration, registration_calibration


exposure_time = 50e-3


def display_video(exposure_time):
    with IDSCam(serial=4103697390, exposure_time=exposure_time, normalize=True) as cam:  # [4103697390, 4103198121]
        grid = Grid(shape=cam.shape, step=cam.pixel_pitch)

        fig, axs = plt.subplots(2, 2)
        im = np.empty((2, 2), dtype=object)
        im[0, 0] = axs[0, 0].imshow(np.zeros(grid.shape), extent=grid2extent(grid) / 1e-6)
        axs[0, 0].set(xlabel=r'x [$\mu m$]', ylabel=r'y [$\mu m$]', title='recorded')
        im[1, 0] = axs[1, 0].imshow(np.zeros(grid.shape), extent=grid2extent(grid.f.as_origin_at_center) * 1e-3)
        axs[1, 0].set(xlabel='$f_x$ [cycles/mm]', ylabel='$f_y$ [cycles/mm]', title='spectrum')
        im[0, 1] = axs[0, 1].imshow(np.zeros(grid.shape), extent=grid2extent(grid) / 1e-6)
        axs[0, 1].set(xlabel=r'x [$\mu m$]', ylabel=r'y [$\mu m]$', title='complex')
        im[1, 1] = axs[1, 1].imshow(np.zeros(grid.shape), extent=grid2extent(grid) / 1e-6)
        axs[1, 1].set(xlabel=r'x [$\mu m$]', ylabel=r'y [$\mu m]$', title='phase only')

        try:
            # registration = get_registration()
            registration = registration_calibration(cam.acquire()) #, registration_guess=get_registration())
            for idx in range(1000):
                interference_image = cam.acquire()
                interferogram = Interferogram(interference_image, registration)  # exact reg
                # interferogram = Interferogram(interference_image, approximate_registration=registration)  # approx reg
                # registration = interferogram.registration
                complex_phase_image = np.array(interferogram)

                # if idx == 0 and True:  # corrects system aberrations, assuming that the final wavefront should be planar
                #     aberration_reference_phase = np.exp(1j * np.angle(complex_phase_image))
                #     np.save('aberration_reference.npy', aberration_reference_phase)
                # else:
                #     aberration_reference_phase = np.load('aberration_reference.npy')
                # complex_phase_image /= aberration_reference_phase

                log.info(f'Updating registration to {registration} with a frequency shift of {registration.shift}.')
                # save_registration(registration)

                # Display
                im[0, 0].set_data(np.repeat(interference_image[:, :, np.newaxis], axis=2, repeats=3))
                im[1, 0].set_data(complex2rgb(ft.fftshift(ft.fft2(complex_phase_image)), 10))
                im[0, 1].set_data(complex2rgb(complex_phase_image, 1))
                im[1, 1].set_data(complex2rgb(np.exp(1j * np.angle(complex_phase_image)), 1))

                plt.pause(1)
                plt.show(block=False)
        finally:
            if False:
                save_registration(registration)


if __name__ == "__main__":
    display_video(exposure_time)


