import time
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from pathlib import Path

from projects.adaptive_optics import log
from optics.utils import ft
from optics.utils.display import complex2rgb, grid2extent

from projects.adaptive_optics.imaging_system import ImagingSystem

from optics.instruments.dm.alpao_dm import AlpaoDM
from optics.calc import zernike
from optics.instruments.cam.ids_cam import IDSCam


if __name__ == '__main__':
    imaging_system = ImagingSystem(ft.Grid(np.full(2, 128)), simulate_noise=False, point_source=False)

    fig, axs = plt.subplots(3, 3, sharex='all', sharey='all')
    test_coefficient_values = np.array([0, 1.0, 5.0])
    for ax_row, tilt in zip(axs, test_coefficient_values):
        for ax, tip in zip(ax_row, test_coefficient_values):
            img = imaging_system.image_from_zernike_coefficients([0.0, tilt, tip], include_system_aberration=False)
            ax.imshow(img)
            ax.set_title(f'tilt={tilt:0.3f}, tip={tip:0.3f}')
    plt.show(block=False)

    fig, axs = plt.subplots(3, 3, sharex='all', sharey='all')
    test_coefficient_values = np.array([0, 0.1, 0.5])
    with IDSCam(index=1, normalize=True, exposure_time=10e-3, gain=1, black_level=110) as cam:
        with AlpaoDM(serial_name='BAX240') as dm:
            for ax_row, tilt in zip(axs, test_coefficient_values):
                for ax, tip in zip(ax_row, test_coefficient_values):
                    aberration = zernike.Polynomial([0.0, tilt, tip]).cartesian
                    log.info(f'Modulating {aberration}...')
                    dm.modulate(lambda u, v: aberration(v, -u))
                    time.sleep(0.1)
                    img = cam.acquire()
                    ax.imshow(img)
                    ax.set_title(f'tilt={tilt:0.3f}, tip={tip:0.3f}')

    log.info('Done!')
    plt.show()

