from optics.instruments.dm.alpao_dm import AlpaoDM
from optics.calc import zernike
from optics import log
import numpy as np
import time

from matplotlib import pyplot as plt
from optics.instruments.cam.ids_cam import IDSCam
from projects.adaptive_optics import log
from optics.calc.interferogram import Interferogram
from optics.utils.display import complex2rgb


row = 0
loop_num = 10

if __name__ == '__main__':

    with IDSCam(index=1, normalize=True, exposure_time=10e-3, gain=1, black_level=110) as cam:
        fig, ax = plt.subplots(1, 1)
        complex_im = ax.imshow(np.zeros(cam.shape))  #, extent=grid2extent(grid))

        interferogram_registration = None

        for _ in range(100):
            img = cam.acquire()
            interferogram = Interferogram(img, registration=interferogram_registration)
            interferogram_registration = interferogram.registration  # So that the next interferograms use the same shift
            # Show
            complex_im.set_data(complex2rgb(interferogram, 1))
            ax.set_title(_)
            plt.pause(0.01)
            plt.show(block=False)

    log.info('Done!')
    plt.show()
