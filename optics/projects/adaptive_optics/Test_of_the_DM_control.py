import torch
from optics.instruments.dm.alpao_dm import AlpaoDM
from optics.calc import zernike
from optics import log
import numpy as np
import time
from datetime import datetime, timezone
from pathlib import Path
from matplotlib import pyplot as plt
from optics.instruments.cam.ids_cam import IDSCam
from projects.adaptive_optics import log


zernike_orders = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
scales = np.linspace(-1, 1, num=21)

# # Temporary check of what the unit tilt actually does
# u = np.arange(-1, 1, 0.01)
# plt.plot(u, zernike.BasisPolynomial(1).cartesian(u, 0.0))
# plt.xlabel('u')
# plt.ylabel('tilt(u)')
# plt.title('The unit tilt actually goes from -2 to +2.')  # Yesterday's plot was computed with a different x-axis then the one used for plotting.
# plt.show()

length_orders = len(zernike_orders)
length_scales = len(scales)
row = 0
if __name__ == '__main__':
    with IDSCam(index=1, normalize=True, exposure_time=10e-3, gain=1, black_level=110) as cam:
        with AlpaoDM(serial_name='BAX240') as dm:  # todo: Try changing gain_factor from 1/16 to something higher
            dm.wavelength = 500e-9

            def modulate_and_capture(_order, _scale) -> float:
                aberration = zernike.BasisPolynomial(_order).cartesian  # defocus, j_index = 4
                dm.modulate(lambda u, v: aberration(u, v) * _scale)
                # actuator_positions = dm.actuator_position
                log.info(f'Displaying {aberration}...')
                time.sleep(0.1)
                img = cam.acquire()
                return img  # pick the maximum (might be noisy)

            for order in zernike_orders:
                folder_path = Path.home() / Path(f'E:/Adaptive Optics/Experimental MLAO/Characterization of DM/From for loops/Zernike order {order}')
                for scale in scales:
                    blured_image = modulate_and_capture(order, scale)
                    fig_M, axs_M = plt.subplots(1, 1, sharex='all', sharey='all', figsize=(18, 8))
                    im0 = axs_M.imshow(blured_image)
                    axs_M.set(title=f"image blured by Zernike order:{order} at scale:{scale}")
                    # log.info(f'Results will be saved in {figure_path}.')
                    folder_path.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-3]
                    full_file_name_figures = folder_path / f'Time_{timestamp}_Zernke_order_{order}_scale_{scale}.png'
                    plt.savefig(full_file_name_figures)
                    plt.close()
