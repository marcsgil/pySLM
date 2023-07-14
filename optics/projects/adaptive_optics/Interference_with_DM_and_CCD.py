import torch
from optics.instruments.dm.alpao_dm import AlpaoDM
from optics.calc import zernike
from optics import log
import numpy as np
import time
import cv2
from PIL import Image

from datetime import datetime, timezone
from pathlib import Path
from matplotlib import pyplot as plt
from optics.instruments.cam.ids_cam import IDSCam
from projects.adaptive_optics import log


zernike_orders = np.linspace(0, 5, num=6)
zernike_orders = zernike_orders.astype(int)
scales = np.linspace(-5, 5, num=41)

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
loop_num = 1

if __name__ == '__main__':
    with IDSCam(index=1, normalize=True, exposure_time=8e-3, gain=1, black_level=110) as cam:
        with AlpaoDM(serial_name='BAX240') as dm:  # todo: Try changing gain_factor from 1/16 to something higher
            dm.wavelength = 500e-9

            def modulate_and_capture(_order, _scale) -> float:
                aberration = zernike.BasisPolynomial(_order).cartesian  # defocus, j_index = 4
                dm.modulate(lambda u, v: aberration(u, v) * _scale)
                # actuator_positions = dm.actuator_position
                # log.info(f'Displaying {aberration}...')
                imgf = cam.acquire()
                time.sleep(0.1)
                return imgf  # pick the maximum (might be noisy)

            def save_float_array_as_bmp(float_array, file_path):
                float_array_normalized = (float_array - np.min(float_array)) / (np.max(float_array) - np.min(float_array))
                float_array_normalized = (float_array_normalized * 255).astype(np.uint8)
                # Create a PIL Image from the normalized float array
                image = Image.fromarray(float_array_normalized, mode='L')
                # Save the image as BMP
                image.save(file_path)

            for loop in range(loop_num):
                for order in zernike_orders:
                    folder_path = Path.home() / Path(f'E:/Adaptive Optics/Experimental MLAO/Interference_profile_to_command_4fsystem_v2/Loop {loop}/Zernike order {order}')
                    for scale in scales:
                        inter_fringe = modulate_and_capture(order, scale)
                        # Save the Numpy array as Image
                        folder_path.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-3]
                        full_file_name_figures = folder_path / f'Time_{timestamp}_Zernke_order_{order}_scale_{scale}.bmp'
                        save_float_array_as_bmp(inter_fringe, full_file_name_figures)
