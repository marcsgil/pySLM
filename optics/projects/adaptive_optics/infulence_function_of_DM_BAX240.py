import torch
from optics.instruments.dm.alpao_dm import AlpaoDM
from optics.calc import zernike
from os.path import dirname, join as pjoin
import scipy.io as sio
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


scales = np.linspace(-1, 1, num=41)
actuator_num = 69

row = 0
loop_num = 1

# data_dir = pjoin(dirname('E:/Expt/DM/Original USB/Config/'))
# matname = pjoin(data_dir, 'BAX240-Z2C.mat')
# mat_contents = sio.loadmat(matname)
# Z2CMatrix = mat_contents['Z2C']

if __name__ == '__main__':
    with IDSCam(index=1, normalize=True, exposure_time=5e-3, gain=1, black_level=0) as cam:
        with AlpaoDM(serial_name='BAX240') as dm:  # todo: Try changing gain_factor from 1/16 to something higher
            dm.wavelength = 500e-9

            def modulate_and_capture(_order, _scale) -> float:
                mirror_command = np.zeros([1, 69])  # defocus, j_index = 4
                mirror_command[0, _order] = _scale
                dm.modulate(mirror_command)
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
                for actuator_index in range(actuator_num):
                    folder_path = Path.home() / Path(f'E:/Adaptive Optics/Experimental MLAO/Interference_influence_function/Loop {loop}/actuator_index {actuator_index}/')
                    for scale in scales:
                        inter_fringe = modulate_and_capture(actuator_index, scale)

                        # fig_z, axs_z = plt.subplots(1, 1, sharex='all', sharey='all', figsize=(18, 8))
                        # imz = axs_z.imshow(inter_fringe)
                        # axs_z.set(title=f" Actuator Index {actuator_index} scale {scale}")
                        # # fig_z.colorbar(imz)
                        # plt.show()

                        # Save the Numpy array as Image
                        folder_path.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-3]
                        full_file_name_figures = folder_path / f'Time_{timestamp}_actuator_index_{actuator_index+1}_scale_{scale}.bmp'
                        save_float_array_as_bmp(inter_fringe, full_file_name_figures)
