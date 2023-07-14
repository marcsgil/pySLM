import numpy as np

from examples.video import log
import matplotlib.pyplot as plt
from optics.utils import Roi
from optics.instruments.cam.ids_peak_cam import IDSPeakCam


if __name__ == '__main__':
    all_cam_infos = IDSPeakCam.list(include_unavailable=True)
    for cam_info in all_cam_infos:
        log.info(f'Found IDS Peak Camera {cam_info.id}, model {cam_info.model} with serial number {cam_info.serial} ' + ('(available)' if cam_info.available else '(in use)'))

    all_available_cams = [_ for _ in all_cam_infos if _.available]

    if len(all_available_cams) > 0:
        fig, axs = plt.subplots(1, len(all_available_cams))
        axs = np.atleast_1d(axs)

        nb_trials = 2
        for trial_idx in range(nb_trials):
            log.info(f'Starting {trial_idx}/{nb_trials}...')
            cams = []
            try:
                for _ in all_available_cams:
                    cams.append(IDSPeakCam())
                for cam_idx, cam in enumerate(cams):
                    log.info(f'Camera {cam_idx} ({cam.serial}) Roi={cam.roi}, exposure={cam.exposure_time:0.6f}s, frame_time={cam.frame_time:0.6f}. gain={cam.gain:0.3f}.')
                    log.info(f'Setting gain, exposure time, and roi for camera {cam_idx}...')
                    cam.gain = 0.5
                    cam.exposure_time = 0.01
                    cam.frame_time = 0.1
                    cam.roi = Roi(11, 13, 55, 75)
                    log.info(f'Camera {cam_idx} ({cam.serial}) Roi={cam.roi}, exposure={cam.exposure_time:0.6f}s, frame_time={cam.frame_time:0.6f}. gain={cam.gain:0.3f}.')
                for _ in range(10):
                    for cam_idx, cam in enumerate(cams):
                        img = cam.acquire()
                        axs[cam_idx].cla()
                        axs[cam_idx].imshow(img)
                        axs[cam_idx].set_title(f'trial {trial_idx}, cam {cam_idx}, frame {_}')
                    plt.show(block=False)
                    plt.pause(0.01)
                log.info(f'Trial {trial_idx}/{nb_trials} completed, closing {len(cams)} cameras.')
            finally:  # Make sure to disconnect all cameras, even when an error occurs
                for cam in cams:
                    cam.disconnect()
                log.info(f'Closed {len(cams)} cameras.')

        log.info('Done! Close window to exit.')
        plt.show()
    else:
        log.info('No cameras available for testing, sorry.')
