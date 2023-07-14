import tkinter as tk
import numpy as np
import scipy.ndimage
from threading import Thread
import queue

from examples.video import log
from optics.gui import Window, SwitchButton, DragZoomCanvas
# from optics.instruments.cam.web_cam import WebCam
from optics.instruments.cam.ids_cam import IDSCam

if __name__ == '__main__':
    window = Window(title='Video Filter')  # , roi=Roi(shape=(600, 800))
    background_button = SwitchButton(window, text=('Background', 'Recording Background...'))
    background_button.pack(fill=tk.BOTH, side=tk.TOP)
    canvas = DragZoomCanvas(window).pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    def camera_prog(image_queue: queue.Queue, switch_queue: queue.Queue):
        with IDSCam() as cam:
            cam.exposure_time = 40e-3

            average_image = 0.0
            nb_images_averaged = 0
            background_image = 0.0
            for img in cam.stream(nb_frames=np.inf):
                img = img[:, ::-1]  # mirror
                if background_button.switch_state:
                    nb_images_averaged += 1
                    average_image = ((nb_images_averaged - 1) / nb_images_averaged) * average_image \
                                    + (1 / nb_images_averaged) * img
                    if nb_images_averaged > 25:
                        switch_queue.put(False)
                        log.info('Background recorded.')
                        nb_images_averaged = 0
                    background_image = average_image
                else:
                    average_image = 0.0
                try:
                    if not np.isscalar(background_image) and nb_images_averaged == 0:
                        # saturated = np.any((img <= 0) | (img >= (244.0/245)), axis=2)
                        filtered_img = img - scipy.ndimage.gaussian_filter(img, sigma=3.0)
                        filtered_background = background_image - scipy.ndimage.gaussian_filter(background_image,
                                                                                               sigma=3.0)
                        # filtered_img = scipy.ndimage.sobel(img)
                        # filtered_background = scipy.ndimage.sobel(background_image)
                        difference = np.sqrt(np.mean(np.abs(filtered_img - filtered_background)**2, axis=2))
                        # contrast = np.sqrt(np.mean(np.abs(filtered_img)**2, axis=2) +
                        #                    np.mean(np.abs(filtered_background)**2, axis=2))
                        large_difference = difference > np.quantile(filtered_background, 0.995)
                        large_difference = scipy.ndimage.binary_dilation(large_difference, iterations=10)
                        large_difference = scipy.ndimage.binary_fill_holes(large_difference)
                        large_difference = scipy.ndimage.binary_erosion(large_difference, iterations=10)
                        # high_contrast = contrast > np.quantile(contrast, 0.75)
                        # high_contrast = scipy.ndimage.binary_dilation(high_contrast, iterations=10)
                        # high_contrast = scipy.ndimage.binary_fill_holes(high_contrast)
                        # high_contrast = scipy.ndimage.binary_erosion(high_contrast, iterations=10)

                        mask = large_difference  # & high_contrast
                        # img[np.logical_not(great_difference)] = np.array([1, 0, 0])[np.newaxis, np.newaxis, :]
                        # img[np.logical_not(high_contrast)] = np.array([0, 0, 1])[np.newaxis, np.newaxis, :]
                        img[np.logical_not(mask)] = np.array([0.5, 0.5, 1])[np.newaxis, np.newaxis, :]  # blue background
                    image_queue.put_nowait(img / np.maximum(np.amax(img), 1e-6))
                except queue.Full:
                    pass

    image_queue = queue.Queue(maxsize=2)
    switch_queue = queue.Queue()
    cam_thread = Thread(target=camera_prog, daemon=True, args=(image_queue, switch_queue))
    cam_thread.start()

    def gui_update():
        if not switch_queue.empty():
            background_button.switch_state = switch_queue.get()
        canvas.image = image_queue.get()

    Window.run(action=gui_update)
