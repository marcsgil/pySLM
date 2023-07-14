#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
from threading import Thread
from queue import Full
from queue import Queue
# from multiprocessing import Queue
from multiprocessing import Process
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import time
import pathlib
from datetime import datetime
import PIL

from examples.video import log
from optics.instruments.cam.simulated_cam import SimulatedCam
from optics.instruments.cam.web_cam import WebCam
# from optics.instruments.cam.ids_cam import IDSCam
# from optics.instruments.cam.ids_peak_cam import IDSPeakCam
from optics.utils import Roi


class Item:
    pass


@dataclass(order=True)
class Frame(Item):
    index: int
    data: np.ndarray


class Done(Item):
    pass


def acquisition_worker(storage_queue: Queue, display_queue: Queue):
    try:
        with SimulatedCam(shape=(1024, 2048)) as cam:
        # with WebCam() as cam:
        # with IDSPeakCam(normalize=False) as cam:
            cam.exposure_time = 100e-3
            cam.frame_time = None

            cam.roi = None  # Roi(shape=(300, 400), center=cam.shape/2)
            log.info(f"Set exposure time to {cam.exposure_time} and region-of-interest to {cam.roi}.")
            elapsed_times = []
            with cam.stream(nb_frames=1000) as image_stream:
                start_time = time.perf_counter()
                for frame_idx, img in enumerate(image_stream):
                    elapsed_times.append(time.perf_counter() - start_time)
                    log.info(f'Acquired frame {frame_idx}...')
                    item = Frame(frame_idx, img)
                    try:
                        storage_queue.put(item, block=False)
                    except Full:
                        log.warning(f'Dropping frame {item.index}!')
                    try:
                        display_queue.put(item, block=False)
                    except Full:
                        pass
                    start_time = time.perf_counter()

            log.info('Done acquiring!')
            if len(elapsed_times) > 0:
                log.info(f'Average acquisition time is {np.mean(np.asarray(elapsed_times)) * 1e3:0.3f} ms.')
    finally:
        storage_queue.put(Done())
        display_queue.put(Done())


def display_worker(in_queue: Queue):
    log.info('Started display_worker.')
    plt_image = None
    item = in_queue.get()
    while isinstance(item, Frame):
        log.info(f"Displaying frame {item.index}....")
        # img_data = item.data.astype(np.float)
        # img_data //= np.maximum(np.amax(img_data), 1)  # avoid division by zero
        # if plt_image is None:
        #     plt_image = plt.imshow(img_data)
        #     plt.colorbar()
        # else:
        #     plt_image.set_data(img_data)
        # plt.title(f"frame {item.index}")
        # plt.pause(0.001)
        # plt.draw()
        # log.info(f"Displayed frame {idx}.")
        item = in_queue.get()
    log.info('Display worker exits!')


output_path = pathlib.Path(__file__).parent.absolute() \
              / 'results' / ('test_' + datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3])


def storage_worker(in_queue: Queue):
    log.info('Started storage_worker.')
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        log.info(f'Writing output to {output_path}...')

        elapsed_times = []
        item = in_queue.get()
        while isinstance(item, Frame):
            try:
                log.info(f'Storing frame {item.index}...')
                img = PIL.Image.fromarray(item.data)
                img = img.convert('L')
                start_time = time.perf_counter()
                img.save(str(output_path / f'frame_{item.index:06.0f}.png'), format='png', compress_level=5)  # Slow but can be a 10th the size
                # img.save(str(output_path / f'frame_{item.index:06.0f}.jpg'), format='jpeg', quality=95)  # Only 8 bit, twice as slow as tiff but a quarter the size on disk
                # img.save(str(output_path / f'frame_{item.index:06.0f}.tif'), format='tiff')  # Fast, no compression
                # img.save(str(output_path / f'frame_{item.index:06.0f}.tif'), format='tiff', compression='tiff_deflate')  # Slow but strong compression
                elapsed_time = time.perf_counter() - start_time
                elapsed_times.append(elapsed_time)
            except FileNotFoundError:
                pass
            item = in_queue.get()
        log.info('Storage worker exits!')
        if len(elapsed_times) > 0:
            average_time = np.mean(np.asarray(elapsed_times))
            log.info(f'Average storage time: {average_time * 1e3:0.3f} ms.')
    finally:
        # Signal that the show is over to the other workers
        in_queue.put(Done())
        # # Signal the end of the line to downstream processors
        # out_queue.put(Done())


if __name__ == '__main__':
    nb_workers = 4
    queue_storage = Queue()
    queue_display = Queue(maxsize=1)

    # turn-on the worker thread
    acquisition_prog = Thread(target=acquisition_worker, daemon=False, args=(queue_storage, queue_display, ))
    # acquisition_prog = Process(target=acquisition_worker, args=(queue_acq_sto, queue_display, ))
    acquisition_prog.start()
    storage_progs = []
    for idx in range(nb_workers):
        storage_progs.append(Thread(target=storage_worker, daemon=False, args=(queue_storage,)))
        # storage_progs.append(Process(target=storage_worker, args=(queue_storage,)))
        storage_progs[-1].start()

    start_time = time.perf_counter()
    # The display should be on the main Thread for Tk
    display_worker(queue_display)

    # block until all tasks are done
    acquisition_prog.join()
    log.info('Acquisition completed')
    for storage_prog in storage_progs:
        storage_prog.join()
    elapsed_time = time.perf_counter() - start_time
    log.info(f'All work completed in {elapsed_time:0.3f} s.')

