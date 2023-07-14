#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Demonstration of the Stage class.
#

import numpy as np
import time
import matplotlib.pyplot as plt

from examples.instruments import log
from optics.instruments.stage import SimulatedStage
# from optics.instruments.stage.thorlabs_kinesis import ThorlabsKinesisStage


with SimulatedStage() as stage:
    log.info(f'Using {stage}.')

    log.info(f'Current position: {stage.position*1e3} mm.')

    log.info('Scanning FOV...')
    dz = 5.0e-6
    dt = 5e-3
    Dz = 5.0e-3
    v = dz / dt
    origin = 5e-3
    destination = origin+Dz
    times = []
    positions_interp = []
    positions_poll = []
    with stage.translate(origin=origin, destination=destination, velocity=v) as t:
        while t.in_progress:
            pos_poll = stage.position[0]
            # pos = pos[0]
            pos = t.position[0]
            # print(f'At {pos*1e3} mm')
            previous_time = time.perf_counter()
            times.append(previous_time)
            positions_interp.append(pos)
            positions_poll.append(pos_poll)

            if len(positions_interp) > 1000:
                print('break!')
                break

            elapsed_time = time.perf_counter() - previous_time
            remaining_time = 5e-3 - elapsed_time
            if remaining_time > 0:
                time.sleep(remaining_time)

    # Convert to numpy arrays
    times = np.array(times)
    positions_interp = np.array(positions_interp)
    positions_poll = np.array(positions_poll)

    log.info('Plotting results...')
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(times, positions_interp)
    axs[0].plot(times, positions_poll)
    axs[0].plot(times, positions_poll - positions_interp)
    axs[1].plot(times[:-1] + 0.5 * np.diff(times[:2]), np.diff(positions_interp) / np.diff(times))
    axs[1].plot(times[:-1] + 0.5 * np.diff(times[:2]), np.diff(positions_poll) / np.diff(times))
    axs[1].plot(times[:-1] + 0.5 * np.diff(times[:2]), np.diff((positions_poll - positions_interp)) / np.diff(times))
    log.info('Done.')
    plt.show(block=True)

