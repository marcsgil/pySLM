#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

# import optics.gui.lightsheet as ls
#
# ls.start()

import numpy as np
import time

from optics import log
from optics.instruments.cam.simulated_cam import SimulatedCam
from optics.instruments.source.laser import SimulatedLaser
from optics.instruments.stage import SimulatedStage
from optics.instruments.cam.ids_cam import IDSCam
from optics.instruments.source.toptica import IBeamSMARTLaser
from optics.instruments.stage.thorlabs_kinesis import ThorlabsKinesisStage
from optics.utils.ft import Grid


grid = Grid(extent=200e-6, step=5e-6, center=10.0e-3)


# with SimulatedLaser(wavelength=488e-9, max_power=100e-3) as laser, SimulatedCam() as cam,\
#         SimulatedStage(ranges=[[0, 25e-3]], ndim=1, max_velocity=2e-3, max_acceleration=1e-3) as stage:
with SimulatedLaser(wavelength=488e-9, max_power=100e-3) as laser, IDSCam() as cam, \
        ThorlabsKinesisStage() as stage:
    laser.emitting = False
    stage.position = grid[0]
    laser.power = 1e-3
    cam.frame_time = None
    cam.exposure_time = 20e-3
    cam.normalize = False
    # img = cam.acquire()
    # log.info('ACQUIRED')
    # Start recording
    with cam.stream(nb_frames=grid.shape[0]) as image_stream:
        start_time = time.perf_counter()
        for idx, img in enumerate(image_stream):
            log.info(f'Image index {idx}: max value = {np.amax(img)}')
        total_time = time.perf_counter() - start_time
        log.info(f'{total_time} s / {len(image_stream)} = {total_time / len(image_stream)}s (should be {cam.frame_time}s)')
    with cam.stream(nb_frames=10) as image_stream:
        start_time = time.perf_counter()
        for idx, img in enumerate(image_stream):
            log.info(f'Image index {idx}: max value = {np.amax(img)}')
        total_time = time.perf_counter() - start_time
        log.info(f'{total_time} s / {len(image_stream)} = {total_time / len(image_stream)}s (should be {cam.frame_time}s)')
    exit()
    # Start run
    target_velocity = grid.step / cam.frame_time
    with stage.translate(velocity=target_velocity, origin=grid[0], destination=grid[-1]) as stage_translation,\
            cam.stream(nb_frames=grid.shape[0]) as image_stream:
        laser.emitting = True
        times = []
        positions = []
        for idx, img in enumerate(image_stream):
            log.info(f'velocity to {target_velocity * 1e3} mm/s, it is instead {stage_translation.velocity * 1e3} mm/s.')
            if not np.isclose(stage_translation.velocity, target_velocity):
                log.warning(f'Could not set velocity to {target_velocity[0] * 1e3:0.3f} mm/s, it is instead {stage_translation.velocity[0] * 1e3:0.3f} mm/s.')
            img = cam.acquire()
            pos = stage_translation.position
            times.append(time.perf_counter())
            positions.append(pos)
            log.info(f'index = {idx}: z = {pos[0]*1e3:0.3f} mm: std = {np.std(img)}')
            log.info(f'Current position: {pos[0]*1e3:0.3f} mm')
    # Run finished
    laser.emitting = False
    stage.velocity = 0.0
    stage.position = grid[grid.shape[0] // 2]
    log.info(stage)
    log.info(cam)
    log.info(laser)

log.info('Done recording.')
