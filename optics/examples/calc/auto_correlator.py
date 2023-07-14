#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

#
# Processes auto-correlation measurement.
# todo: This is not really an example, move to better place?
#

import numpy as np
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import re

from optics.utils import ft
from optics.utils.ft import Grid
import optics.utils.ft.subpixel as subpix
from optics.utils.video_reader import VideoReader
from examples.calc import log

if __name__ == '__main__':
    video_file_path = Path(r'20200721_v10ums_Scan central maximum to next.avi')

    output_file_path = video_file_path.with_suffix('')

    subsampling_factors = np.ones(2, dtype=np.int) * 1

    min_nb_vertical_lines = 4
    max_nb_vertical_lines = 25
    max_nb_horizontal_lines = 10

    # Determine stage translation velocity from file name
    try:
        stage_translation_velocity = float(re.search(r'_v(\d+)ums_', video_file_path.name).groups()[0]) * 1e-6  # m / s
        log.info(f'The stage translated at {stage_translation_velocity*1e6:0.1f} micrometers per second.')
    except AttributeError as exc:
        log.error(f'Filename {video_file_path.name} should contain speed in the format _v0ums.')
        raise exc

    center_wavelength = 488e-9  # approximately


    @dataclass
    class Result:
        position: float
        average: float
        factor: np.complex
        frequency: np.ndarray


    with VideoReader(video_file_path) as video_reader:
        frame_rate = video_reader.frame_rate
        log.info(f'Recorded at {frame_rate} fps.')
        dz = stage_translation_velocity / frame_rate

        fov_grid = None
        results = []
        # background_ft = None
        tau = 0.01
        for idx, frame in enumerate(video_reader):
            # if idx > 1500:
            #     break
            if np.any(subsampling_factors > 1):
                # frame = frame[::subsample[0], ::subsample[1]]
                # Round down to sub sampling interval
                frame = frame[:(frame.shape[0] // subsampling_factors[0]) * subsampling_factors[0], :(frame.shape[1] // subsampling_factors[1]) * subsampling_factors[1]]
                # Sum intensity in each block
                frame = frame.reshape((subsampling_factors[0], frame.shape[0] // subsampling_factors[0], subsampling_factors[1], frame.shape[1] // subsampling_factors[1]))
                frame = np.mean(frame.astype(np.int32), axis=(0, 2))
            # Study image in Fourier space
            frame_ft = ft.fft2(ft.ifftshift(frame).astype(np.float) / 256.0)
            # if background_ft is not None:
            #     background_ft *= 1.0 - tau
            #     background_ft += frame_ft * tau
            # else:
            #     background_ft = frame_ft

            if fov_grid is None:
                fov_grid = Grid(shape=frame.shape, extent=1)
                bandpass_filter = (np.abs(fov_grid.f[0]) < max_nb_horizontal_lines) \
                                  & (fov_grid.f[1] >= min_nb_vertical_lines) & (fov_grid.f[1] <= max_nb_vertical_lines)

            average_intensity = np.real(frame_ft.ravel()[0]) / frame_ft.size

            registered = subpix.Reference(ndim=2).register(frame_ft * bandpass_filter, precision=1/10)
            # registered = subpix.Reference(ndim=2).register((frame_ft - background_ft) * bandpass_filter, precision=1/16)
            # frequency = registered.shift
            # factor = subpix.roll(frame_ft, shift=-frequency).ravel()[0] / frame_ft.size
            # registered = subpix.Registration(shift=frequency, original_ft=frame_ft, factor=factor)

            if idx % 100 == 0:
                log.info(f'Frame {idx:5.0f}| Spatial frequency: {registered.shift} with amplitude {np.abs(registered.factor):0.6f} and phase {np.angle(registered.factor):0.3f} (signal = {average_intensity:0.6f})')

            results.append(Result(position=idx, average=average_intensity, factor=registered.factor,
                                  frequency=registered.shift))

    positions = np.array([_.position for _ in results])
    averages = np.array([_.average for _ in results])
    contrasts = np.array([np.abs(_.factor) for _ in results])
    phases = np.array([np.angle(_.factor) for _ in results])
    frequencies = np.array([_.frequency for _ in results])

    opd = 2.0 * dz * positions

    opd_from_phase_shift = np.unwrap(phases) * center_wavelength / (2 * np.pi)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(opd * 1e6, contrasts * 100, label='contrast')
    # ax.plot(delays * 1e6, phases * 100 / np.pi)
    axs[0, 0].plot(opd * 1e6, averages * 100, label='average')
    axs[0, 0].set(xlabel='z  [$\\mu$m]', ylabel='I [%]')
    axs[0, 0].legend()

    axs[1, 0].plot(opd_from_phase_shift * 1e6, contrasts * 100, 'o-', label='contrast')
    axs[1, 0].plot(opd_from_phase_shift * 1e6, averages * 100, 'o-', label='average')
    axs[1, 0].set(xlabel='opd$_z$  [$\\mu$m]', ylabel='I [%]')
    axs[1, 0].legend()

    # axs[0, 1].plot(frequencies[:, 1], frequencies[:, 0], 'o-')
    axs[0, 1].plot(opd * 1e6, np.sqrt(np.sum(np.abs(frequencies) ** 2, axis=1)), 'o-', label='|f|')
    axs[0, 1].plot(opd * 1e6, frequencies[:, 1], 'o-', label='f$_x$')
    axs[0, 1].plot(opd * 1e6, frequencies[:, 0], 'o-', label='f$_y$')
    # axs[0, 1].set(xlabel='horizontal  [periods / FOV]', ylabel='vertical  [periods / FOV]')
    axs[1, 0].set(xlabel='opd$_z$  [$\\mu$m]', ylabel='frequency [periods / FOV]')
    # axs[0, 1].set_xlim(np.array([-1, 1]) * np.amax(frequencies))
    # axs[0, 1].set_ylim(np.array([-1, 1]) * np.amax(frequencies))
    axs[0, 1].legend()

    axs[1, 1].plot(opd * 1e6, opd_from_phase_shift * 1e6, 'o-', label='opd')
    axs[1, 1].set(xlabel='z  [$\\mu$m]', ylabel='z [um]')
    axs[1, 1].legend()

    log.info(f'Saving figure to {output_file_path}...')
    fig.savefig(output_file_path.with_suffix('.png'), dpi=300, transparent=False)
    fig.savefig(output_file_path.with_suffix('.svg'))

    plt.show(block=True)

