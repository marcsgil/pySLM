#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Code related to the SMILE project.

import numpy as np
import matplotlib.pyplot as plt

from optics.utils.ft import Grid
from optics.utils.display.hsv import hsv2rgb
from examples.display import log

if __name__ == '__main__':
    # Thorlabs Echelle Grating
    grating_spacing = 1e-3 / 31.6
    blaze_angle = 63 * np.pi / 180
    center_wavelength = 488e-9
    spectral_width = 5e-9
    wavelengths = Grid(center=center_wavelength, extent=spectral_width, shape=15, include_last=True)

    input_angle = -blaze_angle  #-np.pi / 2  #0.0
    incident_beam_height = 5e-3

    output_angle_list = []
    output_grid = Grid(extent=np.pi, center=0.0, shape=4096)
    output_intensity = np.zeros(output_grid.shape)
    for wavelength in wavelengths:
        output_field = np.zeros(output_grid.shape, dtype=complex)
        output_angle_list.append([])
        m_lims = (np.array([-1, 1]) + np.sin(input_angle)) * grating_spacing / wavelength
        for m in range(int(m_lims[0]), int(m_lims[-1]+1)):
            sin_output_angle = m * wavelength / grating_spacing - np.sin(input_angle)
            if np.abs(sin_output_angle) < 1:
                output_angle = np.arcsin(sin_output_angle)
                output_angle_list[-1].append(output_angle)
                if -2 <= m + 116 <= 2:
                    delta_output_angle = output_grid - output_angle
                    incident_beam_height_at_output = incident_beam_height / np.cos(input_angle) * np.cos(output_angle)
                    beta = 2 * np.pi * np.sin(delta_output_angle) * incident_beam_height_at_output / wavelength
                    output_field += np.sinc(beta / np.pi)
        output_intensity += np.abs(output_field) ** 2

    #
    # Display
    #
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex='all')

    for output_angles, wavelength in zip(output_angle_list, wavelengths):
        output_angles = np.array(output_angles)
        hue = (wavelengths[-1] - wavelength) / (wavelengths[-1] - wavelengths[0]) * 2 / 3
        axs[0].scatter(output_angles * 180 / np.pi, np.ones(output_angles.shape) * wavelength*1e9, None, hsv2rgb([[hue, 1, 1]]))
    axs[0].set(xlabel='$\\beta_{out}$  [degrees]', ylabel='$\\lambda$ [nm]',
               ylim=(center_wavelength + np.array([-1, 1]) * 50 * spectral_width) * 1e9)
    axs[0].set_title(f'Angle of incidence: {input_angle * 180 / np.pi:0.1f} degrees')

    axs[1].plot(np.array(output_grid) * 180 / np.pi, output_intensity)
    axs[0].set(xlabel='$\\beta_{out}$  [degrees]', ylabel='I [a.u.]')
    plt.show(block=True)
