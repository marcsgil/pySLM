import numpy as np
from typing import Union, Sequence, Optional
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

import logging
log = logging.getLogger(__name__)

from optics.instruments.slm import DisplaySLM
from optics.utils.ft import Grid


__all__ = ['PhaseSLM', 'encode_amplitude_as_phase']


def encode_amplitude_as_phase(amplitude: Union[None, float, Sequence, np.ndarray] = 1,
                              grid: Optional[Grid] = None,
                              deflection_frequency: Union[None, Sequence, np.ndarray] = (0, 0)) -> np.ndarray:
    """
    Calculate the required phase modulation to emulate amplitude variation.

    :param amplitude: A real 2d-array with the amplitude with range [0, 1)
    :param grid: The pixel grid of the spatial light modulator. Divide this by an integer to bin pixels.
    :param deflection_frequency: The deflection frequency of the SLM. When specified, the alternating pattern is chosen
        to be approximately orthogonal to the frequency: a checkerboard for approximately diagonal deflections and a
        horizontal or vertical zebra pattern for approximately vertical or horizontal deflections, respectively.
        If not specified, a 2d checkerboard pattern is used.
    :return: A 2d array with phase deviation values.
    """
    if grid is None:
        grid = Grid(np.asarray(amplitude).shape)

    # Modulate the amplitude with the phase SLM
    phase_deviation_for_amplitude = np.arccos(amplitude)  # this is still the absolute value  [pi/2 -> 0]

    # Select pixels for binary coding  # todo: consider pixel binning to reduce cross-talk on some SLMs
    if not np.all(deflection_frequency[:2] == 0):
        # deflection_angle = np.mod(
        #     np.arctan2(deflection_frequency[1], deflection_frequency[0]) + np.pi / 4.0,
        #     np.pi / 2.0
        # ) - np.pi / 4.0
        positive_pixels = np.mod(grid[0] + grid[1], 2) > 0
        # if np.abs(deflection_angle) > np.pi / 8.0:
        #     # Dump intensity in the highest diagonal orders
        #     selected_pixels = np.mod(grid[0] + grid[1], 2) > 0
        # else:
        #     # Don't dump intensity diagonally because the first order
        #     # deflection_frequency is too close to the diagonal
        #     if np.abs(deflection_frequency[1]) > np.abs(deflection_frequency[0]):
        #         selected_pixels = np.mod(grid[0], 2) > 0
        #     else:
        #         selected_pixels = np.mod(grid[1], 2) > 0
    else:
        # Or, do zeroth order modulation
        positive_pixels = np.mod(grid[0] + grid[1], 2) > 0  # dump intensity at highest orders

    # Convert the absolute value to a signed value.
    # Half of the pixels should be biased upwards and the other half downwards
    phase_deviation_for_amplitude *= (2.0 * positive_pixels - 1.0)

    return phase_deviation_for_amplitude


class PhaseSLM(DisplaySLM):
    def __init__(self, display_target: Union[None, int, Axes, AxesImage] = None, deflection_frequency=(0, 0),
                 two_pi_equivalent: float = 1.0, pixel_pitch=None, shape=None):
        super().__init__(display_target, deflection_frequency=deflection_frequency, two_pi_equivalent=two_pi_equivalent,
                         pixel_pitch=pixel_pitch, shape=shape)

        log.info(f"PhaseSLM initialized with display shape {self.shape}, " +
                 f"deflection_frequency {self.deflection_frequency} and " +
                 f"2 pi-equivalent {self.two_pi_equivalent * 100:0.3f}%.")

    def _modulate(self, phase: np.ndarray, amplitude: Union[float, np.ndarray]):
        """
        Low level modulation of the device. This method is called from the superclass SLM.modulate() and
        SLM.complex_field. The PhaseSLM modulates the phase to emulate amplitude variation in 0th or 1st order.

        :param phase: A real 2d-array with the phase with range [-pi, pi) mod 2pi.
        :param amplitude: A real 2d-array with the amplitude with range [0, 1]
        """
        with self._lock:
            phase += encode_amplitude_as_phase(amplitude, self.grid, self.deflection_frequency)  # encode the amplitude as phase
            quantized_phase = self._quantize_phase(phase)  # convert float -> np.uint8

            # Use red and green for maximum compatibility
            # Set blue channel to 255 so that this code can LAO be used on a dual head setup
            image_for_slm = np.stack((quantized_phase,  # red
                                      quantized_phase,  # green
                                      np.full(quantized_phase.shape, 255, dtype=np.uint8)  # blue
                                      ), axis=-1
            )

            self.image_on_slm = image_for_slm  # Show on the display

