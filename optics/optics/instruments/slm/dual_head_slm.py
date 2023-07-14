import numpy as np
from typing import Union
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

import logging
log = logging.getLogger(__name__)
from optics.instruments.slm import DisplaySLM

__all__ = ['DualHeadSLM']


class DualHeadSLM(DisplaySLM):
    def __init__(self, display_target: Union[None, int, Axes, AxesImage] = None, deflection_frequency=(0, 0),
                 two_pi_equivalent: float = 1.0, pixel_pitch=None, shape=None):
        super().__init__(display_target, deflection_frequency=deflection_frequency, two_pi_equivalent=two_pi_equivalent,
                         pixel_pitch=pixel_pitch, shape=shape)

        log.info(f'PhaseSLM initialized with display shape {self.shape}, deflection_frequency {self.deflection_frequency} and 2 pi-equivalent {self.two_pi_equivalent * 100}%.')

    def _modulate(self, phase: np.ndarray, amplitude: Union[float, np.ndarray]):
        """
        Low level modulation of the device. This method is called from the superclass SLM.modulate() and
        SLM.complex_field. The DualDisplaySLM modulates the phase one the green color channel and the amplitude on the
        blue color channel.

        :param phase: A real 2d-array with the phase with range [-pi, pi) mod 2pi.
        :param amplitude: A real 2d-array with the amplitude with range [0, 1]
        """
        with self._lock:
            quantized_phase = self._quantize_phase(phase)  # convert float -> np.uint8
            amplitude_for_slm = np.array(amplitude * 255 + 0.5, dtype=np.uint8)  # 255 to avoid wrap around, 0.5 for rounding
            # Use green = phase, blue = amplitude
            image_for_slm = np.stack(
                (quantized_phase,  # The red channel is ignored
                 quantized_phase,
                 amplitude_for_slm
                 ), axis=-1
            )

            self.image_on_slm = image_for_slm  # Show on the display
