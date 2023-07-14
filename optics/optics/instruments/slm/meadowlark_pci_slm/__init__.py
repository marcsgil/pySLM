import numpy as np
from typing import Union, Sequence

import os
import ctypes
import pathlib
import traceback
import logging

from optics.instruments.slm import SLM
from optics.instruments.slm.phase_slm import encode_amplitude_as_phase

log = logging.getLogger(__name__)


class MeadowlarkPCISLM(SLM):
    def __init__(self, board_number: int = 1, lut_filepath: [str, pathlib.Path] = None,
                 wavelength: float = 633e-9, deflection_frequency=(0, 0),
                 two_pi_equivalent: float = 1.0, pixel_pitch=None, shape=None):
        """
        Initializes a Meadowlark SLM with PCI board.

        :param board_number: (optional) board number, default 1.
        :param lut_filepath: (optional) .lut file, as a string or pathlib.Path object to indicate where the look-up-table is for the wavelength of interest.
        :param wavelength: (optional) The wavelength in meters that indicates what look-up-table to use.
        This should be a file in instruments/slm/meadowlar_pci_slm/luts/ with format slm4937_at***.lut, where *** is the wavelength in nanometers.
        :param deflection_frequency: (optional) The spatial frequency of the first order grating in the horizontal and vertical direction.
        :param two_pi_equivalent:  (optional) The fraction of the dynamic range to use for one wave (2pi) modulation.
        :param pixel_pitch: (optional) The size of the pixels for later display or calculations.
        :param shape: (optional) The height and width in pixels to use.
        """
        self.__board_number = ctypes.c_uint(board_number)
        self.__wavelength = wavelength

        self.serial_number = 4937

        dll_path = pathlib.Path(__file__).parent.absolute() / 'lib'
        log.info(f'Adding {dll_path.as_posix()} to dll search path...')
        os.add_dll_directory(dll_path.as_posix())

        log.info('Loading Blink_C_wrapper...')
        ctypes.cdll.LoadLibrary('Blink_C_wrapper')
        self.__slm_lib = ctypes.CDLL('Blink_C_wrapper')
        # Don't do anything here, because if it crashes, the library will not be unloaded
        try:
            # Basic parameters for calling Create_SDK
            bit_depth = ctypes.c_uint(12)
            num_boards_found = ctypes.c_uint(0)
            constructed_okay = ctypes.c_uint(-1)
            is_nematic_type = ctypes.c_bool(1)
            ram_write_enable = ctypes.c_bool(1)
            use_gpu = ctypes.c_bool(1)
            max_transients = ctypes.c_uint(20)

            # Call the Create_SDK constructor
            # Returns a handle that's passed to subsequent SDK calls
            self.__slm_lib.Create_SDK(bit_depth, ctypes.byref(num_boards_found), ctypes.byref(constructed_okay),
                                      is_nematic_type, ram_write_enable, use_gpu, max_transients, 0)

            if constructed_okay.value == 0:
                log.info('Blink SDK did not construct successfully')
                # Python ctypes assumes the return value is always int
                # We need to tell it the return type by setting restype
                self.__slm_lib.Get_last_error_message.restype = ctypes.c_char_p
                log.info(self.__slm_lib.Get_last_error_message())

            if num_boards_found.value >= 1:
                log.info('Blink SDK was successfully constructed')
                log.info(f'Found {num_boards_found.value} SLM controller(s)')
                slm_height = ctypes.c_uint(self.__slm_lib.Get_image_height(self.__board_number)).value
                slm_width = ctypes.c_uint(self.__slm_lib.Get_image_width(self.__board_number)).value
                slm_shape = np.array([slm_height, slm_width])
                if shape is None:
                    shape = slm_shape
                else:
                    shape = np.min(shape, slm_shape)

                if lut_filepath is None:
                    lut_filepath = pathlib.Path(__file__).parent.absolute() / f'luts/slm{self.serial_number}_at{self.wavelength / 1e-9:0.0f}.lut'
                if not isinstance(lut_filepath, pathlib.Path):
                    lut_filepath = pathlib.Path(lut_filepath).absolute()
                lut_filepath = lut_filepath.as_posix().encode('utf-8')
                log.info(f'Loading lookup table {lut_filepath}...')  # convert to bytes using encode!
                self.__slm_lib.Load_LUT_file(self.__board_number, lut_filepath)
            else:
                raise ValueError('No Meadowlark PCI boards found. Is the board plugged into a regular PCI slot and not a video specific one?')
        except Exception as e:
            log.error('An unexpected exception occurred:')
            log.error(traceback.format_exc())

        super().__init__(shape=shape, deflection_frequency=deflection_frequency,
                         two_pi_equivalent=two_pi_equivalent, pixel_pitch=pixel_pitch)

        # Prepare a buffer the size of the whole SLM with C-ordering.
        self.__hardware_image_on_slm = np.zeros(self.shape, dtype=np.uint8, order='C')

        log.info(f'MeadowlarkPCISLM initialized for board {self.board_number} and maximum shape {self.shape}, ' +
                 f'deflection_frequency {self.deflection_frequency} and ' +
                 f'2 pi-equivalent {self.two_pi_equivalent * 100:0.3f}%.')

    @property
    def board_number(self) -> int:
        return self.__board_number.value

    @property
    def wavelength(self) -> float:
        return self.__wavelength

    @property
    def image_on_slm(self):
        """
        The 2D numpy.ndarray of uint8 values that is sent to the display.
        """
        return self.__hardware_image_on_slm[slice(self.roi.top, self.roi.bottom), slice(self.roi.left, self.roi.right)]

    @image_on_slm.setter
    def image_on_slm(self, new_image_on_slm):
        """
        The 2D numpy.ndarray of uint8 values that is sent to the display.

        :param new_image_on_slm: The target array, without correction or 1st order deflection.
        """
        if callable(new_image_on_slm):
            new_image_on_slm = new_image_on_slm(*self.grid)
        new_image_on_slm = np.atleast_1d(new_image_on_slm)
        if new_image_on_slm.ndim < 2 or np.any(np.asarray(new_image_on_slm.shape[:2]) < self.roi.shape):
            new_image_on_slm = np.broadcast_to(new_image_on_slm, self.roi.shape)
        with self._lock:
            self.__hardware_image_on_slm[slice(self.roi.top, self.roi.bottom), slice(self.roi.left, self.roi.right)] = new_image_on_slm.astype(self.__hardware_image_on_slm.dtype)

    def _modulate(self, phase: Union[float, Sequence, np.ndarray] = 0, amplitude: Union[None, float, Sequence, np.ndarray] = 1):
        """
        This method is called from the superclass SLM modulate method.

        :param phase: A real 2d-array with the phase with range [-pi, pi) mod 2pi.
        :param amplitude: A real 2d-array with the amplitude with range [0, 1)
        :return: An integer (uint8) array with the (RGB) image as sent to the SLM.
        """
        # Settings
        wait_for_trigger = ctypes.c_uint(0)
        timeout_ms = ctypes.c_uint(5000)
        # Both pulse options can be false, but only one can be true. You either generate a pulse when the new image begins loading to the SLM
        # or every 1.184 ms on SLM refresh boundaries, or if both are false no output pulse is generated.
        output_pulse_image_flip = ctypes.c_uint(0)
        output_pulse_image_refresh = ctypes.c_uint(0)  #only supported on 1920x1152, FW rev 1.8.

        with self._lock:
            phase += encode_amplitude_as_phase(amplitude, self.grid, self.deflection_frequency)  # encode the amplitude as phase
            quantized_phase = self._quantize_phase(phase)  # convert float -> np.uint8

            # Push data to Meadowlark PCI SLM
            self.__hardware_image_on_slm[slice(self.roi.top, self.roi.bottom), slice(self.roi.left, self.roi.right)] = quantized_phase.astype(np.uint8)  # Copy data in new order
            self.__slm_lib.Write_image(self.__board_number, self.__hardware_image_on_slm.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                       self.__hardware_image_on_slm.size, wait_for_trigger, output_pulse_image_flip,
                                       output_pulse_image_refresh, timeout_ms)
            self.__slm_lib.ImageWriteComplete(self.__board_number, timeout_ms)

    def _disconnect(self):
        with self._lock:
            if self.__slm_lib is not None:
                log.info('Closing Meadowlark Blink SDK...')
                self.__slm_lib.Delete_SDK()

    def __str__(self) -> str:
        return f'{__class__.__name__}(board_number={self.board_number}, deflection_frequency={self.deflection_frequency}, wavelength={self.wavelength})'
