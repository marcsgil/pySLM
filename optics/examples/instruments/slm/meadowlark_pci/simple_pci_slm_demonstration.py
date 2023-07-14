# Example usage of Blink_C_wrapper.dll
# Meadowlark Optics Spatial Light Modulators
# September 12 2019

import os
from ctypes import *
from scipy import misc
from time import sleep
import pathlib
import traceback
import numpy as np

from examples.instruments.slm.meadowlark_pci import log
from optics.utils.ft import Grid


if __name__ == '__main__':
    serial_number = 4937
    wavelength = 488e-9
    
    # Load the DLL
    # Blink_C_wrapper.dll, Blink_SDK.dll, ImageGen.dll, FreeImage.dll and wdapi1021.dll
    # should all be located in the same directory as the program referencing the
    # library
    log.info(f'Adding {pathlib.Path(__file__).parent.absolute().as_posix()} to dll search path.')
    os.add_dll_directory(pathlib.Path(__file__).parent.absolute().as_posix())

    log.info('Loading Blink_C_wrapper...')
    cdll.LoadLibrary('Blink_C_wrapper')
    slm_lib = CDLL('Blink_C_wrapper')
    # Don't do anything here, because if it crashes, the library will not be unloaded
    try:
        log.info('Loading image generation functions...')
        cdll.LoadLibrary('ImageGen')
        image_lib = CDLL('ImageGen')

        # Basic parameters for calling Create_SDK
        bit_depth = c_uint(12)
        num_boards_found = c_uint(0)
        constructed_okay = c_uint(-1)
        is_nematic_type = c_bool(1)
        ram_write_enable = c_bool(1)
        use_gpu = c_bool(1)
        max_transients = c_uint(20)
        board_number = c_uint(1)
        wait_for_trigger = c_uint(0)
        timeout_ms = c_uint(5000)

        # Both pulse options can be false, but only one can be true. You either generate a pulse when the new image begins loading to the SLM
        # or every 1.184 ms on SLM refresh boundaries, or if both are false no output pulse is generated.
        output_pulse_image_flip = c_uint(0)
        output_pulse_image_refresh = c_uint(0)  #only supported on 1920x1152, FW rev 1.8.

        # Call the Create_SDK constructor
        # Returns a handle that's passed to subsequent SDK calls
        slm_lib.Create_SDK(bit_depth, byref(num_boards_found), byref(constructed_okay), is_nematic_type, ram_write_enable,
                           use_gpu, max_transients, 0)

        if constructed_okay.value == 0:
            log.info("Blink SDK did not construct successfully")
            # Python ctypes assumes the return value is always int
            # We need to tell it the return type by setting restype
            slm_lib.Get_last_error_message.restype = c_char_p
            log.info (slm_lib.Get_last_error_message())

        if num_boards_found.value == 1:
            log.info("Blink SDK was successfully constructed")
            log.info("Found %s SLM controller(s)" % num_boards_found.value)
            slm_height = c_uint(slm_lib.Get_image_height(board_number)).value
            slm_width = c_uint(slm_lib.Get_image_width(board_number)).value
            slm_shape = np.array([slm_height, slm_width])

            grid = Grid(slm_shape)
            #test_phases = (grid[0]**2 + grid[1]**2 < 300**2) * np.pi

            test_phases = np.zeros(slm_shape)
            #test_phases[:, test_phases.shape[1] // 2:] = np.pi

            # ***you should replace *bit_linear.LUT with your custom LUT file***
            # but for now open a generic LUT that linearly maps input graylevels to output voltages
            # ***Using *bit_linear.LUT does NOT give a linear phase response***
            lut_filepath = (pathlib.Path(__file__).parent.absolute() / f'slm{serial_number}_at{wavelength / 1e-9:0.0f}.lut').as_posix().encode('utf-8')  # convert to bytes!
            log.info(f'Loading lookup table {lut_filepath}...')
            slm_lib.Load_LUT_file(board_number, lut_filepath)

            # Create two vectors to hold values for two SLM images with opposite ramps.
            phases = np.empty([slm_height, slm_width], np.uint8, 'C')
            # Write a blank pattern to the SLM to get going
            # image_lib.Generate_Solid(image_one.ctypes.data_as(POINTER(c_ubyte)), slm_width, slm_height, 0)

            for _ in range(1):
                log.info(_)
                phases[:] = (128 + test_phases * (128 / np.pi)).astype(np.uint8)
                # image_one[:, :slm_width//2] = 0  # todo: figure this out
                # phases[:, slm_width // 2:] = 128  # todo: figure this out
                slm_lib.Write_image(board_number, phases.ctypes.data_as(POINTER(c_ubyte)), slm_height * slm_width,
                                    wait_for_trigger, output_pulse_image_flip, output_pulse_image_refresh, timeout_ms)
                slm_lib.ImageWriteComplete(board_number, timeout_ms)
                sleep(10e-3)
            input("Press Enter to close the program.")
    except Exception as e:
        log.error('An unexpected exception occurred:')
        log.error(traceback.format_exc())

        # Always call Delete_SDK before exiting
    log.info('Closing SDK...')
    slm_lib.Delete_SDK()

    log.info('Done.')
