# Example usage of Blink_SDK_C.dll
# Meadowlark Optics Spatial Light Modulators
# last updated: September 10 2020

# Load the DLL
# Blink_C_wrapper.dll, Blink_SDK.dll, ImageGen.dll, FreeImage.dll and wdapi1021.dll
# should all be located in the same directory as the program referencing the library

import numpy as np
from ctypes import *
import time
import os
import pathlib
import traceback
import matplotlib.pyplot as plt
from tqdm import tqdm


from optics.instruments.cam.ids_cam import IDSCam
from optics.utils import Roi

from examples import log
from optics.utils.display import grid2extent

if __name__ == '__main__':
    output_path = pathlib.Path(__file__).parent.absolute() / 'output'
    pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)

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

        log.info('Defining basic parameters for calling Create_SDK')
        bit_depth = c_uint(12)  # input
        num_boards_found = c_uint(0)  # output
        construction_success = c_uint(-1)  # output
        is_nematic_type = c_bool(True)  # input
        ram_write_enable = c_bool(True)  # input
        use_gpu = c_bool(True)  # input
        max_transients = c_uint(20)  # input
        board_number = c_uint(1)  # input
        wait_for_trigger = c_uint(0)  # input
        timeout_ms = c_uint(5000)  # input

        # Both pulse options can be false, but only one can be true. You either generate a pulse when the new image begins loading to the SLM
        # or every 1.184 ms on SLM refresh boundaries, or if both are false no output pulse is generated.
        output_pulse_image_flip = c_uint(0)
        output_pulse_image_refresh = c_uint(0)  #only supported on 1920x1152, FW rev 1.8.

        # - This regional LUT file is only used with Overdrive Plus, otherwise it should always be a null string
        reg_lut = c_char_p #This parameter is specific to the small 512 with Overdrive, do not edit

        log.info('Constructing the SDK...')  # Returns a handle that's passed to subsequent SDK calls
        slm_lib.Create_SDK(bit_depth, byref(num_boards_found), byref(construction_success), is_nematic_type, ram_write_enable, use_gpu, max_transients, 0)

        if construction_success.value == 0:
            log.info('Blink SDK did not construct successfully')
            # Python ctypes assumes the return value is always int
            # We need to tell it the return type by setting restype
            slm_lib.Get_last_error_message.restype = c_char_p
            log.info(f'Last error message: {slm_lib.Get_last_error_message()}')

        if num_boards_found.value >= 1:
            log.info('Blink SDK was successfully constructed')
            log.info(f'Found {num_boards_found.value} SLM controller(s)')

            log.info('Getting dimensions...')
            slm_height = c_uint(slm_lib.Get_image_height(board_number)).value
            slm_width = c_uint(slm_lib.Get_image_width(board_number)).value
            slm_shape = np.array((slm_height, slm_width))
            log.info(f'SLM shape [height, width] = {slm_shape} in pixels.')

            log.info('To measure the raw response we want to disable the LUT by loading a linear LUT...')
            if slm_width == 512:
                linear_lut_file_name = b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\8bit_linear.LUT"
            else:
                linear_lut_file_name = b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\12bit_linear.LUT"
            slm_lib.Load_LUT_file(board_number, linear_lut_file_name)

            nb_gray_levels = 256 #todo change back to 256
            nb_regions = 1
            stripe_width_px = 8

            log.info('Create two vectors to hold values for two SLM images with opposite ramps.')
            image = np.empty([np.prod(slm_shape)], np.uint8, 'C')

            # Generate a blank wavefront correction image, you should load your
            # custom wavefront correction that was shipped with your SLM.
            log.info('Writing a blank pattern to the SLM to get going.')
            pixel_value = 0
            image_lib.Generate_Solid(image.ctypes.data_as(POINTER(c_ubyte)), slm_width, slm_height, pixel_value)
            slm_lib.Write_image(board_number, image.ctypes.data_as(POINTER(c_ubyte)), slm_height * slm_width,
                                wait_for_trigger, output_pulse_image_flip, output_pulse_image_refresh, timeout_ms)
            slm_lib.ImageWriteComplete(board_number, timeout_ms)

            # Generate the stripe pattern and mask out current region
            mid_range_gray_level = 128
            test_region_index = 0
            image_lib.Generate_Stripe(image.ctypes.data_as(POINTER(c_ubyte)), slm_width, slm_height,
                                      pixel_value, mid_range_gray_level, stripe_width_px)
            image_lib.Mask_Image(image.ctypes.data_as(POINTER(c_ubyte)), slm_width, slm_height, test_region_index,
                                 nb_regions)
            # write the image
            slm_lib.Write_image(board_number, image.ctypes.data_as(POINTER(c_ubyte)), slm_height * slm_width,
                                wait_for_trigger, output_pulse_image_flip, output_pulse_image_refresh, timeout_ms)

            #input("Align iris. Press Enter to find the peak intensity.")
            # Always call Delete_SDK before exiting

            with IDSCam(index=0, normalize=True, exposure_time=10e-3) as cam:
                # need to produce a first image for the adjustment of the parameters of the camera
                log.info('Centering area of interest around brightest spot...')
                cam.center_roi_around_peak_intensity(shape=(511, 511), target_graylevel_fraction=0.40)  # also sets the exposure time so that it is not saturated or too dark
                #cam.roi = Roi(top_left=(750, 1300), bottom_right=(1550, 2500))
                cam.roi = None

                log.info(f'Centered camera {cam.roi.shape} region of interest around {cam.roi.center} with exposure time {cam.exposure_time / 1e-3} ms.')

                # fig, ax = plt.subplots(1, 1)
                # print(ax)
                fig= plt.figure()
                #ax = axs[:]
                # plt.imshow(cam.acquire())
                # plt.show()
                ax = plt.imshow(np.zeros(cam.roi.shape), extent=grid2extent(cam.roi.grid),
                                cmap='jet', clim=[0, 1])
                fig_status = True
                log.info('Adjusting camera position..')
                while fig_status == True:
                    ax.set_data(cam.acquire())
                    #plt.imshow(cam.acquire())
                    plt.show(block=False)
                    plt.pause(0.1)
                    fig_status = plt.fignum_exists(fig.number)

                log.info('Adjusting camera roi...')
                logical_control = ''
                # while logical_control != '':
                #     x_pixels = int(input('Number of pixel in x axis:'))
                #     y_pixels = int(input('Number of pixel in y axis:'))
                #     cam.center_roi_around_peak_intensity(shape=(x_pixels, y_pixels), target_graylevel_fraction=0.40)
                #     fig, ax = plt.subplots(1, 1)
                #     ax.set_data(cam.acquire())
                #     plt.show()




                def int_measurement():
                    number_measurements = 10
                    cam.acquire()  # discard the first one to clear the queue
                    intensity = np.zeros(number_measurements)
                    for _ in range(number_measurements):
                        intensity[_] = np.mean(cam.acquire())
                    mean_intensity = np.mean(intensity)
                    standard_deviation = np.std(intensity)
                    #return np.mean([np.mean(cam.acquire()) for _ in range(10)]) #todo change back to 10
                    return mean_intensity, standard_deviation
                #input("Press Enter to start measurement.")

                intensity_value = np.zeros([2, nb_gray_levels]) #todo remove after the test
                gray_array = np.zeros(nb_gray_levels)

                for region_index in range(nb_regions):
                    log.info(f'Looping through each graylevel for region {region_index} / {nb_regions}')
                    # Create an array to hold measurements from the analog input (AI) board
                    analog_input_intensities = []
                    for gray_level in tqdm(range(nb_gray_levels), desc='gray level'):
                        gray_array[gray_level] = gray_level #todo remove after the test
                        # Generate the stripe pattern and mask out current region
                        image_lib.Generate_Stripe(image.ctypes.data_as(POINTER(c_ubyte)), slm_width, slm_height,
                                                  pixel_value, gray_level, stripe_width_px)
                        image_lib.Mask_Image(image.ctypes.data_as(POINTER(c_ubyte)), slm_width, slm_height, region_index,
                                             nb_regions)
                        # write the image
                        slm_lib.Write_image(board_number, image.ctypes.data_as(POINTER(c_ubyte)), slm_height * slm_width,
                                            wait_for_trigger, output_pulse_image_flip, output_pulse_image_refresh, timeout_ms)

                        # let the SLM settle for 10 ms
                        time.sleep(0.010)

                        # YOU FILL IN HERE...FIRST: read from your specific AI board, note it might help to clean up noise to average several readings
                        #measured_value = int_measurement()  # TODO: This should be read from the camera
                        intensity_value[:, gray_level] = int_measurement()

                        # SECOND: store the measurement in your AI_Intensities array
                        #analog_input_intensities.append((gray_level, measured_value))
                        analog_input_intensities.append((gray_level, intensity_value[0, gray_level]))

                    # dump the AI measurements to a csv file
                    output_filepath = output_path / f'Raw{region_index}.csv'
                    log.info(f'Writing output for region {region_index} to {output_filepath}...')
                    np.savetxt(output_filepath, analog_input_intensities, delimiter=', ', fmt=['%0.0f', '%0.9f'])
                    log.info(f'Written output for region {region_index} to {output_filepath}.')

    except Exception as e:
        log.error('An unexpected exception occurred:')
        log.error(traceback.format_exc())

    # Always call Delete_SDK before exiting
    log.info('Closing SDK...')
    slm_lib.Delete_SDK()

    log.info('Done.')

    plt.errorbar(gray_array, intensity_value[0, :], intensity_value[1, :], ecolor = 'red', color='black')
    plt.show()