import numpy as np
import logging
import ctypes
from typing import Union, Sequence, Optional, Callable, List
import time
from .cam import Cam, CamError, CamDescriptor
from optics.utils.roi import Roi
# from optics.instruments.cam.ids_peak_cam import IDSCamDescriptor

log = logging.getLogger(__name__)


class IDSCamDescriptor(CamDescriptor):
    def __init__(self, id: str, constructor: Callable, available: bool = True, serial: str = '', model: str = '', index: int = -1):
        super().__init__(id, constructor, available, serial, model)
        self.__index: str = index

    @property
    def index(self) -> int:
        """The index of this camera."""
        return self.__index

try:
    from pyueye import ueye  # This requires the installation of the ueye driver library
except ImportError as err:
    log.warning('Python bindings for the uEye API not found. This is required to control IDS cameras. Please install the pyueye package installed with "pip install pyueye".')


class IDSCamError(CamError):
    @staticmethod
    def check(status: int, error_message=None) -> int:
        if status != ueye.IS_SUCCESS:
            raise IDSCamError(status, error_message=error_message)
        return status

    def __init__(self, status: int = None, error_message=None):
        if status is None:
            message = error_message
        else:
            message = f"Error {status}: "
            if status == ueye.IS_SUCCESS:
                message += 'Success. No error code returned.'
            elif status == ueye.IS_NO_SUCCESS:
                message += 'No success.'
            elif status == ueye.IS_INVALID_CAMERA_HANDLE:
                message += 'Invalid camera handle.'
            elif status == ueye.IS_CANT_OPEN_DEVICE:
                message += 'Cannot open device.'
            elif status == ueye.IS_IO_REQUEST_FAILED:
                message += 'IO request failed.'
            elif status == ueye.IS_NOT_SUPPORTED:
                message += 'Not supported. The used camera model does not support this function or attitude.'
            elif status == ueye.IS_INVALID_PARAMETER:
                message += 'Invalid parameter. One of the parameters is outside the valid range, or is not supported for this sensor and/or is not accessible in this mode.'
            elif status == ueye.IS_OUT_OF_MEMORY:
                message += 'Out of memory'
            elif status == ueye.IS_NO_USB20:
                message += 'Not USB 2.0'
            elif status == ueye.IS_CAPTURE_RUNNING:
                message += 'Current acquisition must be terminated before a new one can begin.'
            elif status == ueye.IS_NO_ACTIVE_IMG_MEM:
                message += "NO_ACTIVE_IMG_MEM"
            elif status == ueye.IS_NOT_ALLOWED:
                message += "NOT_ALLOWED"
            elif status == ueye.IS_SEQUENCE_BUF_ALREADY_LOCKED:
                message += "Sequence buffer is already locked."
            elif status == 208:  # From file:///C:/Program%20Files/IDS/uEye/Help/uEye_Manual/index.html > Error codes
                message += "The image sequence is locked."
            elif status == ueye.IS_NO_SUCH_DEVICE:  # From file:///C:/Program%20Files/IDS/uEye/Help/uEye_Manual/index.html > Error codes
                message += "No such device found. Is the camera connected and available?"
            else:
                message += "Unknown cause"
            # Error Codes can be found here file:///C:/Program%20Files/IDS/uEye/Help/uEye_Manual/index.html

            if error_message is not None:
                message += " " + error_message

        super().__init__(message)


class IDSCam(Cam):
    """
    A class to control the IDS cameras.
    """

    @classmethod
    def list(cls, recursive: bool = False, include_unavailable: bool = False) -> List[Union[IDSCamDescriptor, List]]:
        """
        Return all constructors.
        :return: A dictionary with as key the class and as value, a dictionary with subclasses.
        """
        nb_connected_cameras_c = ctypes.c_int32()
        IDSCamError.check(ueye.is_GetNumberOfCameras(nb_connected_cameras_c))
        nb_connected_cameras = nb_connected_cameras_c.value

        camera_descriptors = []
        if nb_connected_cameras >= 1:
            camera_list_c = ueye.UEYE_CAMERA_LIST(uci=(ueye.UEYE_CAMERA_INFO * nb_connected_cameras))
            camera_list_c.dwCount = nb_connected_cameras
            IDSCamError.check(ueye.is_GetCameraList(camera_list_c), 'Could not list cameras')
            for cam_idx in range(camera_list_c.dwCount.value):
                cam = camera_list_c.uci[cam_idx]
                log.debug(f'IDSCam{cam_idx}(camera={cam.dwCameraID}, device={cam.dwDeviceID}, sensor={cam.dwSensorID},' +
                          f' in_use={cam.dwInUse == 1}, status={cam.dwStatus}, SerNo={cam.SerNo}, Model={cam.Model},' +
                          f'FullModelName={cam.FullModelName})')
                if include_unavailable or cam.dwInUse.value != 1:
                    index = cam.dwDeviceID.value
                    model = cam.FullModelName.decode('utf-8')
                    serial = cam.SerNo.decode('utf-8')
                    available = cam.dwInUse.value
                    camera_descriptors.append(
                        IDSCamDescriptor(id=f'IDSCam{index}_{serial}_{model}', constructor=lambda: IDSCam(index=index, serial=serial),
                                         available=available, serial=serial, model=model, index=index)
                    )

        return camera_descriptors

    class ImageBuffer:
        def __init__(self, cam_id, shape, bit_depth):
            self.__cam_id = cam_id
            self.id = ueye.int()
            self.ptr = ueye.c_mem_p()
            self.__shape = shape
            self.__bit_depth = bit_depth

            self.__image_array = np.empty(shape=self.__shape, dtype=self.dtype)

        @property
        def shape(self):
            return self.__shape

        @property
        def bit_depth(self):
            return self.__bit_depth

        @property
        def dtype(self):
            if self.__bit_depth == 8:
                return np.uint8
            elif self.__bit_depth <= 16:
                return np.uint16
            else:
                return np.uint32

        @property
        def array(self):
            """Get array (irreversible) """
            try:
                # # The buffer is locked as we receive it. Do not double lock it!
                # IDSCamError.check(ueye.is_LockSeqBuf(self.__cam_id, self.id, self.ptr),
                #                   f'Could not lock sequence buffer {self.id}.')
                # log.info(f'Locked sequence buffer {self.id}...')
                start_time = time.perf_counter()

                # width_c = ctypes.c_int32()
                # height_c = ctypes.c_int32()
                # bit_depth_c = ctypes.c_int32()
                # image_memory_line_increment_c = ctypes.c_int32()
                # IDSCamError.check(ueye.is_InquireImageMem(self.__cam_id, self.ptr, self.id,
                #                                           width_c, height_c, bit_depth_c, image_memory_line_increment_c))
                # sh = np.array([height_c.value, width_c.value])
                # bit_depth = bit_depth_c.value
                # pitch = image_memory_line_increment_c.value
                # log.info(f'{sh}, bits: {bit_depth},  pitch: {pitch}')
                # self.__image_array = ueye.get_data(self.ptr,
                #                                    1280, 1024,
                #                                    self.bit_depth,
                #                                    1280,
                #                                    copy=True)  # Undocumented
                # self.__image_array = self.__image_array.reshape(self.shape)

                IDSCamError.check(ueye.is_CopyImageMem(self.__cam_id, self.ptr, self.id, self.__image_array.ctypes.data),
                                  'Could not copy image memory.')
                log.debug(f'Copied image in {(time.perf_counter() - start_time)*1e3:0.3f} ms')
            finally:
                IDSCamError.check(ueye.is_UnlockSeqBuf(self.__cam_id, self.id, self.ptr),
                                  f'Could not unlock sequence buffer {self.id}.')
                log.debug(f'Unlocked sequence buffer {self.id}.')

            return self.__image_array

    def __init__(self, index: Optional[int] = None, serial: Union[int, str, None] = None, model: Optional[str] = None,
                 normalize=True, exposure_time: Optional[float] = None, frame_time: Optional[float] = None,
                 gain: float = 0.0, black_level=128):
        super().__init__(normalize=normalize)

        self.__gain = None
        self.__hardware_exposure_time_target = None  # Longest exposure time for the frame rate
        self.__hardware_frame_time_target = None  # Fastest frame rate for given exposure time
        self.__acquiring = False

        cam_list_c = ueye.mem_p()
        IDSCamError.check(ueye.is_GetCameraList(cam_list_c))
        if index is None:
            if serial is not None:
                if isinstance(serial, int):
                    serial = f'{serial:0.0f}'
                index = -1
                cam_desc_list = IDSCam.list()
                for cam_desc in cam_desc_list:
                    if cam_desc.serial == serial:
                        index = cam_desc.index
                        break
                if index == -1:
                    raise ValueError(f'Camera with serial number "{serial}" not found or in use.')
            elif model is not None:
                index = -1
                cam_desc_list = IDSCam.list()
                for cam_desc in cam_desc_list:
                    if cam_desc.model.lower() == model.lower():
                        index = cam_desc.index
                        break
                if index == -1:
                    raise ValueError(f'Camera model "{model}" not found or in use.')
            else:
                index = -1  # assignable int (0 = any camera, 1 = first camera, 2 = second camera, ...)
        log.info(f'Opening camera with index {index}...')
        cam_desc_list = IDSCam.list()
        for cam_desc in cam_desc_list:
            if cam_desc.index == index:
                self.__index = index
                self.__serial = cam_desc.serial
                self.__model = cam_desc.model
                break
        self.__id = ueye.HIDS(index | ueye.IS_USE_DEVICE_ID)
        try:
            self.__standby_supported = ueye.is_CameraStatus(self.__id, ueye.IS_STANDBY_SUPPORTED,
                                                            ueye.IS_GET_STATUS) != 0
            standby = ueye.is_CameraStatus(self.__id, ueye.IS_STANDBY, ueye.IS_GET_STATUS) != 0
            log.debug(f'Standby supported {self.__standby_supported}. Standby currently enabled: {standby}.')

            display_window_handle = ctypes.c_void_p()
            IDSCamError.check(ueye.is_InitCamera(self.__id, display_window_handle),
                              "Could not initialize camera. Is it connected to this computer and is any other program using it switched off?")
            log.debug("Camera initialized.")
            IDSCamError.check(ueye.is_EnableAutoExit(self.__id, ctypes.c_uint(1)), "Could not enable auto exit.")
            log.debug("Camera connected.")

            self.trigger = False

            # Initialize
            IDSCamError.check(ueye.is_ResetToDefault(self.__id), "Could not reset camera.")
            IDSCamError.check(ueye.is_SetDisplayMode(self.__id, ueye.IS_SET_DM_DIB),
                              "Could not set display mode to Device Independent Bitmap.")

            self.bit_depth = 8  # default to a bit depth of 8 bits

            sensor_info = ueye.SENSORINFO()
            IDSCamError.check(ueye.is_GetSensorInfo(self.__id, sensor_info), "Could not get sensor info.")
            cam_info = ueye.CAMINFO()
            IDSCamError.check(ueye.is_GetCameraInfo(self.__id, cam_info), "Could not get camera info.")

            self.__global_shutter_supported = sensor_info.bGlobShutter == ueye.IS_SET_GLOBAL_SHUTTER_ON
            if self.__global_shutter_supported:
                log.info('Global shutter is supported.')
            else:
                log.info('Global shutter is NOT supported.')

            max_shape = np.array([sensor_info.nMaxHeight, sensor_info.nMaxWidth])
            name = sensor_info.strSensorName.decode('utf-8')
            camera_id = sensor_info.SensorID
            color_mode = sensor_info.nColorMode
            color = color_mode != ueye.IS_COLORMODE_MONOCHROME
            log.debug(f'Connected camera model {name}(id={camera_id})[{max_shape[1]}x{max_shape[0]}].')
            pixel_pitch = np.ones(2) * sensor_info.wPixelSize * 0.01e-6

            self.__memory_buffers = []

            # Turn off the gamma correction
            # gamma_c = ctypes.c_int32(int(0.01 * 100))
            # IDSCamError.check(ueye.is_Gamma(self.__id, ueye.IS_GAMMA_CMD_SET, gamma_c, ueye.sizeof(gamma_c)))
            IDSCamError.check(ueye.is_SetHardwareGamma(self.__id, ueye.IS_SET_HW_GAMMA_OFF))

            # Set black level to a fixed value
            try:
                self.black_level = 128
            except IDSCamError as exc:
                log.error(f'Could not set black level on this model of IDS camera: {exc}')

            # Turn on global shutter
            log.debug('Global shutter' + (' ' if self.__global_shutter_supported else ' not ') + 'supported.')
            try:
                self.global_shutter = True
            except IDSCamError as exc:
                self.__global_shutter_supported = False
                log.error(f'Could not set global shutter on this model of IDS camera: {exc}')

            self.pixel_clock = np.inf
            log.debug(f'Pixel clock set to {self.pixel_clock*1e-6:0.3f} MHz.')

            self._set_hardware_bin(1)

            self.__auto_exposure = False
        except Exception as exc:
            log.warning(f"Error during construction of IDSCam object: {str(exc)}.")
            try:
                IDSCamError.check(ueye.is_CameraStatus(self.__id, ueye.IS_STANDBY, True))
            except CamError as ce:
                log.info(f'Could not set camera in standby mode: {ce}')
            try:
                IDSCamError.check(ueye.is_ExitCamera(self.__id))
            except CamError as ce:
                pass
            finally:
                self.__id = None
                raise exc

        self.exposure_time = exposure_time
        self.frame_time = frame_time
        try:
            self.gain = gain
        except IDSCamError as exc:
            log.error(f'Could not set gain on this model of IDS camera: {exc}')

        try:
            self.black_level = black_level
        except IDSCamError as exc:
            log.error(f'Could not set black level on this model of IDS camera: {exc}')

        # The super class sets the roi and thereby starts continuous acquisition
        super().__init__(shape=max_shape, normalize=normalize, color=color, pixel_pitch=pixel_pitch,
                         exposure_time=self.exposure_time, frame_time=self.frame_time, gain=self.gain)

    @property
    def index(self) -> int:
        return self.__index

    @property
    def serial(self) -> int:
        return self.__serial

    @property
    def model(self) -> str:
        return self.__model

    @property
    def bit_depth(self) -> int:
        with self._lock:
            return self.__bit_depth

    @bit_depth.setter
    def bit_depth(self, new_bit_depth: int):
        with self._lock:
            try:
                if new_bit_depth == 16:
                    bit_depth_c = ueye.IS_CM_MONO16  # also: IS_CM_SENSOR_RAW8
                elif new_bit_depth == 12:
                    bit_depth_c = ueye.IS_CM_MONO12
                    # bit_depth_c = ctypes.c_int32(ueye.IS_SENSOR_BIT_DEPTH_12_BIT)
                elif new_bit_depth == 10:
                    bit_depth_c = ueye.IS_CM_MONO10
                else:
                    bit_depth_c = ueye.IS_CM_MONO8

                # bit_depths_c = ctypes.c_int32(0)

                IDSCamError.check(ueye.is_SetColorMode(self.__id, bit_depth_c), "Could not set color mode.")
                # IDSCamError.check(ueye.is_SetColorMode(self.__id, ueye.IS_CM_SENSOR_RAW8), "Could not set color mode RAW8.")
                # IDSCamError.check(ueye.is_SetColorMode(self.__id, ueye.IS_CM_SENSOR_RAW10), "Could not set color mode RAW10.")

                # IDSCamError.check(ueye.is_DeviceFeature(self.__id, ueye.IS_DEVICE_FEATURE_CMD_GET_SUPPORTED_SENSOR_BIT_DEPTHS,
                #                                     bit_depths_c, ctypes.sizeof(bit_depths_c)), 'Could not set device feature bit depth.')
                # log.info(bit_depths_c.value)
                # if (bit_depths_c.value & bit_depth_c.value) == 0:
                #     # If not supported, set to 8 bit
                #     bit_depth_c = ctypes.c_int32(ueye.IS_SENSOR_BIT_DEPTH_8_BIT)
                # # Set the new bit depth
                # IDSCamError.check(ueye.is_DeviceFeature(self.__id, ueye.IS_DEVICE_FEATURE_CMD_SET_SENSOR_BIT_DEPTH,
                #                                     bit_depth_c, ctypes.sizeof(bit_depth_c)))
                # Check the new bit depth
                # bit_depth_c = ctypes.c_int32()
                # IDSCamError.check(ueye.is_DeviceFeature(self.__id, ueye.IS_DEVICE_FEATURE_CMD_GET_SENSOR_BIT_DEPTH,
                #                                     bit_depth_c, ctypes.sizeof(bit_depth_c)))
                # self.__bit_depth = bit_depth_c.value
                color_mode = ueye.is_SetColorMode(self.__id, ueye.IS_GET_COLOR_MODE)
                if color_mode == ueye.IS_CM_MONO16 or color_mode == ueye.IS_CM_SENSOR_RAW16:
                    self.__bit_depth = 16
                elif color_mode == ueye.IS_CM_MONO12 or color_mode == ueye.IS_CM_SENSOR_RAW12:
                    self.__bit_depth = 12
                elif color_mode == ueye.IS_CM_MONO10 or color_mode == ueye.IS_CM_SENSOR_RAW10:
                    self.__bit_depth = 10
                elif color_mode == ueye.IS_CM_MONO8 or color_mode == ueye.IS_CM_SENSOR_RAW8:
                    self.__bit_depth = 8
                else:
                    raise CamError(f'Unknown color mode: {color_mode}')
            except CamError as ce:
                log.info(ce)
                self.__bit_depth = 8
            log.debug(f'Bit depth is set to {self.__bit_depth}')
            # Set frame rate (and exposure time) again
            self._set_hardware_frame_time(self.__hardware_frame_time_target)

    @property
    def gain(self) -> float:
        """Get the gain as a value between 0 and 1."""
        with self._lock:
            return self.__gain

    @gain.setter
    def gain(self, gain: float):
        """Set the gain as a value between 0 and 1."""
        try:
            with self._lock:
                if self.__id is not None:
                    master = ueye.c_int(np.clip(int(gain * 100.0), 0, 100))
                    red = ueye.c_int(0)
                    green = ueye.c_int(0)
                    blue = ueye.c_int(0)
                    IDSCamError.check(ueye.is_SetHardwareGain(self.__id, master, red, green, blue), f'Could not set gain to {master} ({red}, {green}, {blue}).')

                    self.__gain = float(master.value) / 100.0
        except IDSCamError:  # todo
            pass

    @property
    def global_shutter(self):
        # shutter_mode_c = ctypes.c_int32(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL)
        # shutter_mode_c = ctypes.c_int32(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL_ALTERNATIVE_TIMING)
        shutter_mode_c = ctypes.c_int32(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING)
        # shutter_mode_c = ctypes.c_int32(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING_GLOBAL_START)
        IDSCamError.check(ueye.is_DeviceFeature(self.__id, ueye.IS_DEVICE_FEATURE_CMD_GET_SHUTTER_MODE,
                                                shutter_mode_c, ueye.sizeof(shutter_mode_c)))
        return shutter_mode_c.value == ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL

    @global_shutter.setter
    def global_shutter(self, state: bool):
        state = state and self.__global_shutter_supported
        if state:
            shutter_mode_c = ctypes.c_int32(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL)
        else:
            shutter_mode_c = ctypes.c_int32(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING)
        # shutter_mode_c = ctypes.c_int32(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL_ALTERNATIVE_TIMING)
        # shutter_mode_c = ctypes.c_int32(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING_GLOBAL_START)
        IDSCamError.check(ueye.is_DeviceFeature(self.__id, ueye.IS_DEVICE_FEATURE_CMD_SET_SHUTTER_MODE,
                                                shutter_mode_c, ueye.sizeof(shutter_mode_c)))
        # Set frame rate (and exposure time) again
        self._set_hardware_frame_time(self.__hardware_frame_time_target)

    @property
    def black_level(self) -> Optional[int]:
        auto_black_level_c = ctypes.c_int32()
        IDSCamError.check(ueye.is_Blacklevel(self.__id, ueye.IS_BLACKLEVEL_CMD_GET_MODE,  # disable automatic black level compensation
                                             auto_black_level_c, ueye.sizeof(auto_black_level_c)))
        if auto_black_level_c.value == ueye.IS_AUTO_BLACKLEVEL_ON:
            return None
        else:
            black_level_c = ctypes.c_int32()
            IDSCamError.check(ueye.is_Blacklevel(self.__id, ueye.IS_BLACKLEVEL_CMD_GET_OFFSET,
                                                 black_level_c, ueye.sizeof(black_level_c)))
            return black_level_c.value

    @black_level.setter
    def black_level(self, new_level: Optional[int]):
        if new_level is None:
            auto_black_level_c = ctypes.c_int32(ueye.IS_AUTO_BLACKLEVEL_ON)
            IDSCamError.check(ueye.is_Blacklevel(self.__id, ueye.IS_BLACKLEVEL_CMD_SET_MODE,  # enable automatic black level compensation
                                                 auto_black_level_c, ueye.sizeof(auto_black_level_c)))
        else:
            auto_black_level_c = ctypes.c_int32(ueye.IS_AUTO_BLACKLEVEL_OFF)
            IDSCamError.check(ueye.is_Blacklevel(self.__id, ueye.IS_BLACKLEVEL_CMD_SET_MODE,  # disable automatic black level compensation
                                                 auto_black_level_c, ueye.sizeof(auto_black_level_c)))
            black_level_c = ctypes.c_int32(new_level)
            IDSCamError.check(ueye.is_Blacklevel(self.__id, ueye.IS_BLACKLEVEL_CMD_SET_OFFSET,
                                                 black_level_c, ueye.sizeof(black_level_c)))

    @property
    def pixel_clock(self) -> float:
        pixel_clock_c = ctypes.c_int32()
        IDSCamError.check(ueye.is_PixelClock(self.__id, ueye.IS_PIXELCLOCK_CMD_GET,
                                             pixel_clock_c, ueye.sizeof(pixel_clock_c)))
        return pixel_clock_c.value * 1e6

    @pixel_clock.setter
    def pixel_clock(self, new_frequency: Optional[float]):
        if new_frequency is None:
            # # Measure transfer speed
            # mode_c = ctypes.c_int32(ueye.IS_BEST_PCLK_RUN_ONCE)
            # timeout_c = ctypes.c_int32(4*1000)  # measurement time in ms, should be between 4 and 20s
            # max_pixel_clock_c = ctypes.c_int32()  # maximum pixel clock in MHz
            # max_frame_rate_c = ctypes.c_double()  # maximum frame rate in Hz
            # IDSCamError.check(ueye.is_SetOptimalCameraTiming(self.__id, mode_c, timeout_c,
            #                                              max_pixel_clock_c, max_frame_rate_c))
            # max_pixel_clock = max_pixel_clock_c.value * 1.0e6
            # min_frame_time = 1.0 / max_frame_rate_c.value
            # print(f'max_pixel_clock={max_pixel_clock} MHz,  min_frame_time={min_frame_time} s')

            # Set the default
            pixel_clock_c = ctypes.c_int32(int(new_frequency * 1e-6))
            IDSCamError.check(ueye.is_PixelClock(self.__id, ueye.IS_PIXELCLOCK_CMD_GET_DEFAULT,
                                                 pixel_clock_c, ueye.sizeof(pixel_clock_c)))
            new_frequency = pixel_clock_c.value * 1e6
        # Get the clock range
        pixel_clock_range_c = (ctypes.c_int32 * 3)()
        IDSCamError.check(ueye.is_PixelClock(self.__id, ueye.IS_PIXELCLOCK_CMD_GET_RANGE,
                                             pixel_clock_range_c, ctypes.sizeof(pixel_clock_range_c)))
        # Clipping the frequency to the allowed range
        new_frequency = np.clip(new_frequency, pixel_clock_range_c[0] * 1e6, pixel_clock_range_c[1] * 1e6)

        pixel_clock_c = ctypes.c_int32(int(new_frequency * 1e-6))
        IDSCamError.check(ueye.is_PixelClock(self.__id, ueye.IS_PIXELCLOCK_CMD_SET,
                                             pixel_clock_c, ueye.sizeof(pixel_clock_c)))
        # Set frame rate (and exposure time) again
        self._set_hardware_frame_time(self.__hardware_frame_time_target)

    @property
    def trigger(self) -> bool:
        trigger_setting = IDSCamError.check(ueye.is_SetExternalTrigger(self.__id, ueye.IS_GET_EXTERNALTRIGGER),
                                            "Could not get hardware trigger setting.")
        return trigger_setting == ueye.IS_SET_TRIGGER_SOFTWARE or trigger_setting == ueye.IS_SET_TRIGGER_OFF

    @trigger.setter
    def trigger(self, hardware: bool):
        if hardware:
            IDSCamError.check(ueye.is_SetExternalTrigger(self.__id, ueye.IS_SET_TRIGGER_LO_HI),
                              "Could not set hardware trigger.")
            log.debug("Hardware trigger set.")
        else:
            IDSCamError.check(ueye.is_SetExternalTrigger(self.__id, ueye.IS_SET_TRIGGER_SOFTWARE),
                              "Could not set software trigger.")
            log.debug("Software trigger set.")
            # IDSCamError.check(ueye.is_SetExternalTrigger(self.__id, ueye.IS_SET_TRIGGER_OFF),
            #                   "Could not switch off trigger.")
            # log.info("No trigger set.")
        # Set frame rate (and exposure time) again
        self._set_hardware_frame_time(self.__hardware_frame_time_target)

    def _set_hardware_roi(self, minimum_roi: Roi) -> Roi:
        with self._lock:
            set_roi = minimum_roi
            if self.__id is not None:
                # Determine the new region of interest, compatible with hardware
                if minimum_roi is None:
                    minimum_roi = Roi(shape=self.shape)
                # The following values are valid for UI-124x/UI-324x/UI-524x
                # file:///C:/Program%20Files/IDS/uEye/Help/uEye_Manual/index.html?hw_sensoren.html
                # pos_step = np.array([2, 2])  # todo: detect this automatically
                # min_shape = np.array([4, 16])  # todo: detect this automatically
                # shape_step = np.array([2, 4])  # todo: detect this automatically
                pos_step = np.array([2, 16])  # todo: detect this automatically
                min_shape = np.array([4, 128])  # todo: detect this automatically
                shape_step = np.array([2, 16])  # todo: detect this automatically
                new_roi = Roi(top_left=pos_step*np.floor(minimum_roi.top_left/pos_step),
                              shape=np.maximum(min_shape, shape_step*np.ceil(minimum_roi.shape/shape_step)),
                              dtype=int)
                # min_shape or shape_step rounding can take the roi outside the sensor area, shift it top_left to compensate
                adjusted_top_left = new_roi.top_left
                for axis in range(2):
                    if new_roi.bottom_right[axis] > self.shape[axis]:
                        adjusted_top_left[axis] = new_roi.top_left[axis] - (new_roi.bottom_right[axis] - self.shape[axis])
                new_roi.top_left = adjusted_top_left

                # Set the new region of interest
                rect = ueye.IS_RECT()
                rect.s32Width = ueye.c_int(new_roi.width)
                rect.s32Height = ueye.c_int(new_roi.height)
                rect.s32X = ueye.c_int(new_roi.left)
                rect.s32Y = ueye.c_int(new_roi.top)

                if self.__id is None:
                    raise CamError("Camera closed.")
                try:
                    IDSCamError.check(ueye.is_AOI(self.__id, ueye.IS_AOI_IMAGE_SET_AOI, rect, ueye.sizeof(rect)),
                                      f"Could not set region of interest to {new_roi}.")
                    # Set frame rate (and exposure time) again
                    self._set_hardware_frame_time(self.__hardware_frame_time_target)
                finally:
                    # Read out ROI to be sure
                    IDSCamError.check(ueye.is_AOI(self.__id, ueye.IS_AOI_IMAGE_GET_AOI, rect, ueye.sizeof(rect)),
                                      "Could not read current region of interest.")
                    set_roi = Roi(left=rect.s32X, top=rect.s32Y, width=rect.s32Width, height=rect.s32Height, dtype=int)

                    if set_roi != new_roi:
                        log.warning(f'Failed to set region of interest to {new_roi}, is {set_roi} instead.')

                    IDSCamError.check(ueye.is_SetDisplayMode(self.__id, ueye.IS_SET_DM_DIB),
                                      'Could not set display mode.')

                    # Allocate memory corresponding to this ROI
                    # self._stop_live()

                    # memory initialization
                    for mem_buf in self.__memory_buffers:
                        IDSCamError.check(ueye.is_FreeImageMem(self.__id, mem_buf.ptr, mem_buf.id),
                                          "Could not free image buffer memory.")
                        log.debug("Released memory and allocating for new ROI...")

                    # Allocate new memory
                    self.__memory_buffers = [IDSCam.ImageBuffer(cam_id=self.__id, shape=set_roi.shape,
                                                                bit_depth=self.bit_depth) for _ in range(3)]
                    bits_per_pixel_c = ctypes.c_int(self.bit_depth)
                    for mem_buf in self.__memory_buffers:
                        IDSCamError.check(ueye.is_AllocImageMem(self.__id, set_roi.shape[1], set_roi.shape[0],
                                                                bits_per_pixel_c, mem_buf.ptr, mem_buf.id),
                                          'Could not allocate image memory.')
                        IDSCamError.check(ueye.is_AddToSequence(self.__id, mem_buf.ptr, mem_buf.id),
                                          'Could not add allocated image memory to the ring buffer sequence.')
                    IDSCamError.check(ueye.is_InitImageQueue(self.__id, 0),
                                      'Could not enable queue mode for existing image memory sequences.')
                    log.debug('Image memory allocated and activated.')

                    # self._start_live()

            return set_roi

    def _set_hardware_bin(self, bin_shape: Union[int, Sequence, np.ndarray]) -> np.ndarray:
        bin_shape = np.atleast_1d(bin_shape)

        new_state = ueye.IS_BINNING_DISABLE
        vertical = bin_shape[0]
        if vertical == 2:
            new_state |= ueye.IS_BINNING_2X_VERTICAL
        elif vertical == 3:
            new_state |= ueye.IS_BINNING_3X_VERTICAL
        elif vertical == 4:
            new_state |= ueye.IS_BINNING_4X_VERTICAL
        elif vertical == 5:
            new_state |= ueye.IS_BINNING_5X_VERTICAL
        elif vertical == 6:
            new_state |= ueye.IS_BINNING_6X_VERTICAL
        elif vertical == 8:
            new_state |= ueye.IS_BINNING_8X_VERTICAL
        elif vertical == 16:
            new_state |= ueye.IS_BINNING_16X_VERTICAL
        horizontal = bin_shape[-1]
        if horizontal == 2:
            new_state |= ueye.IS_BINNING_2X_HORIZONTAL
        elif horizontal == 3:
            new_state |= ueye.IS_BINNING_3X_HORIZONTAL
        elif horizontal == 4:
            new_state |= ueye.IS_BINNING_4X_HORIZONTAL
        elif horizontal == 5:
            new_state |= ueye.IS_BINNING_5X_HORIZONTAL
        elif horizontal == 6:
            new_state |= ueye.IS_BINNING_6X_HORIZONTAL
        elif horizontal == 8:
            new_state |= ueye.IS_BINNING_8X_HORIZONTAL
        elif horizontal == 16:
            new_state |= ueye.IS_BINNING_16X_HORIZONTAL

        with self._lock:
            if self.__id is not None:
                # Only permit new binning states that are implemented for this camera
                supported_binning = ueye.is_SetBinning(self.__id, ueye.IS_GET_SUPPORTED_BINNING)
                new_state &= supported_binning  # mask

                # Update the binning setting
                IDSCamError.check(ueye.is_SetBinning(self.__id, new_state))

                # Determine the actually bin shape that was set
                bin_shape = np.ones(2, dtype=int)
                if new_state & ueye.IS_BINNING_2X_VERTICAL:
                    bin_shape[0] = 2
                elif new_state & ueye.IS_BINNING_3X_VERTICAL:
                    bin_shape[0] = 3
                elif new_state & ueye.IS_BINNING_4X_VERTICAL:
                    bin_shape[0] = 4
                elif new_state & ueye.IS_BINNING_5X_VERTICAL:
                    bin_shape[0] = 5
                elif new_state & ueye.IS_BINNING_6X_VERTICAL:
                    bin_shape[0] = 6
                elif new_state & ueye.IS_BINNING_8X_VERTICAL:
                    bin_shape[0] = 8
                elif new_state & ueye.IS_BINNING_16X_VERTICAL:
                    bin_shape[0] = 16
                if new_state & ueye.IS_BINNING_2X_HORIZONTAL:
                    bin_shape[-1] = 2
                elif new_state & ueye.IS_BINNING_3X_HORIZONTAL:
                    bin_shape[-1] = 3
                elif new_state & ueye.IS_BINNING_4X_HORIZONTAL:
                    bin_shape[-1] = 4
                elif new_state & ueye.IS_BINNING_5X_HORIZONTAL:
                    bin_shape[-1] = 5
                elif new_state & ueye.IS_BINNING_6X_HORIZONTAL:
                    bin_shape[-1] = 6
                elif new_state & ueye.IS_BINNING_8X_HORIZONTAL:
                    bin_shape[-1] = 8
                elif new_state & ueye.IS_BINNING_16X_HORIZONTAL:
                    bin_shape[-1] = 16

                # Set frame rate (and exposure time) again
                self._set_hardware_frame_time(self.__hardware_frame_time_target)

                return bin_shape

    def _set_hardware_exposure_time(self, exposure_time: Optional[float]= None) -> Optional[float]:
        """
        Set the hardware exposure time and update the frame rate accordingly.

        :param exposure_time: The target exposure time in seconds.
        :return: The actual set exposure time in seconds.
        """
        with self._lock:
            self.__hardware_exposure_time_target = exposure_time
            if self.__id is not None:
                # Get the maximum exposure for this frame rate
                max_exposure_c = ueye.double()
                IDSCamError.check(ueye.is_Exposure(self.__id, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX,
                                                   max_exposure_c, ctypes.sizeof(max_exposure_c)))
                max_exposure = max_exposure_c.value * 1e-3
                if exposure_time is None:
                    exposure_time = max_exposure
                else:
                    if max_exposure < exposure_time:
                        frame_rate_c = ueye.c_double(1.0 / (exposure_time + 15e-6))
                        new_frame_rate_c = ueye.double()
                        IDSCamError.check(ueye.is_SetFrameRate(self.__id, frame_rate_c, new_frame_rate_c))
                        new_frame_time = 1.0 / new_frame_rate_c.value
                        log.info(f'Increased frame time to {new_frame_time*1e3:0.3f} ms')

                return self.__set_hardware_exposure_time_direct(exposure_time)

    def __set_hardware_exposure_time_direct(self, exposure_time: Optional[float]) -> float:
        """
        Set the hardware exposure time without updating the frame rate.

        :param exposure_time: The target exposure time in seconds.
        :return: The actual set exposure time in seconds.
        """
        with self._lock:
            if self.__id is not None:
                if exposure_time is None:
                    # Get the maximum exposure for this frame rate
                    max_exposure_c = ueye.double()
                    IDSCamError.check(ueye.is_Exposure(self.__id, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX,
                                                       max_exposure_c, ctypes.sizeof(max_exposure_c)))
                    exposure_time = max_exposure_c.value * 1e-3

                # min_frame_time_c = ueye.double()
                # max_frame_time_c = ueye.double()
                # frame_time_step_c = ueye.double()
                # IDSCamError.check(ueye.is_GetFrameTimeRange(self.__id, min_frame_time_c, max_frame_time_c, frame_time_step_c))
                # # t_range = np.arange(min_c.value, max_c.value + step_c.value, step_c.value)  # in seconds
                # log.info(f'is_GetFrameTimeRange {min_frame_time_c.value*1e3:0.3f} ms to {max_frame_time_c.value*1e3:0.3f} ms.')
                # if self.__hardware_exposure_time_target is None:
                #     exposure_time = max_frame_time_c.value
                #     log.info(f'Setting maximum exposure: {exposure_time*1e3:0.3f} ms')
                # exposure_time = np.clip(exposure_time, 0.0, max_frame_time_c.value)

                # default_exposure_c = ueye.double()
                # IDSCamError.check(ueye.is_Exposure(self.__id, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_DEFAULT,
                #                                    default_exposure_c, ctypes.sizeof(default_exposure_c)))
                # log.info(f'Default exposure {default_exposure_c.value*1e3:0.3f} ms')

                # else:
                #     exposure_time = np.clip(exposure_time, min_c.value, max_c.value)
                # Set exposure
                log.debug(f'Setting exposure to {exposure_time*1e3:0.3f} ms')
                exposure_value_c = ueye.c_double(exposure_time * 1e3)  # convert from ms to seconds
                IDSCamError.check(ueye.is_Exposure(self.__id, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
                                                   exposure_value_c, ueye.sizeof(exposure_value_c)))

                # if self.__hardware_frame_time is None:
                #     max_shutter_c = ctypes.c_double()
                #     IDSCamError.check(ueye.is_SetAutoParameter(self.__id, ueye.IS_GET_AUTO_SHUTTER_MAX,
                #                                            max_shutter_c, ctypes.c_double(0)))
                #     print(f'max_shutter: {max_shutter_c.value}')

                # # Reset the frame time
                # self.frame_time = currently_set_frame_time

                # Return new exposure
                return self._get_hardware_exposure_time()

    def _get_hardware_exposure_time(self) -> float:
        with self._lock:
            if self.__id is not None:
                exposure_value_c = ueye.c_double()
                IDSCamError.check(ueye.is_Exposure(self.__id, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
                                                   exposure_value_c, ueye.sizeof(exposure_value_c)))
                # log.info(f'Exposure: {exposure_value_c.value:0.3f} ms')
                return exposure_value_c.value * 1e-3  # convert from ms to s

    def _set_hardware_frame_time(self, frame_time: Optional[float]) -> float:
        """
        Set the hardware frame time (1 / frame rate) and update the exposure accordingly.

        :param frame_time: The target frame time interval in seconds.
        :return: The actual set frame time interval in seconds.
        """
        with self._lock:
            self.__hardware_frame_time_target = frame_time
            if frame_time is not None or self.__hardware_exposure_time_target is None:
                new_frame_time = self.__set_hardware_frame_time_direct(frame_time)
                # Set exposure time again (this may be more restricted), without altering the frame time
                self.__set_hardware_exposure_time_direct(self.__hardware_exposure_time_target)
            else:
                self.__set_hardware_frame_time_direct(0.0)  # Set as short as possible
                # Set the hardware exposure time and get the optimal frame time
                self._set_hardware_exposure_time(self.__hardware_exposure_time_target)
                new_frame_time = self._get_hardware_frame_time()

            return new_frame_time

    def __set_hardware_frame_time_direct(self, frame_time: Optional[float]) -> float:
        """
        Set the hardware frame time (1 / frame rate) without updating also the exposure.

        :param frame_time: The target frame time interval in seconds.
        :return: The actual set frame time interval in seconds.
        """
        with self._lock:
            self.__hardware_frame_time_target = frame_time
            if self.__id is not None:
                min_c = ueye.double()
                max_c = ueye.double()
                step_c = ueye.double()
                IDSCamError.check(ueye.is_GetFrameTimeRange(self.__id, min_c, max_c, step_c))
                # t_range = np.arange(min_c.value, max_c.value + step_c.value, step_c.value)  # in seconds
                # if frame_time is not None:
                #     frame_time = np.clip(frame_time, min_c.value, max_c.value)
                if frame_time is None or frame_time <= 0.0:
                    log.debug(f'Requested frame time: {frame_time} s')
                    frame_time = min_c.value
                    log.debug(f'Setting minimum frame time to {frame_time*1e3:0.3f} ms')
                else:
                    log.debug(f'Setting frame time to {frame_time*1e3:0.3f} ms')
                # Set inter-frame time
                frame_rate_c = ueye.c_double(1.0 / frame_time)
                new_frame_rate_c = ueye.double()
                IDSCamError.check(ueye.is_SetFrameRate(self.__id, frame_rate_c, new_frame_rate_c))
                new_frame_time = 1.0 / new_frame_rate_c.value
                log.debug(f'Inter-frame time: {new_frame_time*1e3:0.3f} ms')

                return new_frame_time

    def _get_hardware_frame_time(self) -> float:
        with self._lock:
            if self.__id is not None:
                frames_per_second_c = ueye.c_double(0)
                IDSCamError.check(ueye.is_SetFrameRate(self.__id, ueye.IS_GET_FRAMERATE, frames_per_second_c),
                                  'Could not get frames per second.')
                frames_per_second = frames_per_second_c.value
                return 1.0 / frames_per_second

    def _start_continuous(self):
        with self._lock:
            if self.__id is not None:
                # Unlock all buffers
                for mem_buffer in self.__memory_buffers:
                    tmp = mem_buffer.array
                # IDSCamError.check(ueye.is_InitImageQueue(self.__id, 0),
                #                   'Could not enable queue mode for existing image memory sequences.')
                log.debug('Starting continuous video capture...')
                IDSCamError.check(ueye.is_CaptureVideo(self.__id, ueye.IS_DONT_WAIT), 'Could not start live video.')
                log.info('Started continuous video capture.')
                self.__acquiring = True

    def _stop_continuous(self):
        with self._lock:
            if self.__id is not None:
                log.debug('Stopping continuous video capture...')
                IDSCamError.check(ueye.is_StopLiveVideo(self.__id, ueye.IS_DONT_WAIT), 'Could not stop live video.')
                log.debug('Stopped continuous video capture.')
                # IDSCamError.check(ueye.is_ExitImageQueue(self.__id),
                #                   'Could not exit image queue.')
                self.__acquiring = False

    def __print_status(self):
        with self._lock:
            if self.__id is not None:
                # Replace ueye.IS_GET_STATUS by any other value to reset counters
                dropped_frames = self.__get_dropped_frames()
                sequence_count = ueye.is_CameraStatus(self.__id, ueye.IS_SEQUENCE_CNT, ueye.IS_GET_STATUS)
                buffer_size = ueye.is_CameraStatus(self.__id, ueye.IS_SEQUENCE_SIZE, ueye.IS_GET_STATUS)
                missed_triggers = self.__get_missed_triggers()
                log.info(f'Buffer (of size {buffer_size}). Dropped {dropped_frames}/{sequence_count} images due to USB bus congestion. Missed {missed_triggers} triggers.')

    def __get_dropped_frames(self) -> int:
        with self._lock:
            if self.__id is not None:
                # Replace ueye.IS_GET_STATUS by any other value to reset counters
                return ueye.is_CameraStatus(self.__id, ueye.IS_FIFO_OVR_CNT, ueye.IS_GET_STATUS)

    def __get_missed_triggers(self) -> int:
        with self._lock:
            if self.__id is not None:
                # Replace ueye.IS_GET_STATUS by any other value to reset counters
                return ueye.is_CameraStatus(self.__id, ueye.IS_TRIGGER_MISSED, ueye.IS_GET_STATUS)

    def _acquire(self):
        nb_trials = 3
        if self.__get_dropped_frames() > 0 or self.__get_missed_triggers() > 0:
            self.__print_status()

        with self._lock:
            image_data = None
            if self.__id is not None:
                for trial in range(nb_trials):
                    try:
                        if not self.__acquiring:
                            IDSCamError.check(ueye.is_FreezeVideo(self.__id, ueye.IS_DONT_WAIT),
                                              'Could not trigger snapshot.')
                        timeout_c = ctypes.c_uint32(10 * 1000)  # in milliseconds
                        mem_ptr_c = ueye.mem_p()  # A ctypes.c_void_p with mixin
                        next_image_id_c = ctypes.c_int32()
                        start_time = time.perf_counter()
                        IDSCamError.check(ueye.is_WaitForNextImage(self.__id, timeout_c, mem_ptr_c, next_image_id_c),
                                          'Error while waiting for image.')
                        next_image_id = next_image_id_c.value
                        log.debug(f'Waited {(time.perf_counter() - start_time)*1e3:0.3f} ms for image {next_image_id}.')
                        log.debug(f'Next image is {next_image_id} / {len(self.__memory_buffers)}.')
                        for mem_buf in self.__memory_buffers:
                            if next_image_id == mem_buf.id:
                                image_data = mem_buf.array
                                IDSCamError.check(ueye.is_GetImageMem(self.__id, mem_buf.ptr),
                                                  'Could not mark as read.')
                                break
                        break
                    except CamError as exc:
                        if trial + 1 >= nb_trials:
                            raise exc
                        log.warning(f'Dropping frames because {exc}!')
                        IDSCamError.check(ueye.is_InitImageQueue(self.__id, 0),
                                          'Could not re-enable queue mode for existing image memory sequences.')

            if image_data is None:
                log.warning('IDSCam could not capture the frame, defaulting to all zeros!')
                image_data = np.zeros(self.shape)

            return image_data

    def _disconnect(self):
        with self._lock:
            if self.__id is not None:
                if len(self.__memory_buffers) > 0:
                    try:
                        IDSCamError.check(ueye.is_StopLiveVideo(self.__id, ueye.IS_FORCE_VIDEO_STOP),
                                          'Could not stop live video')
                        IDSCamError.check(ueye.is_ClearSequence(self.__id),
                                          'Could not clear sequence after stopping live video.')
                    except CamError as ce:
                        pass
                    IDSCamError.check(ueye.is_ExitImageQueue(self.__id),
                                      'Could not disable queue mode for existing image memory sequences.')
                    try:
                        for mem_buf in self.__memory_buffers:
                            IDSCamError.check(ueye.is_FreeImageMem(self.__id, mem_buf.ptr, mem_buf.id))
                        self.__memory_buffers = []
                    except CamError as ce:
                        pass
                try:
                    IDSCamError.check(ueye.is_CameraStatus(self.__id, ueye.IS_STANDBY, True))
                except CamError as ce:
                    log.info(f'Could not set camera in standby mode: {ce}')
                try:
                    IDSCamError.check(ueye.is_ExitCamera(self.__id))
                except CamError as ce:
                    pass
                finally:
                    self.__id = None

    def __str__(self):
        return f'IDSCam(serial={self.serial}, model={self.model})'
