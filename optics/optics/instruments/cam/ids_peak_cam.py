from typing import Union
import logging
log = logging.getLogger(__name__)

try:
    from ids_peak import ids_peak, ids_peak_ipl_extension   # This needs to be installed with the wheel that comes with the IDS download.
    from ids_peak_ipl import ids_peak_ipl
except ModuleNotFoundError as exc:
    log.warning(f"{exc}\nCould not find ids_peak Python module. Make sure that "
                + "(1) the drivers are installed and the camera can be opened in IDS Peak Cockpit, and "
                + "(2) that the module itself is installed, typically as a wheel using something like"
                + r" 'pip install C:\Program Files\IDS\ids_peak\generic_sdk\api\binding\python\wheel\x86_64\ids_peak....whl' "
                + "and 'pip install C:\Program Files\IDS\ids_peak\generic_sdk\ipl\binding\python\wheel\x86_64\ids_peak_idl....whl'.")

import ids_peak, ids_peak_ipl

import numpy as np
from typing import Union, Optional, List, Callable

from .cam import Cam, CamError, CamDescriptor
from optics.utils.roi import Roi


class IDSCamDescriptor(CamDescriptor):
    def __init__(self, id: str, constructor: Callable, available: bool = True, serial: str = '', model: str = '', index: int = -1):
        super().__init__(id, constructor, available, serial, model)
        self.__index: int = index

    @property
    def index(self) -> int:
        """The index of this camera."""
        return self.__index


class IDSCamError(CamError):
    pass


class IDSPeakCamDescriptor(IDSCamDescriptor):
    def __init__(self, id: str, constructor: Callable, available: bool = True, serial: str = '', model: str = '',
                 index: int = -1, name: str = '', interface: str = '', system: str = '', version: str = ''):
        super().__init__(id, constructor, available, serial, model, index)
        self.__name: str = name
        self.__interface: str = interface
        self.__system: str = system
        self.__version: str = version


class IDSPeakCam(Cam):
    """
    A class to control the IDS cameras using the newer IDS Peak SDK.
    """
    # initialize library
    log.info('Initializing IDS Peak library...')
    ids_peak.Library.Initialize()
    __ids_peak_version = ids_peak.Library.Version()

    # create a device manager object
    __device_manager = ids_peak.DeviceManager.Instance()

    @classmethod
    def list(cls, recursive: bool = False, include_unavailable: bool = False) -> List[Union[IDSPeakCamDescriptor, List]]:
        """
        Return constructor information for all cameras.

        :return: A dictionary with as key the class and as value, a dictionary with subclasses.
        """
        camera_descriptors = []

        device_access_type_control = 3  #Control = 3, ReadOnly = 2, Exclusive = 4; Custom = 1000
        cls.__device_manager.Update()
        for _, device in enumerate(cls.__device_manager.Devices()):
            name = device.ModelName()
            serial = device.SerialNumber()
            model = device.ModelName()
            interface = device.ParentInterface().DisplayName()
            system = device.ParentInterface().ParentSystem().DisplayName()
            version = device.ParentInterface().ParentSystem().Version()
            available = device.IsOpenable(device_access_type_control)

            if include_unavailable or available:
                camera_descriptors.append(
                    IDSPeakCamDescriptor(id=f'IDSPeakCam{_}_{serial}_{model}',
                                         constructor=lambda: IDSPeakCam(index=_, serial=serial),
                                         available=available, serial=serial, model=model, index=_, name=name,
                                         interface=interface, system=system, version=version)
                )

        return camera_descriptors

    def __init__(self, index: Optional[int] = None, serial: Union[int, str, None] = None, model: Optional[str] = None,
                 normalize=True, exposure_time: Optional[float] = None, frame_time: Optional[float] = None,
                 gain: Optional[float] = None, black_level: Optional[float] = None):
        super().__init__(normalize=normalize)  # Another __init__ below! This is just to initialize self._lock

        cam_descriptors = IDSPeakCam.list()
        log.debug(f'Available cameras:\n{cam_descriptors}')

        if serial is not None:
            if isinstance(serial, int):
                serial = f'{serial:0.0f}'
            cam_descriptors = [info for info in cam_descriptors if info.serial == serial]
            if len(cam_descriptors) < 1:
                raise ValueError(f'No camera available with serial number {serial}.')
        if model is not None:
            cam_descriptors = [info for info in cam_descriptors if info.model.lower() == model.lower()]
            if len(cam_descriptors) < 1:
                raise ValueError(f'No camera available of model {model}.')
        if index is not None and index >= 0:
            cam_descriptors = [info for info in cam_descriptors if info.index == index]
            if len(cam_descriptors) < 1:
                raise ValueError(f'No camera available with index {index}.')
        # Select the first camera that satisfies all criteria
        if len(cam_descriptors) > 1:
            log.warning(f'Multiple cameras available that fit the requirements: {cam_descriptors}')
            indices = [info.index for info in cam_descriptors]
            cam_descriptors = [info for info in cam_descriptors if info.index == min(indices)]

        cam_descriptor = cam_descriptors[0]
        log.debug(f'Selected camera: {cam_descriptor}')

        self.__index = cam_descriptor.index
        self.__model = cam_descriptor.model
        self.__serial = cam_descriptor.serial

        # open selected device
        self.__device_manager.Update()
        self.__device = self.__device_manager.Devices()[self.__index].OpenDevice(ids_peak.DeviceAccessType_Control)
        self.__nmd = self.__device.RemoteDevice().NodeMaps()[0]

        # # Getting sensor shape
        # w_max = self.__nmd.FindNode('Width').Maximum()
        # h_max = self.__nmd.FindNode('Height').Maximum()
        # max_shape = np.array([h_max, w_max])

        # Get a list of all available entries of SensorShutterMode
        try:
            all_entries = self.__nmd.FindNode('SensorShutterMode').Entries()
            available_entries = [_.SymbolicValue() for _ in all_entries
                                 if _.AccessStatus() != ids_peak.NodeAccessStatus_NotAvailable
                                 and _.AccessStatus() != ids_peak.NodeAccessStatus_NotImplemented]
        except ids_peak.NotFoundException:
            available_entries = []
        self.__global_shutter_supported = 'Global' in available_entries
        try:
            self.__nmd.FindNode('SensorShutterMode').SetCurrentEntry('Global')
        except (ids_peak.NotFoundException, ids_peak.BadAccessException) as exc:
            self.__global_shutter_supported = False
            log.warning(f'Could not set global shutter! {exc}')

        # Open standard data stream
        datastreams = self.__device.DataStreams()
        if len(datastreams) <= 0:
            raise IOError(f'Camera with serial number {self.__serial} and model {self.__model} has no datastream!')
        self.__datastream = datastreams[0].OpenDataStream()

        try:
            self.__nmd.FindNode('ExposureAuto').SetCurrentEntry('Off')  # or Once or Continuous
        except ids_peak.Exception:
            log.warning(f'Setting auto-exposure is not possible for camera with serial number {self.__serial} and model {self.__model}.')
        try:
            self.__nmd.FindNode('GainAuto').SetCurrentEntry('Off')
        except ids_peak.Exception:
            log.warning(f'Setting auto-gain is not possible for camera with serial number {self.__serial} and model {self.__model}.')
        try:
            # To prepare for untriggered continuous image acquisition, load the default user set if available and wait until execution is finished
            self.__nmd.FindNode('UserSetSelector').SetCurrentEntry('Default')
            self.__nmd.FindNode('UserSetLoad').Execute()
            self.__nmd.FindNode('UserSetLoad').WaitUntilDone()
        except ids_peak.Exception:
            log.warning(f'UserSet is not available for camera with serial number {self.__serial} and model {self.__model}.')

        self.__gain = None
        self.__hardware_exposure_time_target = None  # Longest exposure time for the frame rate
        self.__hardware_frame_time_target = None  # Fastest frame rate for given exposure time
        self.__streaming = False
        self.__acquiring = False
        self.trigger = False
        self.bit_depth = 8  # default to a bit depth of 8 bits
        self.global_shutter = True
        self.pixel_clock = np.inf
        self._set_hardware_bin(1)
        self.__auto_exposure = False
        if exposure_time is not None:
            self.exposure_time = exposure_time
        self.frame_time = frame_time if frame_time is not None else 0.0
        if gain is not None:
            self.gain = gain
        if black_level is not None:
            self.black_level = black_level

        # Determine the current entry of PixelFormat (str)
        value = self.__nmd.FindNode('PixelFormat').CurrentEntry().SymbolicValue()
        # Get a list of all available entries of PixelFormat
        all_modes = self.__nmd.FindNode('PixelFormat').Entries()
        available_modes = []
        for entry in all_modes:
            if (entry.AccessStatus() != ids_peak.NodeAccessStatus_NotAvailable
                    and entry.AccessStatus() != ids_peak.NodeAccessStatus_NotImplemented):
                available_modes.append(entry.SymbolicValue())
        mono_modes = [_ for _ in available_modes if _.lower().startswith('mono')]
        color = len(mono_modes) < len(available_modes)

        if self.color:
            self.__nmd.FindNode('BalanceWhiteAuto').SetCurrentEntry('Off')

        try:
            pixel_pitch = np.array([self.__nmd.FindNode('SensorPixelHeight').Value(),
                                    self.__nmd.FindNode('SensorPixelWidth').Value()])
        except ids_peak.NotFoundException as exc:
            log.warning('Could not detect physical pixel size.')
            pixel_pitch = [1, 1]

        self.__create_buffer()  # This changes the maximum shape of the ROI!

        w_max = self.__nmd.FindNode('Width').Maximum()
        h_max = self.__nmd.FindNode('Height').Maximum()
        max_shape = np.array([h_max, w_max])
        # The super class sets the roi and thereby starts continuous acquisition
        super().__init__(shape=max_shape, normalize=normalize, color=color, pixel_pitch=pixel_pitch,
                         exposure_time=self.exposure_time, frame_time=self.frame_time, gain=self.gain)

    # @property
    # def shape(self) -> np.ndarray:
    #     """
    #     The physical shape of this sensor array sets a maximum for the region-of-interest.
    #     """
    #     with self._lock:
    #         w_max = self.__nmd.FindNode('Width').Maximum()
    #         h_max = self.__nmd.FindNode('Height').Maximum()
    #         return np.array([h_max, w_max])

    def __create_buffer(self):
        # Flush queue and prepare all buffers for revoking
        self.__datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        # Clear all old buffers
        for buffer in self.__datastream.AnnouncedBuffers():
            self.__datastream.RevokeBuffer(buffer)

        payload_size = self.__nmd.FindNode('PayloadSize').Value()

        # Get minimum number of required buffers
        num_buffers_min_required = self.__datastream.NumBuffersAnnouncedMinRequired()

        # Allocate new buffers
        for count in range(num_buffers_min_required):
            buffer = self.__datastream.AllocAndAnnounceBuffer(payload_size)
            self.__datastream.QueueBuffer(buffer)

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
    def temperature(self) -> float:
        """The temperature of the camera in Kelvin"""
        try:
            return self.__nmd.FindNode('DeviceTemperature').Value() + 273.15
        except ids_peak.NotFoundException as exc:
            log.debug('Temperature could not be detected.')
            return 1000 + 273.15

    @property
    def bit_depth(self) -> int:
        # # Determine the current entry of PixelSize (str)
        # bit_per_pixel_description = self.__nmd.FindNode('PixelSize').CurrentEntry().SymbolicValue()
        # total_bits = int(re.search(r'Bpp(\d+)', bit_per_pixel_description).group(1))
        # # Get a list of all available entries of PixelSize
        # all_entries = self.__nmd.FindNode('PixelSize').Entries()
        # bits_per_pixel_options = [_.SymbolicValue() for _ in all_entries
        #                           if _.AccessStatus() != ids_peak.NodeAccessStatus_NotAvailable
        #                           and _.AccessStatus() != ids_peak.NodeAccessStatus_NotImplemented
        #                           ]
        # log.debug(f'Bits per pixel options {bits_per_pixel_options}. Current: {self.__bit_depth}')
        with self._lock:
            return self.__bit_depth

    @bit_depth.setter
    def bit_depth(self, new_bit_depth: int):
        with self._lock:
            if self.color:
                log.warning('Color mode not implemented yet!')
                self.__nmd.FindNode('PixelFormat').SetCurrentEntry(f'Mono{new_bit_depth:0.0f}')
            else:
                self.__nmd.FindNode('PixelFormat').SetCurrentEntry(f'Mono{new_bit_depth:0.0f}')
            self.__bit_depth = new_bit_depth
            # Save also the pixel format for conversion
            if new_bit_depth == 8:
                self.__pixel_format = ids_peak_ipl.PixelFormatName_Mono8
            elif new_bit_depth == 10:
                self.__pixel_format = ids_peak_ipl.PixelFormatName_Mono10
            elif new_bit_depth == 12:
                self.__pixel_format = ids_peak_ipl.PixelFormatName_Mono12

    @property
    def gain(self) -> float:
        """Get the gain as a value between 0 and 1."""
        with self._lock:
            gain_min = self.__nmd.FindNode('Gain').Minimum()
            gain_max = self.__nmd.FindNode('Gain').Maximum()
            return (self.__nmd.FindNode('Gain').Value() - gain_min) / (gain_max - gain_min)

    @gain.setter
    def gain(self, gain: float):
        """Set the gain as a value between 0 and 1."""
        try:
            with self._lock:
                if self.__device is not None:
                    gain_min = self.__nmd.FindNode('Gain').Minimum()
                    gain_max = self.__nmd.FindNode('Gain').Maximum()
                    self.__nmd.FindNode('Gain').SetValue(gain_min + (gain_max - gain_min) * np.clip(gain, 0, 1))
        except IDSCamError:  # todo
            pass

    @property
    def global_shutter(self) -> bool:
        # Determine the current entry of SensorShutterMode (str)
        value = self.__nmd.FindNode('SensorShutterMode').CurrentEntry().SymbolicValue()

        return value == 'Global'

    @global_shutter.setter
    def global_shutter(self, state: bool):
        if self.__global_shutter_supported:
            if state:
                self.__nmd.FindNode('SensorShutterMode').SetCurrentEntry('Global')
            else:
                self.__nmd.FindNode('SensorShutterMode').SetCurrentEntry('Rolling')

            # Set frame rate (and exposure time) again
            self._set_hardware_frame_time(self.__hardware_frame_time_target)

    @property
    def black_level(self) -> float:
        """Returns a value between 0 and 1."""
        min_value = self.__nmd.FindNode('BlackLevel').Minimum()
        max_value = self.__nmd.FindNode('BlackLevel').Maximum()
        hardware_value = self.__nmd.FindNode('BlackLevel').Value()
        return (hardware_value - min_value) / (max_value - min_value)

    @black_level.setter
    def black_level(self, value: float):
        """Set a value between 0 and 1."""
        min_value = self.__nmd.FindNode('BlackLevel').Minimum()
        max_value = self.__nmd.FindNode('BlackLevel').Maximum()
        hardware_value = min_value + (max_value - min_value) * np.clip(value, 0.0, 1.0)

        self.__nmd.FindNode('BlackLevel').SetValue(hardware_value)

        # Set frame rate (and exposure time) again
        self._set_hardware_frame_time(self.__hardware_frame_time_target)

    @property
    def trigger(self) -> bool:
        self.__nmd.FindNode('TriggerSelector').SetCurrentEntry('ExposureStart')
        return self.__nmd.FindNode('TriggerMode').CurrentEntry().SymbolicValue() == 'On'

    @trigger.setter
    def trigger(self, hardware: bool):
        # # Single frame acquisition configurations
        # self.__nmd.FindNode('AcquisitionMode').SetCurrentEntry('SingleFrame')
        if hardware:
            # Load UserSet "Default", if the configuration of the camera is unclear
            self.__nmd.FindNode('UserSetSelector').SetCurrentEntry('Default')
            self.__nmd.FindNode('UserSetLoad').Execute()

            # Activate the ExposureStart trigger and configure its source to an IO line
            self.__nmd.FindNode('TriggerSelector').SetCurrentEntry('ExposureStart')
            self.__nmd.FindNode('TriggerMode').SetCurrentEntry('On')
            self.__nmd.FindNode('TriggerSource').SetCurrentEntry('Line0')

            # Use the falling edge of Line0 to activate the trigger
            self.__nmd.FindNode('TriggerActivation').SetCurrentEntry('FallingEdge')

            # Here without ExposureStart trigger, replace it by software or hardware trigger if necessary
            self.__nmd.FindNode('TriggerSelector').SetCurrentEntry('ExposureStart')
            self.__nmd.FindNode('TriggerMode').SetCurrentEntry('Off')
            log.debug('Hardware trigger set.')
        else:
            # Here without ExposureStart trigger, replace it by software or hardware trigger if necessary
            self.__nmd.FindNode('TriggerSelector').SetCurrentEntry('ExposureStart')
            self.__nmd.FindNode('TriggerMode').SetCurrentEntry('Off')
            log.debug('Software trigger set.')

        # Set frame rate (and exposure time) again
        self._set_hardware_frame_time(self.__hardware_frame_time_target)

    @property
    def pixel_clock(self) -> float:
        return self.__nmd.FindNode('DeviceClockFrequency').Value()

    @pixel_clock.setter
    def pixel_clock(self, new_frequency: Optional[float]):
        if new_frequency is None:
            new_frequency = self.__nmd.FindNode('DeviceClockFrequency').Maximum()
        try:
            new_frequency = np.clip(new_frequency,
                                    self.__nmd.FindNode('DeviceClockFrequency').Minimum(),
                                    self.__nmd.FindNode('DeviceClockFrequency').Maximum())
            log.info(f'Trying to set device clock frequency to {new_frequency}...')
            self.__nmd.FindNode('DeviceClockFrequency').SetValue(new_frequency)

            # Set frame rate (and exposure time) again
            self._set_hardware_frame_time(self.__hardware_frame_time_target)
        except ids_peak.BadAccessException as exc:
            log.warning(f'Could not set clock frequency: {exc}')

    def _set_hardware_roi(self, minimum_roi: Roi) -> Roi:
        with self._lock:
            set_roi = minimum_roi
            if self.__device is not None:
                # Determine the new region of interest, compatible with hardware
                if minimum_roi is None:
                    minimum_roi = Roi(shape=self.shape)

                # Increase the region of interest to a valid one
                # Get the minimum ROI
                x_min = self.__nmd.FindNode('OffsetX').Minimum()
                y_min = self.__nmd.FindNode('OffsetY').Minimum()
                w_min = self.__nmd.FindNode('Width').Minimum()
                h_min = self.__nmd.FindNode('Height').Minimum()

                # Set the minimum ROI. This removes any size restrictions due to previous ROI settings
                self.__nmd.FindNode('OffsetX').SetValue(x_min)
                self.__nmd.FindNode('OffsetY').SetValue(y_min)
                self.__nmd.FindNode('Width').SetValue(w_min)
                self.__nmd.FindNode('Height').SetValue(h_min)

                # Get the maximum ROI values
                x_max = self.__nmd.FindNode('OffsetX').Maximum()
                y_max = self.__nmd.FindNode('OffsetY').Maximum()
                w_max = self.__nmd.FindNode('Width').Maximum()
                h_max = self.__nmd.FindNode('Height').Maximum()

                # Get the increment
                x_inc = self.__nmd.FindNode('OffsetX').Increment()
                y_inc = self.__nmd.FindNode('OffsetY').Increment()
                w_inc = self.__nmd.FindNode('Width').Increment()
                h_inc = self.__nmd.FindNode('Height').Increment()

                # Clip the region of interest to the permitted
                top_left_min = [y_min, x_min]
                bottom_right_max = [min(y_max, y_min + h_max), min(x_max, x_min + w_max)]
                top_left = np.clip(minimum_roi.top_left, top_left_min, bottom_right_max)
                bottom_right = np.clip(minimum_roi.bottom_right, top_left_min, bottom_right_max)
                # increase roi to a permitted position and shape
                top_left_inc = np.array([y_inc, x_inc])
                shape_inc = np.array([h_inc, w_inc])
                top_left = np.minimum(top_left, (top_left // top_left_inc) * top_left_inc)
                shape = bottom_right - top_left
                shape = np.maximum(shape, ((shape + shape_inc - 1) // shape_inc) * shape_inc)  # round up
                shape = np.maximum(shape, [h_min, w_min])
                # create a new, larger, region of interest
                set_roi = Roi(top_left=top_left, shape=shape)
                # And set the valid region of interest
                self.__nmd.FindNode('OffsetX').SetValue(int(set_roi.left))
                self.__nmd.FindNode('OffsetY').SetValue(int(set_roi.top))
                self.__nmd.FindNode('Width').SetValue(int(set_roi.width))
                self.__nmd.FindNode('Height').SetValue(int(set_roi.height))

                if self.__device is None:
                    raise CamError("Camera closed.")

            return set_roi

    def _set_hardware_exposure_time(self, exposure_time: Optional[float] = None) -> Optional[float]:
        """
        Set the hardware exposure time and update the frame rate accordingly.

        :param exposure_time: The target exposure time in seconds.
        :return: The actual set exposure time in seconds.
        """
        with self._lock:
            self.__hardware_exposure_time_target = exposure_time
            if self.__device is not None:
                # Get the maximum exposure for this frame rate
                return self.__set_hardware_exposure_time_direct(exposure_time)

    def __set_hardware_exposure_time_direct(self, exposure_time: Optional[float]) -> float:
        """
        Set the hardware exposure time without updating the frame rate.

        :param exposure_time: The target exposure time in seconds.
        :return: The actual set exposure time in seconds.
        """
        with self._lock:
            if self.__device is not None:
                if exposure_time is None:
                    # Get the maximum exposure for this frame rate
                    exposure_time = np.inf

                # Set exposure
                log.debug(f'Setting exposure to a target of {exposure_time*1e3:0.3f} ms')
                exposure_time_min = self.__nmd.FindNode('ExposureTime').Minimum()
                exposure_time_max = self.__nmd.FindNode('ExposureTime').Maximum()
                exposure_time_scaled = np.clip(exposure_time / 1e-6, exposure_time_min, exposure_time_max)
                log.debug(f'Setting exposure to {exposure_time_scaled * 1e-6 * 1e3:0.3f} ms')
                self.__nmd.FindNode('ExposureTime').SetValue(exposure_time_scaled)  # todo: is this in microseconds?

                # Return new exposure
                return self._get_hardware_exposure_time()

    def _get_hardware_exposure_time(self) -> float:
        with self._lock:
            if self.__device is not None:
                return self.__nmd.FindNode('ExposureTime').Value() * 1e-6  # todo: is this in seconds, or is conversion needed?

    def _set_hardware_frame_time(self, frame_time: Optional[float] = None) -> float:
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

    def __set_hardware_frame_time_direct(self, frame_time: Optional[float] = 0.0) -> float:
        """
        Set the hardware frame time (1 / frame rate) without updating also the exposure.

        :param frame_time: The target frame time interval in seconds.
        :return: The actual set frame time interval in seconds.
        """
        with self._lock:
            self.__hardware_frame_time_target = frame_time
            if self.__device is not None:
                max_frame_rate = self.__nmd.FindNode('AcquisitionFrameRate').Maximum()
                min_frame_rate = self.__nmd.FindNode('AcquisitionFrameRate').Minimum()
                if frame_time is None or frame_time <= 0.0:
                    frame_time = 1.0 / max_frame_rate
                if np.isinf(frame_time):
                    frame_time = 1.0 / min_frame_rate

                frame_rate = np.clip(1.0 / frame_time, min_frame_rate, max_frame_rate)

                self.__nmd.FindNode('AcquisitionFrameRate').SetValue(frame_rate)

                return frame_time

    def _get_hardware_frame_time(self) -> float:
        with self._lock:
            if self.__device is not None:
                frames_per_second = self.__nmd.FindNode('AcquisitionFrameRate').Value()
                return 1.0 / frames_per_second

    def _start_continuous(self):
        with self._lock:
            if self.__device is not None:
                log.debug('Starting continuous video capture...')

                # Lock critical features to prevent them from changing during acquisition
                self.__nmd.FindNode('TLParamsLocked').SetValue(True)

                # Start acquisition on camera
                self.__datastream.StartAcquisition()
                self.__nmd.FindNode('AcquisitionStart').Execute()
                self.__nmd.FindNode('AcquisitionStart').WaitUntilDone()

                log.info('Started continuous video capture.')
                self.__streaming = True
                self.__acquiring = True

    def _stop_continuous(self):
        with self._lock:
            if self.__device is not None:
                log.debug('Stopping continuous video capture...')

                # Execute AcquisitionStop
                self.__nmd.FindNode('AcquisitionStop').Execute()
                # Check if the command has finished before you continue (optional)
                self.__nmd.FindNode('AcquisitionStop').WaitUntilDone()
                # Unlock critical features after acquisition
                self.__nmd.FindNode('TLParamsLocked').SetValue(False)
                log.debug('Stopped continuous video capture.')
                self.__acquiring = False
                self.__streaming = False

    def _acquire(self):
        nb_trials = 3
        with self._lock:
            image_data = None
            if self.__device is not None:
                for trial in range(nb_trials):
                    try:
                        if not self.__acquiring:
                            # Start a snapshot acquisition
                            # log.debug('Starting single shot acquisition.')
                            # self.__nmd.FindNode('AcquisitionMode').SetCurrentEntry('Continuous')   # SingleFrame
                            # # Here without ExposureStart trigger, replace it by software or hardware trigger if necessary
                            # self.__nmd.FindNode('TriggerSelector').SetCurrentEntry('ExposureStart')
                            # self.__nmd.FindNode('TriggerMode').SetCurrentEntry('Off')
                            self.__nmd.FindNode('TLParamsLocked').SetValue(1)
                            self.__datastream.StartAcquisition(ids_peak.AcquisitionStartMode_Default, 1)  #ids_peak.DataStream.INFINITE_NUMBER)
                            self.__nmd.FindNode('AcquisitionStart').Execute()
                            self.__acquiring = True
                        buffer = self.__datastream.WaitForFinishedBuffer(3000) # Todo
                        if not buffer.HasNewData():
                            log.error('Buffer has no data!')
                        if buffer.IsIncomplete():
                            log.error('Buffer is incomplete!')
                        # if buffer.HasChunks():
                        #     self.__nmd.UpdateChunkNodes(buffer)
                        #     exposure_time = self.__nmd.FindNode('ChunkExposureTime').Value()
                        #     timestamp = self.__nmd.FindNode('Timestamp').Value()
                        #     exposure_timestamp = self.__nmd.FindNode('ExposureTriggerTimestamp').Value()

                        # Create IDS peak IPL image for debayering and convert it to RGBa8 format
                        ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)

                        # ipl_image = ids_peak_ipl.Image_CreateFromSizeAndBuffer(
                        #     buffer.PixelFormat(),
                        #     buffer.BasePtr(),
                        #     buffer.Size(),
                        #     buffer.Width(),
                        #     buffer.Height()
                        # )
                        # ipl_image = ipl_image.ConvertTo(ids_peak_ipl.PixelFormatName_BGRa8, ids_peak_ipl.ConversionMode_Fast)

                        # Queue buffer so that it can be used again
                        self.__datastream.QueueBuffer(buffer)

                        # Stop the acquisition right away
                        if not self.__streaming:
                            self.__datastream.StopAcquisition()
                            self.__nmd.FindNode('AcquisitionStop').Execute()
                            self.__nmd.FindNode('TLParamsLocked').SetValue(False)
                            self.__acquiring = False

                        # Get raw image data from converted image and construct a QImage from it
                        if self.color:
                            image_data = ipl_image.get_numpy_3D()
                        else:
                            image_data = ipl_image.get_numpy_2D()
                        break
                    except CamError as exc:
                        if trial + 1 >= nb_trials:
                            raise exc
                        log.warning(f'Dropping frames because {exc}!')

            if image_data is None:
                log.warning('IDSPeakCam could not capture the frame, defaulting to all zeros!')
                image_data = np.zeros(self.shape)

            return image_data

    def _disconnect(self):
        with self._lock:
            if self.__device is not None:
                self.__nmd.FindNode('TLParamsLocked').SetValue(False)
                if self.__streaming:
                    self.__datastream.StopAcquisition()
                    self.__nmd.FindNode('AcquisitionStop').Execute()
                # Flush queue and prepare all buffers for revoking
                self.__datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                # Clear all old buffers
                for buffer in self.__datastream.AnnouncedBuffers():
                    self.__datastream.RevokeBuffer(buffer)

                # ids_peak.Library.Close()
                # self.__device_manager = None
                self.__device = None

    def __str__(self):
        return f'IDSPeakCam(serial={self.serial}, model={self.model})'
