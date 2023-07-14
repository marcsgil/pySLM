import numpy as np
from typing import Union, Sequence, List
from pathlib import Path
import sys
import os
from pylablib.devices import Thorlabs

import logging
log = logging.getLogger(__name__)
from .stage import Stage, Translation


class ThorlabsKinesisStage(Stage):
    """"
    A class to control Thorlabs Kinesis motorized stages MTS25-Z8 and MTS50E-Z8 through the Thorlabs Kinesis DOTNET libraries.

    This requires the installation of the Thorlabs Kinesis software in `Program Files\Thorlabs\Kinesis`.
    It can be downloaded from https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=Motion_Control&viewtab=0
    Note that this is not compatible with the older APT libraries.
    """
    def __init__(self, thorlabs_kinesis_folder: Union[None, str, Path] = None, serial: str = None, home: bool = True):
        """
        Represents a Thorlabs Kinesis Stage.

        :param thorlabs_kinesis_folder: (optional) The folder where to find the DOTNET libraries (such as
        `Thorlabs.MotionControl.KCube.DCServo.dll`). Default: `Program Files\Thorlabs\Kinesis`
        :param serial: (optional) The serial number of the stage as a character string. Default: the first detected stage.
        :param home: (optional) Whether to home the stage on initialization. Default: True
        """
        max_acceleration = 1.5e-3  # m/s^2
        self.__poll_interval = 100e-3

        if thorlabs_kinesis_folder is None:
            thorlabs_kinesis_folder = os.environ["ProgramFiles"] / Path('Thorlabs/Kinesis')
        sys.path.append(str(thorlabs_kinesis_folder))  # Make sure we can load the Thorlabs Kinesis .NET dlls
        import clr
        from System import Decimal  #,  String, Action
        # Add references so Python can see .NET
        clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
        import Thorlabs.MotionControl.DeviceManagerCLI as diCLI
        clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
        import Thorlabs.MotionControl.GenericMotorCLI as GenericMotorCLI
        import Thorlabs.MotionControl.GenericMotorCLI.Settings as gmSettings
        clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
        import Thorlabs.MotionControl.KCube.DCServoCLI as DCServoCLI

        self.__forward = GenericMotorCLI.MotorDirection.Forward
        self.__backward = GenericMotorCLI.MotorDirection.Backward
        self.__to_decimal_mm = lambda _: Decimal(float(1e3 * _))  # Make sure that it is not a numpy scalar or array
        self.__from_decimal_mm = lambda _: Decimal.ToDouble(_) * 1e-3

        if serial is None:
            # Detect stages
            serial_numbers = self.find_instruments(thorlabs_kinesis_folder)

            # Pick the first KCubeDServo
            if len(serial_numbers) > 0:
                serial = serial_numbers[0]
            else:
                raise TypeError('No switched-on stages detected. Make sure that the Thorlabs Kinesis driver is installed, '
                                + 'that the stages are connected to both power and PC, and that the red switch on the '
                                + 'driver cube is toggled.')

        log.debug(f'Trying to connect to stage with serial number {serial}...')
        self.__device = DCServoCLI.KCubeDCServo.CreateKCubeDCServo(serial)
        self.__device.Connect(serial)
        if not self.__device.IsSettingsInitialized():
            log.debug('Waiting for settings init...')
            self.__device.WaitForSettingsInitialized(5000)

        device_info = self.__device.GetDeviceInfo()
        log.debug(f'Connected to Stage {device_info.Name} with serial number {device_info.SerialNumber}.')

        # Configure the motor
        motor_configuration = self.__device.LoadMotorConfiguration(serial)  #, DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)

        self.__device.EnableDevice()
        # time.sleep(0.500)  # Needs a delay to give time for the device to be enabled
        log.debug('Stage enabled.')

        unit_converter = self.__device.UnitConverter
        if unit_converter.RealUnits != 'mm':
            log.error(f'Real units of the stage should be mm, not {unit_converter.RealUnits}!')
            raise TypeError(f'Real units of the stage should be mm, not {unit_converter.RealUnits}!')
        conversion_factors = [unit_converter.GetFactor(_) for _ in range(2)]
        log.debug('Unit conversion: ' + str(conversion_factors))

        # Determine the travel range of this stage
        travel_range = [Decimal.ToDouble(
            unit_converter.DeviceUnitToReal(Decimal(_), GenericMotorCLI.DeviceUnitConverter.UnitType.Length)
        ) for _ in (self.__device.MotorPositionLimits.MinValue, self.__device.MotorPositionLimits.MaxValue)]
        travel_range = np.array(travel_range) * 1e-3
        log.debug(f'The stage can travel from {travel_range[0]*1e3:0.3f} mm to {travel_range[1]*1e3:0.3f} mm.')

        # time.sleep(0.500)
        mmi_params = self.__device.GetMMIParams()  # Man Machine Interface
        mmi_params.WheelMode = gmSettings.KCubeMMISettings.KCubeWheelMode.Velocity  # allow manual movements as before
        # mmi_params.DirectionSense = gmSettings.KCubeMMISettings.WheelDirectionTypes.Backward

        self.__device.SetMMIParams(mmi_params)

        self.__open = True

        super().__init__(ranges=travel_range, ndim=1, max_velocity=2.4e-3, max_acceleration=max_acceleration)

        # Initialized communication, setting max speed and homing.
        log.debug('Setting velocity...')
        self.__device.SetVelocityParams(self.__to_decimal_mm(self.max_velocity),
                                        self.__to_decimal_mm(self.max_acceleration))  # Velocity in mm/s and acceleration in mm/s^2

        if home and self.__device.NeedsHoming:
            self.home()

        self.__velocity = np.zeros(self.ndim)
        self.__translation = None

    @property
    def serial(self):
        with self._lock:
            if self.__open:
                info = self.__device.GetDeviceInfo()
                serial = str(info.SerialNumber)
            else:
                serial = ''
        return serial

    @property
    def position(self) -> np.ndarray:
        return np.atleast_1d(self.__from_decimal_mm(self.__device.Position))  # convert mm to meters

    @position.setter
    def position(self, new_position: Union[float, Sequence, np.ndarray]):
        new_position = np.atleast_1d(new_position)
        if new_position.size != self.ndim:
            raise TypeError(f'This stage has {self.ndim} axes, but the position is a {new_position.size}-element vector!')
        new_position = np.clip(new_position, self.ranges[:, 0], self.ranges[:, 1])
        distance = new_position - self.position
        time_difference = np.amax(np.abs(distance) / self.max_velocity)
        wait_time = np.maximum(2 * time_difference, 10.0)
        with self._lock:
            if self.__open:
                self.velocity = 0.0  # Make sure that the stage is not moving continuously at the moment
                self.__device.SetVelocityParams(
                    self.__to_decimal_mm(self.max_velocity[0]), self.__to_decimal_mm(self.max_acceleration))
                self.__device.SetMoveAbsolutePosition(self.__to_decimal_mm(new_position[0]))  # Convert meters to mm
                self.__device.MoveAbsolute(int(wait_time*1e3))  # timeout in ms

    @property
    def velocity(self):
        return self.__velocity

    @velocity.setter
    def velocity(self, new_velocity: Union[float, Sequence, np.ndarray]):
        new_velocity = np.atleast_1d(new_velocity)
        if new_velocity.size != self.ndim:
            raise TypeError(f'This stage has {self.ndim} axes, but the velocity is a {new_velocity.size}-element vector!')
        new_velocity = np.minimum(new_velocity, self.max_velocity)
        with self._lock:
            if self.__open:
                if not np.allclose(new_velocity, 0.0):
                    self.__device.SetVelocityParams(
                        self.__to_decimal_mm(np.abs(new_velocity[0])), self.__to_decimal_mm(self.max_acceleration))
                    log.debug(f'Polling stage every {int(self.__poll_interval*1e3)} ms...')
                    self.__device.StartPolling(int(self.__poll_interval*1e3))
                    log.info(f'Translating with speed {new_velocity*1e6} um/s...')
                    self.__device.MoveContinuous(self.__forward if new_velocity[0] > 0 else self.__backward)
                else:
                    log.debug('Stopping stage now!')
                    self.__device.StopImmediate()
                    self.__device.StopPolling()
        self.__velocity = new_velocity

    def translate(self, velocity: Union[None, float, Sequence, np.ndarray] = None,
                  origin: Union[None, float, Sequence, np.ndarray] = None,
                  destination: Union[None, float, Sequence, np.ndarray] = None) -> Translation:
        """
        Translate the stage.
        Usage:
        with self.translation(1e-3) as t:
            while t.position < 10e-3:
                print(t.position)

        :param velocity: The translation velocity in m/s (default: the maximum velocity for this stage).
        :param origin: (optional) The origin position in meters (default: the current position).
        :param destination: (optional) The destination in meters (default: the end of the stage's range).
        :return: A context manager object for the stage Translation.
        """
        if velocity is None:
            velocity = self.max_velocity
            if origin is not None and destination is not None:
                direction = destination - origin > 0
                velocity = velocity * (2 * direction - 1)
        else:
            velocity = np.atleast_1d(velocity)
            velocity = np.minimum(self.max_velocity, np.abs(velocity)) * np.sign(velocity)
        # Make sure that there is only one Translation object
        if self.__translation is not None:
            self.__translation.stop()
        self.__translation = Translation(self, self.max_acceleration, velocity=velocity,
                                         origin=origin, destination=destination)
        return self.__translation

    def home(self):
        log.debug('Homing...')
        with self._lock:
            if self.__open:
                self.__device.Home(60000)  # timeout in ms

    def __str__(self) -> str:
        with self._lock:
            if self.__open:
                info = self.__device.GetDeviceInfo()
                desc = info.Description
            else:
                desc = 'Stage Shutdown'
        return f'ThorlabsKinesisStage {desc} with serial number {self.serial} and travel range {str(self.ranges)}'

    def _disconnect(self):
        with self._lock:
            if self.__open:
                log.info(f'Closing stage {str(self)}.')
                self.__device.StopImmediate()
                self.__device.StopPolling()
                self.__device.ShutDown()
                log.info('Closed stage.')
                self.__open = False

    @staticmethod
    def find_instruments(self, thorlabs_kinesis_folder: Union[None, str, Path] = None) -> List:
        if thorlabs_kinesis_folder is None:
            thorlabs_kinesis_folder = os.environ["ProgramFiles"] / Path('Thorlabs/Kinesis')

        sys.path.append(str(thorlabs_kinesis_folder))  # Make sure we can load the Thorlabs Kinesis .NET dlls
        import clr
        # Add references so Python can see .NET
        clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
        import Thorlabs.MotionControl.DeviceManagerCLI as diCLI
        clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
        import Thorlabs.MotionControl.KCube.DCServoCLI as DCServoCLI

        # Detect stages
        device_list_result = diCLI.DeviceManagerCLI.BuildDeviceList()
        serial_numbers = diCLI.DeviceManagerCLI.GetDeviceList(DCServoCLI.KCubeDCServo.DevicePrefix)  #27 == DevicePrefix for Brushed KCube DCServo

        return serial_numbers



