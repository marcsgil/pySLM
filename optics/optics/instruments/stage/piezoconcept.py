from __future__ import annotations

from typing import Union, Pattern, Sequence, Callable
import serial.tools.list_ports
import numpy as np
import re
import time

import logging
import optics.instruments.stage
from optics.instruments.stage import Stage
from optics.utils.ft import Grid

log = logging.getLogger(__name__)


class Translation(optics.instruments.stage.Translation):
    """
    A ContextManager class to represent a stage translation.
    """
    def __init__(self, stage: PiezoConceptStage, max_acceleration: Union[None, float, Sequence, np.ndarray],
                 velocity: Union[None, float, Sequence, np.ndarray] = None,
                 origin: Union[None, float, Sequence, np.ndarray] = None,
                 destination: Union[None, float, Sequence, np.ndarray] = None):
        self.__stage = stage
        super().__init__(stage=stage, max_acceleration=max_acceleration,
                         velocity=velocity, origin=origin, destination=destination)

        self.__stage.translation_init(velocity=velocity, origin=origin, destination=destination)

    def start(self):
        """
        Start the stage translation. This is automatically triggered by the Context Manager.
        """
        self.__stage.translation_trigger()

    def stop(self):
        """
        Stop the stage translation. This is triggered the Context Manager when self.position is called and the
        destination is reached.
        """
        self.__stage.wait()
        super().stop()  # set the velocity on record to all 0s

    @property
    def position(self) -> np.ndarray:
        position = self.__stage.position

        distance_to_destination = np.linalg.norm(position - self.destination)
        if distance_to_destination < 0.25e-6:
            self.stop()
            position = self.__stage.position  # update again

        return position

    def __str__(self):
        return f'piezoconcept.{__class__.__name__}(velocity={self.velocity}, origin={self.origin}, destination={self.destination})'


class PiezoConceptStage(Stage):
    """
    A Class for controlling Piezoconcept Nanostage LT3.300 through the USB interface
    """

    def __init__(self, serial_port: Union[serial.Serial, str, None] = None, power_down_on_disconnect: bool = True):
        """
        Initializes the
        :param serial_port: The serial port if provided, if not, auto-detection is attempted.
        :param power_down_on_disconnect: If True, the power_down() method is called when disconnecting. Default: True
        """
        super().__init__()  # Initialize the super class, even though we are not connected yet.

        self.__encoding = 'latin1'
        self.__serial_port: serial.Serial = serial_port
        self._connect()

        infos = [self.__query('INFOS')]
        while len(infos[-1].strip()) != 0:
            infos.append(self.__query())
        infos = '\n'.join(infos)

        self.__name = re.search(r'product\s+:\s*\n(.+)\n', infos, re.MULTILINE | re.IGNORECASE).groups()[0]
        log.debug(f'Connected to {self.__name}.')
        # Set the range bounds from the returned info
        max_ranges = [self.__parse_measurement(
            re.search(r'^Travel\s+range\s*' + _ + r'\s*:\s*(.+)(\n|$)', infos, re.MULTILINE | re.IGNORECASE).groups()[0]
        ) for _ in 'XYZ']
        max_ranges = np.asarray(max_ranges)[:, np.newaxis]
        stage_range = np.concatenate((np.zeros(max_ranges.shape), max_ranges), axis=1)

        super().__init__(ranges=stage_range, power_down_on_disconnect=power_down_on_disconnect)
        self.__velocity = np.zeros(self.ndim)
        self.__translating = False
        self.__translation: Translation = None

        self.set_trigger(ttl_connector=4, output=True, axis=0)

    @property
    def name(self) -> str:
        return self.__name

    def _connect(self):
        # Should only be called from within with self._lock
        def open_port(port_device: str) -> serial.Serial:
            return serial.Serial(port=port_device, baudrate=115200, bytesize=8, parity=serial.PARITY_NONE, stopbits=1,
                                 timeout=100e-3)

        if isinstance(self.__serial_port, serial.Serial):
            if not self.__serial_port.isOpen():
                self.__serial_port.open()
        else:
            if isinstance(self.__serial_port, str):
                self.__serial_port = open_port(self.__serial_port)
                if self.__serial_port.isOpen():
                    log.info(f'Port {self.__serial_port} is open.')
                else:
                    raise ConnectionError(f'Could not open port {self.__serial_port}.')
            else:
                serial_port_infos = serial.tools.list_ports.comports(include_links=True)
                if self.__serial_port is None or isinstance(self.__serial_port, str):
                    for port_info in serial_port_infos:
                        log.info(f'Trying to find stage on port {port_info.description}...')
                        try:
                            self.__serial_port = open_port(port_info.device)
                            if self.__serial_port.isOpen():
                                log.info(f'Port {self.__serial_port} is open.')
                                break
                            else:
                                self.__serial_port = None
                        except serial.serialutil.SerialException as se:
                            log.info(f'Could not open port {port_info.description}.')

    def __query(self, command: str = None) -> str:
        with self._lock:
            # Commands:
            #  _RAZ_ : reset to 0V
            # INFOS: returns name LT3.300 and maximum travel 300 um etc
            # MOVRX, MOVRY, MOVRZ -30u
            # MOVEX, MOVEY, MOVEZ 150000n
            # GET_X, GET_Y, GET_Z
            # STIME 20u  for 20 microsecond update interval from USB interface RAM to nanopositioner
            # GTIME: get update interval
            # SWF_X|YZ: a 1-axis ramp waveform e.g. SWF_X 100 0u 100u (100 steps)
            # ARBWF: arbitrary waveform e.g. ARBWF 100 20 10  (100 data points for x, 20 for y and 10 for z)
            # ADDPX|YZ: add coordinate for above
            # RUNWF: run the above, one per GTIME
            # ARB3D: Prepare a 3D waveform, e.g. ARB3D 100, 100 steps
            # ADD3D 10u 20u 30u for above
            # RUN3D: run above
            # DISIO 1|2|3|4: Display the TTL port setup: input, position, Channel1, RisingEdge on different lines
            # CHAIO: set 4 TTL IO ports. Example:
            #       CHAIO 1n: disable port 1
            #       CHAIO 1i2r: set TTL1 as an input, triggering motion on the 2nd axis on the rising edge
            #       CHAIO 1i2f: set TTL1 as an input, triggering motion on the 2nd axis on the falling edge
            #       CHAIO 1o2s: set TTL1 as an output, providing pulse on start of 2nd axis motion
            #       CHAIO 1o2e: set TTL1 as an output, providing pulse on end of 2nd axis motion
            #       CHAIO 1o2n6: set TTL1 as an output, providing pulse at the start of step 6 of 2nd axis motion
            #       CHAIO 1o2g5-10: set TTL1 as an output, providing a gate pulse controlled by the 2nd axis motion,
            #                       TTL output turns on at step 5 and turns off at step 10
            self._connect()
            if command is not None:
                self.__serial_port.write((command + '\n').encode(self.__encoding))
                log.debug(f'Sent: {command} to PiezoConcept Nanostage serial interface.')
            prompt = b'\n'
            response = self.__serial_port.read_until(terminator=prompt)
            log.debug(f'Received: {response} from PiezoConcept Nanostage serial interface.')
            response = response[:-len(prompt)].decode(self.__encoding)

            return response

    def __command(self, command: str = None) -> bool:
        response = self.__query(command=command)
        if response.strip().lower() == 'ok':
            return True
        else:
            log.error(f"Expected 'Ok' from the stage's interface but received '{response}' instead.")

    @staticmethod
    def __parse_measurements(value_str: str) -> np.ndarray:
        values = []
        for match in re.finditer(r'(\d+\.?\d*|\d*\.?\d+)\s*([muμµn])?', value_str, re.MULTILINE):  # Note the two different mus
            if match.groups()[0] is None:
                raise ValueError(f'{value_str} does not appear to be a measurement.')
            value = float(match.groups()[0])
            unit = match.groups()[1]
            if unit is not None:  # parse unit if any
                if unit == 'm':
                    value *= 1e-3
                elif unit in 'uμµ':
                    value *= 1e-6
                elif unit == 'n':
                    value *= 1e-9
            values.append(value)
        return np.array(values)

    def __parse_measurement(self, value_str: str) -> float:
        return self.__parse_measurements(value_str)[0]

    @property
    def position(self) -> np.ndarray:
        """
        Extracts the current position from true_pos function (the position as given by the stage controller)
        :return: A vector with the current position in meters.
        """
        with self._lock:
            responses = [self.__query(f"GET_{'XYZ'[_]}") for _ in range(self.ndim)]
            return np.array([self.__parse_measurement(_) for _ in responses])

    @position.setter
    def position(self, new_position: np.ndarray):
        """
        Move the stage to an absolute position.
        :param new_position: A vector with the target position in meters.
        """
        with self._lock:
            new_position = np.atleast_1d(new_position)
            if new_position.size != self.ndim:
                raise TypeError(f'This stage has {self.ndim} axes, but the position is a {new_position.size}-element vector!')
            new_position = np.clip(new_position, self.ranges[:, 0], self.ranges[:, 1])
            for axis in range(self.ndim):
                if not self.__command(f"MOVE{'XYZ'[axis]} {int(new_position[axis] * 1e9 + 0.5)}n"):
                    log.error(f"Failed MOVE{'XYZ'[axis]} {int(new_position[axis] * 1e9 + 0.5)}n")

    @property
    def time_resolution(self) -> float:
        """Gets the position update time-interval in seconds for translations."""
        return self.__parse_measurement(self.__query('GTIME'))

    @time_resolution.setter
    def time_resolution(self, dt: float):
        """Sets the position update time-interval in seconds for translations."""
        if not self.__command(f'STIME {dt*1e6:0.0f}u'):
            log.error('Could not set time.')

    def set_trigger(self, ttl_connector: int = 1, output: bool = True, axis: int = 0):
        if output:
            response = self.__query(f"CHAIO {ttl_connector}o{axis+1}s")  # start of motion
            log.info(f'Setting TTL{ttl_connector} to output returned {response}.')
        else:
            response = self.__query(f"CHAIO {ttl_connector}i{axis+1}r")  #rising edge
            log.info(f'Setting TTL{ttl_connector} to input returned {response}.')
        if not self.__command():
            log.error(f'Could not set TTL{ttl_connector}.')

    def translate(self, velocity: Union[None, float, Sequence, np.ndarray] = None,
                  origin: Union[None, float, Sequence, np.ndarray] = None,
                  destination: Union[None, float, Sequence, np.ndarray] = None) -> Translation:
        """
        Translate the stage.
        Usage:
        with stage.translation(1e-3) as t:
            while t.position < 10e-3:
                print(t.position)

        :param velocity: The translation velocity in m/s (default: the maximum velocity for this stage).
        :param origin: (optional) The origin position in meters (default: the current position).
        :param destination: (optional) The destination in meters (default: the end of the stage's range).
        :return: A context manager object for the stage Translation.
        """
        with self._lock:
            if velocity is None:
                velocity = self.max_velocity
                if origin is not None and destination is not None:
                    velocity = velocity * np.sign(destination - origin)
            else:
                velocity = np.atleast_1d(velocity)
                velocity = np.minimum(self.max_velocity, np.abs(velocity)) * np.sign(velocity)
            # Make sure that there is only one Translation object associated with this stage
            # if self.__translation is not None:
            #     self.__translation.stop()
            self.__translation = Translation(self, self.max_acceleration,
                                             velocity=velocity, origin=origin, destination=destination)
            return self.__translation

    def translation_init(self, velocity: Union[None, float, Sequence, np.ndarray] = None,
                         origin: Union[None, float, Sequence, np.ndarray] = None,
                         destination: Union[None, float, Sequence, np.ndarray] = None):
        max_translation_time = 60.0  # seconds

        if origin is None:
            origin = self.position
        else:
            origin = np.asarray(origin)
        if destination is None:
            range_end = self.ranges[:, 0] * (velocity < 0) + self.ranges[:, 1] * (velocity > 0)\
                        + origin * (velocity == 0)
            travel_time = np.amin((range_end - origin) / velocity)
            destination = origin + velocity * travel_time
        else:
            destination = np.asarray(destination)
        travel_distance = np.linalg.norm(destination - origin)
        travel_time = np.amax(travel_distance / velocity)

        if travel_time > max_translation_time:
            log.info(f'The stages travel time for a single scan would be {travel_time}s, this is assumed to be an error. Limiting it to {max_translation_time}s.')
            travel_time = max_translation_time

        dt = self.time_resolution
        nb_data_points = np.clip(int(travel_time / dt + 0.5), 2, 2332)
        dt = travel_time / nb_data_points
        self.time_resolution = dt

        dt = self.time_resolution  # get the rounded value
        nb_data_points = int(travel_time / dt + 0.5)

        # move stage to the start position
        self.position = origin

        # Add a waveform
        log.info(f'Adding {nb_data_points} data points...')
        if not self.__command(f'ARB3D {nb_data_points:0.0f} {nb_data_points:0.0f} {nb_data_points:0.0f}'):
            log.error(f'Could not load arbitrary 3D waveform with {nb_data_points} data points.')
        for t in np.arange(0.0, travel_time, dt):
            pos = np.clip(origin + self.velocity * t, self.ranges[:, 0], self.ranges[:, 1])
            if not self.__command(f'ADD3D {pos[0]*1e9:0.0f}n {pos[1]*1e9:0.0f}n {pos[2]*1e9:0.0f}n'):
                log.error(f'Could not add 3D point {pos*1e6} um to waveform.')

    def translation_trigger(self):
        self.__translating = True
        self.__query('RUN3D')  # => returns Ok,
        while self.__query().lower() != 'completed':
            time.sleep(10e-3)
        log.info('Scan terminated.')

    def scan(self, grid: Grid, detector, callback: Callable = None) -> np.ndarray:  # Currently need to specify a 3D grid
        scan_order = np.argsort(grid.shape)[::-1]  # scanning along the longest axis
        scan_extent = np.array([grid.center - grid.extent/2, grid.center + grid.extent/2])[scan_order[0]]

        v = np.zeros(3)
        v[scan_order[0]] = grid.step[scan_order[0]] / detector.dt  # line scan velocity
        n = grid.shape[scan_order[0]]
        # this variable will be updated for the purposes of un-timed stage movement, but is an inaccurate representation of the current position
        position = grid.center

        # preparing to save the result
        result = np.zeros(grid.shape)
        slc_result = [0, 0, 0]
        slc_result[scan_order[0]] = slice(None)

        self.__command(f'ARBWF {grid.shape[0]} {grid.shape[1]} {grid.shape[2]}')

        for count, line in enumerate(grid[scan_order[1]]):

            # moving the stage to the start of the line
            scan_extent = scan_extent[::-1] if (count % 2) else scan_extent
            position[scan_order[0]] = scan_extent[0]
            position[scan_order[1]] = line
            self.position = position

            # new position for line destination
            position[scan_order[0]] = scan_extent[1]

            # scanning the line
            slc_result[scan_order[1]] = count
            with self.translate(velocity=v, origin=self.position, destination=position) as t:
                result[tuple(slc_result)] = detector.read(nb_values=n)
                if callback is not None:
                    if not callback(result):
                        break
        return result

    def home(self):
        log.info('Homing...')
        self.__query('_RAZ_')  # All DAC voltages to 0V

    def wait(self):
        """Wait for any scan to finish."""
        while self.__translating:
            log.info('Waiting for stage to complete scan...')
            time.sleep(0.010)
            self.__query()
        log.info('Stage completed scan.')

    def _disconnect(self):
        self.__serial_port.close()
