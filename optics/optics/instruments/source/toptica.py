import numpy as np
from typing import Union, Pattern
import serial.tools.list_ports
import re
from scipy.constants import convert_temperature

import logging
log = logging.getLogger(__name__)
from optics.instruments.source import Laser


class IBeamSMARTLaser(Laser):
    def __init__(self, serial_port: Union[serial.Serial, str, None] = None, power_down_on_disconnect: bool = True):
        # Initialize the super class, even though we don't know the wavelength and power yet.
        super().__init__()

        self.__encoding = 'utf-8'
        self.__serial_port: serial.Serial = serial_port
        self.__fina_a_b = np.zeros(2)  # Not sure how these can be read from the laser
        self._connect()

        self.emitting = False
        self.channel_1 = True
        self.channel_2 = False
        self.power_1 = 0.0
        self.power_2 = 0.0
        self.fine_a = 0.90  # Not sure how these can be read from the laser
        self.fine_b = 0.10

        # Now initialize the super class correctly.
        super().__init__(max_power=self.max_power, wavelength=self.wavelength, wavelength_bandwidth_fwhm=4e-9,
                         power_down_on_disconnect=power_down_on_disconnect)

    def _connect(self):
        with self._lock:
            def open_port(port_device: str) -> serial.Serial:
                return serial.Serial(port=port_device, baudrate=115200, bytesize=8, parity=serial.PARITY_NONE, stopbits=1)

            def init_sequence():
                # self.__serial_port.write('\r\n'.encode(self.__encoding))
                self.__query('echo off')
                self.__query('talk usual')
                self.__query('flash off')

            if isinstance(self.__serial_port, serial.Serial):
                if not self.__serial_port.isOpen():
                    self.__serial_port.open()
                    init_sequence()
            else:
                if isinstance(self.__serial_port, str):
                    self.__serial_port = open_port(self.__serial_port)
                    if self.__serial_port.isOpen():
                        log.info(f'Port {self.__serial_port} is open.')
                        init_sequence()
                    else:
                        raise ConnectionError(f'Could not open port {self.__serial_port}.')
                else:
                    serial_port_infos = serial.tools.list_ports.comports(include_links=True)
                    if self.__serial_port is None or isinstance(self.__serial_port, str):
                        for port_info in serial_port_infos:
                            log.info(f'Trying to find laser on port {port_info.description}...')
                            try:
                                self.__serial_port = open_port(port_info.device)
                                if self.__serial_port.isOpen():
                                    log.info(f'Port {self.__serial_port} is open.')
                                    init_sequence()
                                    break
                                else:
                                    self.__serial_port = None
                            except serial.serialutil.SerialException as se:
                                log.info(f'Could not open port {port_info.description}.')

    def __query(self, command: str) -> str:
        with self._lock:
            self._connect()
            self.__serial_port.write((command + '\r\n').encode(self.__encoding))
            log.debug(f'Sent: {command}')
            prompt = b'\r\nCMD>'
            line = self.__serial_port.read_until(terminator=prompt)

            return line[:-len(prompt)].decode(self.__encoding).strip()  # Don't return the prompt and strip whitespace

    @staticmethod
    def __extract_response(value_str: str, pattern: Union[Pattern, str]) -> str:
        """

        :param value_str: The string to parse.
        :param pattern: The regular expression pattern to match. The first group `(...)` is returned.
        Use `(?:...)` to ignore parenthesis.
        :return: The first group string.
        """
        if not isinstance(pattern, Pattern):
            pattern = re.compile(pattern, re.MULTILINE)
        value_match = pattern.search(value_str)
        if value_match is None or len(value_match.groups()) < 1:
            raise ValueError(f'{value_str} has no match for {pattern}.')

        value_str = value_match.groups()[0]
        return value_str

    def __get_str(self, command: str, pattern: Union[None, Pattern, str] = None) -> str:
        with self._lock:
            value_str = self.__query(command)
            log.debug(f'Received: {value_str}.')
            if pattern is not None:
                value_str = self.__extract_response(value_str, pattern)
            return value_str

    def __get_float(self, command: str, pattern: Union[None, Pattern, str] = None) -> float:
        value_str = self.__get_str(command=command, pattern=pattern)
        return float(value_str)

    def __get_bool(self, command: str, pattern: Union[None, Pattern, str] = None) -> bool:
        value_str = self.__get_str(command=command, pattern=pattern)
        on = bool(re.match(r'\s*ON\s*', value_str, re.IGNORECASE))
        off = bool(re.match(r'\s*OFF\s*', value_str, re.IGNORECASE))
        if on == off:
            raise ValueError(f'Did not receive either ON or OFF, but "{value_str}".')
        return on

    def init(self):
        pass

    def power_down(self):
        self.power = 0.0
        self.emitting = False

    def _disconnect(self):
        self.__serial_port.close()

    @property
    def max_power(self):
        return self.__get_float('sh sat', r'Pmax:\s*(\d+)\s*mW') * 1e-3

    @property
    def serial(self) -> str:
        return self.__get_str('sh serial')

    @property
    def frequency(self) -> float:
        """
        Get the frequency in Hertz.

        The wavelength is determined from the frequency by the superclass using frequency2wavelength()
        """
        wavelength = float(re.search(r'iBEAM-SMART-(\d+)', self.serial, re.IGNORECASE).groups()[0]) * 1e-9
        return float(self.wavelength2frequency(wavelength))

    @property
    def temperature(self) -> float:
        """Get the laser diode temperature in Kelvin."""
        temp_celcius = self.__get_float('sh temp', r'\s*(\d+\.\d*)\s*C')
        return convert_temperature(temp_celcius, 'Celsius', 'Kelvin')

    @property
    def currents(self) -> np.ndarray:
        current1 = self.__get_float('sh 1 curr', r'LDC\s*=\s*(\d*\.\d*)\s*mA') * 1e-3
        current2 = self.__get_float('sh 2 curr', r'LDC\s*=\s*(\d*\.\d*)\s*mA') * 1e-3
        return np.array([current1, current2])

    @property
    def power(self) -> float:
        with self._lock:
            power_str = self.__get_str('sh level pow')
            power1 = float(self.__extract_response(power_str, r'CH1,\s*PWR:\s*(\d*\.\d*)\s*mW')) * 1e-3
            power2 = float(self.__extract_response(power_str, r'CH2,\s*PWR:\s*(\d*\.\d*)\s*mW')) * 1e-3
            return power1 + power2

    @power.setter
    def power(self, new_power: float):
        with self._lock:
            self.__query(f'set 1 pow {new_power * 1e3:0.1f}')

    @property
    def power_1(self) -> float:
        power_str = self.__get_str('sh level pow')
        return float(self.__extract_response(power_str, r'ch1,\s*pwr:\s*(\d*\.\d*)\s*mW')) * 1e-3

    @power_1.setter
    def power_1(self, new_power: float):
        self.__query(f'set 1 pow {new_power * 1e3:0.1f}')

    @property
    def power_2(self) -> float:
        power_str = self.__get_str('sh level pow')
        return float(self.__extract_response(power_str, r'ch2,\s*pwr:\s*(\d*\.\d*)\s*mW')) * 1e-3

    @power_2.setter
    def power_2(self, new_power: float):
        self.__query(f'set 2 pow {new_power * 1e3:0.1f}')

    @property
    def emitting(self) -> bool:
        return self.__get_bool('sta la')

    @emitting.setter
    def emitting(self, new_status: bool):
        self.__query('la on' if new_status else 'la off')

    @property
    def channel_1(self) -> bool:
        return self.__get_bool('sta 1 ch')

    @channel_1.setter
    def channel_1(self, new_status: bool):
        self.__query('en 1' if new_status else 'di 1')

    @property
    def channel_2(self) -> bool:
        return self.__get_bool('sta 2 ch')

    @channel_2.setter
    def channel_2(self, new_status: bool):
        self.__query('en 2' if new_status else 'di 2')

    @property
    def fine(self) -> bool:
        return self.__get_bool('sta fine')

    @property
    def fine_a(self) -> float:
        with self._lock:
            return self.__fina_a_b[0]

    @fine_a.setter
    def fine_a(self, new_value: float):
        with self._lock:
            new_value = np.clip(new_value, 0, 1)
            self.__query(f'fine a {100 * new_value:0.1f}')
            self.__fina_a_b[0] = new_value

    @property
    def fine_b(self) -> float:
        with self._lock:
            return self.__fina_a_b[1]

    @fine_b.setter
    def fine_b(self, new_value: float):
        with self._lock:
            new_value = np.clip(new_value, 0, 1)
            self.__query(f'fine b {100 * new_value:0.1f}')
            self.__fina_a_b[1] = new_value

    @property
    def skill(self) -> bool:
        return self.__get_bool('sta osc')

    @skill.setter
    def skill(self, new_status: bool):
        self.__query('skill ' + ('on' if new_status else 'off'))

    def __str__(self):
        return f'Toptica iBeamSMART laser with serial number {self.serial}, ' + \
               f'wavelength {self.wavelength * 1e9:0.1f} nm and maximum power {self.max_power*1e3:0.0f} mW.'

