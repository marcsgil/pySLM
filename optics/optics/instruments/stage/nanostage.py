from __future__ import print_function
from typing import Union

from builtins import str
import serial
import time
import numpy as np
import re

import nplab.instrument.serial_instrument as si

import logging

log = logging.getLogger(__name__)

##Some functions was discontinued in Python 3.10
class NanostageLT3(si.SerialInstrument):  # todo: inherit from Stage to make all stages compatible.
    """
    A Class for controlling Piezoconcept LT3 nanostage through USB interface
    """
    def __init__(self, port='COM3'):
        self.termination_character = '\n'
        self.port_settings = {
            'baudrate': 115200,
            'bytesize': serial.EIGHTBITS,
            'parity': serial.PARITY_NONE,
            'stopbits': serial.STOPBITS_ONE,
            'timeout': 1,  # wait at most one second for a response
            #          'writeTimeout':1, #similarly, fail if writing takes >1s
            #         'xonxoff':False, 'rtscts':False, 'dsrdtr':False,
        }
        si.SerialInstrument.__init__(self, port=port)

        self.__axes = ('X', 'Y', 'Z')

        # TODO change to si units
        self.stage_range = np.array([[0, 300e3],
                                    [0, 300e3],
                                    [0, 300e3]])
        self.position = self.__position
        # self.center_all()  # Required to get/set the initial positions

    @property
    def __position(self) -> np.array:
        """
        Extracts the current position from true_pos function (the position as given by the stage controller)
        :return:
        """
        exact_position = self.true_pos()  # position given by controller in a dict
        number_output = re.compile(r'^[0-9.]{1,7}')  # regex pattern for position number search

        position = np.zeros(np.shape(self.__axes))
        for idx, axis in enumerate(self.__axes):
            position[idx] = (float(number_output.search(exact_position[axis]).group())) * 1e3  # units in nm
        return position

    @staticmethod
    def __unit_translator(given_unit: str):  # I would just work with SI units to keep things simple.
        """
        Converts the given units to the stage accepted units
        :param given_unit: Physical unit given by the user
        :return: Unit format accepted by the stage
        """
        if given_unit == 'nm' or given_unit == 'um':
            return given_unit[0]
        elif given_unit == 'n' or given_unit == 'u':
            return given_unit
        else:
            raise Exception('Incorrect unit input was given. The stage accepts only nm and um units.')

    def __unit_interpreter(self, given_unit: str):
        """
        Converts the given units into a numerical multiplier
        :param given_unit: Physical unit given by the user
        :return: multiplier associated with the unit
        """
        unit = self.__unit_translator(given_unit)
        if unit == 'n':
            multiplier = 1
        else:
            multiplier = 1e3
        return multiplier

    def move(self, axis: int, value, unit='nm'):
        """
        Moves the nanostage to the specified position
        :param value: position to which the user wants to move the stage
        :param unit: units in which the value is denoted (can be either nm or um)
        :param axis: LT3 axis, which will be moved todo: define the order of axis (x,y,z)???
        :return: (optional return: current position of stage)
        """
        unit = self.__unit_translator(unit)
        multiplier = self.__unit_interpreter(unit)

        if self.stage_range[axis, 1] >= (value * multiplier) >= self.stage_range[axis, 0]:
            self.issue_command(f"MOVE{self.__axes[axis]} {int(value * multiplier + 0.5)}n")
            self.position[axis] = (value * multiplier)
        else:
            log.warning(f"The value is out of range! {self.stage_range[axis]} nm")

    def move_rel(self, axis: int, value, unit='nm'):
        """
        Moves the nanostage to some position, relative to the current position
        :param value: relative distance by which the user wants to move the stage
        :param unit: units in which the value is denoted (can be either nm or um)
        :param axis: LT3 axis, which will be moved
        :return: (optional return: current position of stage)
        """
        unit = self.__unit_translator(unit)
        multiplier = self.__unit_interpreter(unit)

        if self.stage_range[axis, 1] > (value * multiplier + self.position[axis]) >= self.stage_range[axis, 0]:
            self.issue_command(f"MOVR{self.__axes[axis]} {int(value * multiplier + 0.5)}n")
            self.position[axis] = (value * multiplier + self.position[axis])
        else:
            log.warning(f"The value is out of range! {self.stage_range[axis]} nm")

    def center_all(self):
        """
        Centers the stage
        :return:
        """
        for idx in range(3):
            self.move(axis=idx, value=150, unit="u")
            self.position[idx] = 150e3

    def true_pos(self):
        """
        Gives out the true position as perceived by the nanostage controller
        :return: iteratively prints out the true position for all axes
        """
        # todo: figure out how to get rid of the initialization line
        exact_pos = {}
        for axis in range(3):
            if axis == 0:
                self.query(f"GET_X", multiline=True, termination_line="\n \n \n \n")  # initialization line
            answer = self.query(f"GET_{self.__axes[axis]}", multiline=True, termination_line=" ")
            # log.debug(f'for axis {self.__axes[axis]}, the answer is {answer}')
            exact_pos[self.__axes[axis]] = answer.replace('\n', '')

        return exact_pos

    def scan(self, extent: np.array=None, scan_axes: Union[tuple, list, np.array]=None, nb_points=None, t_step=0.1):
        """
        All arguments should be specified in si units
        :param extent:
        :param scan_axes:
        :param nb_points:
        :param t_step:
        :return:
        """

        if extent is None:
            extent = self.stage_range

        if scan_axes is None:
            scan_axes = np.arange(np.shape(extent)[0])

        if nb_points is None:
            nb_points = (np.ones(len(scan_axes)) * 10).astype(int)

        if np.shape(scan_axes)[0] != np.shape(extent)[0]:
            return 'NanostageScanError: the number of axis in extent and scan_axes does not match!'

        t_step *= 1e3  # The time is given to the stage in units of ms
        # extent *= 1e9  # The range is given to the controller in nm

        # if True:  # TODO: debugging
        #     raise Exception('Figure out what CHAIO command does first, it is important for scanning.')

        # Setting up a waveform structure inside the controller
        for idx, scan_range in enumerate(extent):
            self.issue_command(f'SWF_{self.__axes[scan_axes[idx]]} {nb_points[idx]} {int(scan_range[0])}n {int(scan_range[1])}n')
            log.info(f'SWF_{self.__axes[scan_axes[idx]]} {nb_points[idx]} {int(scan_range[0])}n {int(scan_range[1])}n')
        self.issue_command('CHAIO 1o1s')
        self.issue_command(f'STIME {int(t_step)}m')  # time between steps
        self.issue_command('RUNWF')  # starting the scan






    def issue_command(self, command: str):
        """
        A general function for giving commands to the nanostage via text input. It uses a write
        function, which is inherited from nplab.instrument.serial_instrument.SerialInstrument class.
        :param command: text input for the command, must be given according to the manual
        :return:
        """
        self.write(command)



    # Scans are quite high-level, perhaps should go in a general function or a be a method of Stage so it can be used for all stages?
    def raster(self, nb_points: int = 0, step: float = 0, axes: list = (1, 2), sleep: float = 0, equal_step=False):
        """
        Instructs the nanostage to perform a raster scan, with maximum allowed range, over a plane.
        :param nb_points: number of points to scan in each axis
        :param step: the step size between scanning points
        :param axes: two axes over which the raster scan will be performed (order is important)
        :param sleep: time taken between giving instructions
        :param equal_step: if True, forces the steps to be approximately equal in all directions
        :return:
        """
        scan_range_ax1 = self.stage_range[axes[0]]
        scan_range_ax2 = self.stage_range[axes[1]]

        # Holds points, which will be scanned
        if nb_points != 0 and step == 0:  # definition of scan point positions from the number of points
            points_ax1 = np.linspace(*scan_range_ax1, nb_points)
            if not equal_step:
                points_ax2 = np.linspace(*scan_range_ax2, nb_points)
            else:
                step_size = np.abs(points_ax1[1] - points_ax1[0])
                nb_allowed_points = int(scan_range_ax2[1] / step_size)
                points_ax2 = np.linspace(*scan_range_ax2, nb_allowed_points)
        elif nb_points == 0 and step != 0:  # definition of scan point positions from the given step size
            ax1_max_nb_points = np.floor((scan_range_ax1[1] - scan_range_ax1[0]) / step)
            ax2_max_nb_points = np.floor((scan_range_ax2[1] - scan_range_ax2[0]) / step)
            points_ax1 = np.arange(ax1_max_nb_points) * step + scan_range_ax1[0]
            points_ax2 = np.arange(ax2_max_nb_points) * step + scan_range_ax2[0]
        else:
            raise Exception('nb_points/step inputs are specified incorrectly. \
                            The user needs to specify one of these arguments, but not both.')

        # performs the raster scan
        combined_nb_points = len(points_ax2) * len(points_ax1)
        progress = 0
        log.debug('raster scanning...')
        for ax2_point in points_ax2:
            self.move(axis=axes[1], value=ax2_point, unit='n')
            for ax1_point in points_ax1:
                self.move(axis=axes[0], value=ax1_point, unit='n')
                time.sleep(sleep)

                # progress report
                progress += 1
                relative_progress = progress / combined_nb_points * 100
                if relative_progress % 5 == 0:
                    log.debug(f'raster scanning progress {int(relative_progress)}%...')
        log.debug('raster scan complete...')

    def boustrophedon(self, nb_points: int = 0, step: float = 0, axes: list = (1, 2), sleep: float = 0, equal_step=False):
        """
        Instructs the nanostage to perform a boustrophedon scan, with maximum allowed range, over a plane.
        :param nb_points: number of points to scan in each axis
        :param step: the step size between scanning points
        :param axes: two axes over which the raster scan will be performed (order is important)
        :param sleep: time taken between giving instructions
        :param equal_step: if True, forces the steps to be approximately equal in all directions
        :return:
        """
        scan_range_ax1 = self.stage_range[axes[0]]
        scan_range_ax2 = self.stage_range[axes[1]]

        # Holds points, which will be scanned
        if nb_points != 0 and step == 0:  # definition of scan point positions from the number of points
            points_ax1 = np.linspace(*scan_range_ax1, nb_points)
            if not equal_step:
                points_ax2 = np.linspace(*scan_range_ax2, nb_points)
            else:
                step_size = np.abs(points_ax1[1] - points_ax1[0])
                nb_allowed_points = int(scan_range_ax2[1] / step_size)
                points_ax2 = np.linspace(*scan_range_ax2, nb_allowed_points)
        elif nb_points == 0 and step != 0:  # definition of scan point positions from the given step size
            ax1_max_nb_points = np.floor((scan_range_ax1[1] - scan_range_ax1[0]) / step)
            ax2_max_nb_points = np.floor((scan_range_ax2[1] - scan_range_ax2[0]) / step)
            points_ax1 = np.arange(ax1_max_nb_points) * step + scan_range_ax1[0]
            points_ax2 = np.arange(ax2_max_nb_points) * step + scan_range_ax2[0]
        else:
            raise Exception('nb_points/step inputs are specified incorrectly. \
                            The user needs to specify one of these arguments, but not both.')

        # performs the raster scan
        combined_nb_points = len(points_ax2) * len(points_ax1)
        progress = 0
        log.debug('boustrophedon scanning...')
        for ax2_point in points_ax2:
            self.move(axis=axes[1], value=ax2_point, unit='n')
            if progress % 2 != 0 and progress != 0:  # scan direction alteration condition
                points_ax1 = np.flip(points_ax1)
            for ax1_point in points_ax1:
                self.move(axis=axes[0], value=ax1_point, unit='n')
                time.sleep(sleep)

                # progress report
                progress += 1
                relative_progress = progress / combined_nb_points * 100
                if progress % 10 == 0:
                    log.debug(f'boustrophedon scanning progress {int(relative_progress)}%...')
        log.debug('boustrophedon scan complete...')
