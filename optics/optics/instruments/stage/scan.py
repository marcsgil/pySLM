# TODO: Is /instruments/ the best place for these classes?

import numpy as np
# import time
from typing import Union


class RasterScan:
    pass


class BoustrophedonScan:
    pass


class Scan:
    def __init__(self, sample_pitch: np.array, extent: np.array,
                 direction_vec: Union[list, np.array, tuple], scan_type: str = 'boustrophedon'):
        # TODO: allow to pass a call_back function, where camera \
        #  capture and actual stage movement parameters can be specified

        # TODO: Create a function which is able to transform the point vector_grid into a 2d queue of points

        # TODO: Create a function which does the iteration \
        #  of the point queue and also uses the call_back function

        # TODO: create a movement velocity property and use it in the iterator (or it can be specified in call_back)

        # TODO: Create a pause between steps variable (could do it more precisely by constantly \
        #  calibrating the pause time with current time/ time that passed since last command.

        # Given parameters
        self.__scan_type = scan_type
        self.direction_vec = np.array(direction_vec)
        self.extent = extent  # e.g. [[0, 300e-6], [0, 300e-6], [0, 300e-6]]
        self.sample_pitch = sample_pitch

        # Derived parameters
        self.ndim = self.__ndim
        self.scan_length = self.__scan_length
        self.nb_points = self.__nb_points
        self.adj_length = self.__adj_length
        self.centred_extent = self.__centred_extent
        self.scan_ranges = self.__scan_ranges

    @property
    def vector_grid(self):
        """
        Creates a vector grid for points that are to be scanned.
        :return:
        """
        # TODO: Finish vector_grid function
        vector_grid = np.zeros((*self.nb_points, self.ndim))
        for axis in range(len(self.nb_points)):
            pass

    @property
    def __scan_ranges(self):
        """
        Creates an array of ranges giving positions of all scan points for each axis separately
        """
        extent = self.centred_extent
        scan_ranges = []
        for axis in range(self.ndim):
            scan_ranges.append(np.linspace(extent[axis, 0], extent[axis, 1], self.nb_points[axis]))
        return np.array(scan_ranges)

    @property
    def __centred_extent(self):
        """
        Creates a new extent in the place of the old one with adjusted scan lengths
        to correspond with sample pitch and symmetrically centers the new extent to be within the old extent.
        """
        extent_offset = (self.scan_length - self.adj_length) / 2

        # Centering the new extent
        centered_extent = self.extent
        centered_extent[:, 0] += extent_offset
        centered_extent[:, 1] += -extent_offset
        return centered_extent

    @property
    def __ndim(self):
        """
        Calculates the number of dimmensions, that will be used for the scan.
        """
        return self.extent.shape[0]

    @property
    def __adj_length(self):
        """
        Scan length with adjusted size to agree with sample pitch.
        This is the true scan length, that will be used by this class.
        adj_length <= scan_length
        shape: self.ndim
        """
        return self.sample_pitch * self.nb_points

    @property
    def __nb_points(self):
        """
        Calculates the number of points that will be scanned in each axis.
        shape: self.ndim
        """
        return self.scan_length // self.sample_pitch

    @property
    def __scan_length(self):
        """
        Calculates the length of the given scan range/extent
        shape: self.ndim
        """
        scan_length = np.zeros(self.ndim)  # Will hold the given scan length
        for idx, axis in enumerate(self.extent):
            scan_length[idx] = axis[1] - axis[0]
        return scan_length

    def __rotate_grid(self):
        # TODO: create a function, which is able to rotate the points in \
        #  vector_grid to correspond to some given unit vector
        pass






