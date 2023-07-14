import numpy as np

from optics.utils.ft import Grid
from optics.instruments.slm import SLM
from optics.instruments.cam.ids_cam import IDSCam


class HadamardBasis:
    """
    Provides an orthogonal Hadamard basis for transmission matrix scan.
    Configured through input of degrees of freedom and used through iteration.
    """
    def __init__(self, grid):
        self.__grid = grid

    def __iter__(self):
        pass


class TmScan:
    def __init__(self, grid):
        self.__grid = grid

        self.__tm = None

    @property
    def grid(self):
        return self.__grid

    @property
    def tm(self):
        return self.__tm

    @tm.setter
    def tm(self, input):
        """
        Adds new captured data to tm, while keeping the rest
        :param input:
        :return:
        """
        pass

    def capture2tm(self):
        """
        unravels the captured frames so that they can be put directly into the transmission matrix
        :return:
        """
        pass

    def capture(self):
        """
        capture a frame and add it to rest
        :return:
        """
        pass

    def move_spot(self):
        """
        moves the imaged spot on the sample
        :return:
        """
        pass


class TmScanSLM(TmScan):
    def __init__(self, grid):
        super().__init__(self, grid)

        self.SLM = SLM()  # TODO: complete the initialisation of slm

    def move_spot(self):
        """
        moves the imaged spot on the sample via slm
        :return:
        """
        pass


class TmScanStage(TmScan):
    pass

