import numpy as np
import logging

log = logging.getLogger(__name__)

try:
    import nidaqmx
    from nidaqmx.constants import TerminalConfiguration, AcquisitionType
except ModuleNotFoundError as err:
    log.error('NI-DAQmx Python API not found. Please install the nidaqmx package installed with "pip install nidaqmx"? Make sure that the National Instruments drivers installed as well.')
    raise err

from . import Detector


class NIDAQDetector(Detector):
    """
    A class to represent (photo)-detectors connected to a National Instruments DAQ card.
    """
    def __init__(self, nb_diodes=None, dt=None, nb_samples=None, terminal_config='Differential'):
        super().__init__(dt=dt)

        self._terminal_configs = {'Differential': TerminalConfiguration.DIFFERENTIAL,
                                  'RSE': TerminalConfiguration.RSE}
        self.__terminal_config = self._terminal_configs[terminal_config]

        if nb_diodes is None:
            nb_diodes = 1

        self.max_sampling_fq = int(5e5 / nb_diodes)
        if dt is None:
            dt = 1 / self.max_sampling_fq  # max rate of (nb_diodes / 5e5)

        self.__dt = dt
        self.__nb_diodes = nb_diodes
        self.__nb_samples = nb_samples

        self.task, self.__ai_channels = self.__initialise_terminal()
        self.__set_acquisition()

    @property
    def terminal_config(self):
        return self.__terminal_config

    @terminal_config.setter
    def terminal_config(self, new_config):
        self.__terminal_config = self._terminal_configs[new_config]
        for _ in self.__nb_diodes:
            self.__ai_channels[_].ai_term_cfg = self.__terminal_config

    @property
    def dt(self) -> float:
        return self.__dt

    @dt.setter
    def dt(self, new_dt):
        self.__dt = new_dt

    @property
    def nb_samples(self):
        return self.__nb_samples

    @nb_samples.setter
    def nb_samples(self, nb_samples):
        """
        Pre-specifying the number of samples that need to be acquired provides a more reliable read-out.
        This is especially important when reading out very small number of samples (e.g. on the order of 1k).
        """
        self.__nb_samples = nb_samples
        self.__set_acquisition()

    def __set_acquisition(self):
        with self._lock:
            if self.nb_samples is None:
                self.task.timing.cfg_samp_clk_timing(1 / self.dt, sample_mode=AcquisitionType.CONTINUOUS)
            else:
                self.task.timing.cfg_samp_clk_timing(1 / self.dt, sample_mode=AcquisitionType.FINITE, samps_per_chan=self.nb_samples)

    def __initialise_terminal(self):
        with self._lock:
            task = nidaqmx.Task()
            ai_channels = np.empty(self.__nb_diodes, dtype=object)
            for idx in range(self.__nb_diodes):
                ai_channels[idx] = task.ai_channels.add_ai_voltage_chan(f'Dev1/ai{idx}', terminal_config=self.terminal_config)
            return task, ai_channels

    def read(self, nb_values: int) -> np.ndarray:
        # Do something, wait until values arrive, return
        with self._lock:
            data = self.task.read(nb_values)
            return data

    def power_down(self):
        with self._lock:
            self.task.close()

    def close(self):
        with self._lock:
            self.task.close()


