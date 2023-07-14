import numpy as np

from optics.instruments.dm import log
from optics.instruments.dm.dm import DM

try:
    from optics.external.alpao_dm import asdk
except ImportError as ie:
    log.error('Alpao asdk could not be loaded. Make sure to install the ALPAO device drivers and the configuration for '
              + 'the deformable mirror with the specific serial number and the USB RNDIS Gadget driver from '
              + 'https://www.catalog.update.microsoft.com/Search.aspx?q=USB+RNDIS%20Gadget . Also, check the python interpreter. '
              + 'It works well with Python 3.9 and 3.10. '
              + 'Please install the SDK from: https://www.alpao.com/Download/AlpaoSDK and copy Downloads/Alpao_SDK_4.01.06/Samples/Python3/Lib64 to optics/external/alpao_dm/win')
    raise ie  # todo: replace by something more user-friendly
    # raise DMError('Alpao Software Development Kit (SDK) not found to control the deformable mirror.\n'
    #               + 'Please install the SDK from: https://www.alpao.com/Download/AlpaoSDK and copy Downloads/Alpao_SDK_4.01.06/Samples/Python3/Lib64 to optics/external/alpao_dm/win')
    #Error when


class AlpaoDM(DM):
    """
    A class to control Alpao deformable mirrors.
    """
    def __init__(self, serial_name: str, radius: float = 1, max_stroke: float = 1.0, wavelength: float = 1.0,
                 asynchronous: bool = False, gain_factor: float = 1 / 16, nb_interpolation_steps: int = 16):
        """
        Construct an object to represent an Alpao deformable mirror.

        :param serial_name: The serial number of the device that needs to be controlled.
        :param radius: (optional) The radius of the deformable mirror.
        :param max_stroke: (optional) The maximum stroke of the mirror in meters.
        :param wavelength: (optional) The wavelength in meters.
        :param asynchronous: Don't wait for mirror to be updated before returning from :py:meth:``modulate``. Default: wait.
        :param gain_factor: Gain factor, relative to 8191 (default 1.0, see manual).
        :param nb_interpolation_steps: Interpolation steps in mirror position update (see manual).
        """
        self.__serial_name = serial_name
        self.__dm = asdk.DM(self.__serial_name)
        self.__dm.Set('Gain', int(8191 * gain_factor))  # 8191 is the default value
        self.__dm.Set('SyncMode', 0 if asynchronous else 1)  # 0 = wait for Send to finish, 1 = return immediately
        self.__dm.Set('UseException', 1)  # raise Exception on errors
        log_print_level = self.__dm.Get('LogPrintLevel')  # Can also be set
        #self.__dm.Set('mcff', np.arange(0, 1, 0.01))  # Ramp between positions
        self.__dm.Set('NbSteps', nb_interpolation_steps)  # Number of steps between mirror positions. For real-time (> 1kHz) control, set to 1 to improve performance.
        self.__dm.Set('ResetOnClose', 1)  # Number of steps between mirror positions. For real-time (> 1kHz) control, set to 1 to improve performance.
        config_path = self.__dm.Get('CfgPath')

        self.__nb_actuators = int(self.__dm.Get('NBOfActuator'))

        version_info = self.__dm.Get('VersionInfo')
        log.info(f'Alpao SDK version {version_info}. Log level set to {log_print_level}. Loading deformable mirror with configuration file {config_path}...')

        super().__init__(radius=radius, nb_actuators=self.__nb_actuators, max_stroke=max_stroke, wavelength=wavelength)

        log.info(f'Initialized Alpao {self.__serial_name} deformable mirror with ' +
                 f'{self.actuator_position.shape[0]} actuators.')

    def _modulate(self, actual_stroke_vector: np.ndarray):
        with self._lock:
            if self.__dm is not None:
                voltage_fraction = np.clip(actual_stroke_vector.ravel() / self.max_stroke, -1.0, 1.0)  # Convert stroke to control voltage
                result = self.__dm.Send(voltage_fraction)
                                        # nbPattern=actual_stroke_vector.size // self.__nb_actuators,
                                        # nRepeat=1)  # The last two arguments are optional
                if result != 0:  # The above should throw an exception as well
                    log.info(f'asdk.DM.Send() returned {result}. Could be a failure to set the deformable mirror.')

    def _disconnect(self):
        with self._lock:
            if self.__dm is not None:
                log.info(f'Resetting Alpao {self.__serial_name} deformable mirror and closing it.')
                self.__dm.Reset()
                self.__dm = None

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.__serial_name}, radius={self.radius})'
