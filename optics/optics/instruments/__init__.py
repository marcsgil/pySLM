from __future__ import annotations
from contextlib import AbstractContextManager
import threading
from typing import Callable, List, Union

import logging
log = logging.getLogger(__name__)

__all__ = ['InstrumentError', 'InstrumentDescriptor', 'Instrument', 'cam', 'dm', 'source', 'stage', 'slm', 'objective']


class InstrumentError(Exception):
    """
    An Exception for Instrument objects.
    """
    pass


class InstrumentDescriptor:
    """A base class to describe instruments that can be constructed using the library."""
    def __init__(self, id: str, constructor: Callable, available: bool = True):
        self.__id: str = id
        self.__constructor: Callable = constructor
        self.__available: bool = available

    def open(self, *args, **kwargs) -> Instrument:
        """Open the instrument using the default constructor."""
        return self.constructor(*args, **kwargs)

    @property
    def id(self) -> str:
        """A unique identification string for the instrument."""
        return self.__id

    @property
    def constructor(self) -> Callable:
        """The default constructor."""
        return self.__constructor

    @property
    def available(self) -> bool:
        """Indicates whether the instrument can be opened or not."""
        return self.__available

    def __str__(self) -> str:
        return f'{__class__.__name__}({self.id})'

    def __repr__(self) -> str:
        return str(self)
        # return f'{__class__.__name__}({self.id}, {self.constructor}, {self.available})'


class Instrument(AbstractContextManager):
    """
    An abstract class to represent instruments
    """
    @classmethod
    def list(cls, recursive: bool = False, include_unavailable: bool = False) -> List[Union[InstrumentDescriptor, List]]:
        """
        Return all constructors.
        :return: A dictionary with as key the class and as value, a dictionary with subclasses.
        """
        import pkgutil
        from pathlib import Path
        import inspect
        # import optics.instruments as instr
        #
        # for importer, modname, ispkg in pkgutil.walk_packages(path=instr.__path__,
        #                                                       prefix=instr.__name__+'.',
        #                                                       onerror=lambda x: None):
        #     print(modname)

        # Import all submodules
        current_path = Path(inspect.getfile(cls)).resolve().parent
        # print(current_path)
        # print(inspect.getmodule(cls).__package__)

        def import_submodules(current_module_parent: str):
            for package in pkgutil.iter_modules(path=[current_path.as_posix()]):
                try:
                    __import__(current_module_parent + '.' + package.name)
                    # log.info('Imported ' + str(current_module_parent + '.' + package.name))
                except ImportError as exc:
                    log.warning(exc)

        current_module = inspect.getmodule(cls)
        import_submodules(current_module.__package__)

        # Find all subclasses
        descriptors = [subclass.list(recursive=recursive, include_unavailable=include_unavailable)
                       for subclass in cls.__subclasses__() if recursive]

        return [_ for _ in descriptors if _ != []]

    def __init__(self, power_down_on_disconnect: bool = True):
        """
        Constructs a basic Instrument.
        :param power_down_on_disconnect: If True, the power_down() method is called when disconnecting. Default: True
        """
        super().__init__()

        self.__lock = None

        self.__connected = False
        self.__power_down_on_disconnect = power_down_on_disconnect

    @property
    def _lock(self) -> threading.RLock:
        if self.__lock is None:
            self.__lock = threading.RLock()
        return self.__lock

    def __str__(self):
        return f'{__class__} (' + ('' if self.__connected else 'not ') + 'connected).'

    def __enter__(self):
        with self._lock:
            self.connect()
            return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        with self._lock:
            if self.__connected:
                if self.__power_down_on_disconnect:
                    self.power_down()
                self.disconnect()
                self.__connected = False
                if exc_type is not None:
                    log.error(f'Exception while using Instrument {self}: {exc_type}')
                    # import traceback
                    # traceback.print_tb(exc_tb)
                    raise exc_val
            return True

    def __del__(self):
        try:
            if self.__connected:
                self.__exit__()
        except AttributeError as ae:
            pass

    # @final
    def connect(self):
        """
        Connect to the instrument.

        This method should not be overridden by a subclass. Override _connect instead.
        """
        with self._lock:
            self._connect()
            self.__connected = True

    def power_down(self):
        """
        Power down the instrument.

        This method can be overridden by a subclass.
        """
        pass

    # @final
    def disconnect(self):
        """
        Disconnect the instrument.

        This method should not be overridden by a subclass. Override _disconnect instead.
        """
        with self._lock:
            self._disconnect()
            self.__connected = False

    def _connect(self):
        """
        Connect to the instrument.

        This method can be overridden by a subclass.
        """
        pass

    def _disconnect(self):
        """
        Disconnect the instrument.

        This method can be overridden by a subclass.
        """
        pass



