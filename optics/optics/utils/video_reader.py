from contextlib import AbstractContextManager
from pathlib import Path
import threading
import cv2
from typing import Optional
import numpy as np
import logging

log = logging.getLogger(__name__)


__all__ = ['VideoReader']


class VideoReader(AbstractContextManager):
    def __init__(self, file_path: Path):
        self.__file_path: Path = file_path
        self.__cap = None
        self.__shape = None
        self.__frame_rate = None

        self.__lock = threading.RLock()

    def __enter__(self):
        if not self.__file_path.exists():
            raise FileExistsError(f'Could not find file {self.__file_path}!')

        with self.__lock:
            self.__cap = cv2.VideoCapture(str(self.__file_path))
            self.__shape = np.array([self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                     self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH)])
            self.__frame_rate = self.__cap.get(cv2.CAP_PROP_FPS)
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.__del__()
        if exc_type is not None:
            log.warning('Exception while using Stage object: ' + str(exc_type))
            raise exc_val
        return True

    def __del__(self):
        if self.__cap is not None:
            with self.__lock:
                self.__cap.release()
                self.__cap = None
        self.__shape = None

    @property
    def shape(self) -> Optional[np.ndarray]:
        return self.__shape

    @property
    def frame_rate(self) -> Optional[float]:
        return self.__frame_rate

    @property
    def time_step(self) -> Optional[float]:
        if self.frame_rate is not None:
            return 1.0 / self.frame_rate
        else:
            return None

    def __iter__(self):
        try:
            if self.__cap is None:
                self.__enter__()
            more_frames = True
            while more_frames:
                with self.__lock:
                    more_frames, frame = self.__cap.read()
                if more_frames:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    yield frame
        finally:
            self.__exit__()
