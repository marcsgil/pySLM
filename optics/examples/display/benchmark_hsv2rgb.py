import time
import numpy as np

from examples.display import log
from optics.utils.display.hsv import hsv2rgb


def test_hsv2rgb(nb_loops: int = 100) -> float:
    hsv = np.random.rand(3, 1200, 1280)

    start_time = time.perf_counter()
    for idx in range(nb_loops):
        hsv = hsv2rgb(hsv, axis=0)
    return (time.perf_counter() - start_time) / nb_loops


if __name__ == '__main__':
    log.info(f'Time per hsv2rgb test: {test_hsv2rgb(100)}s')
