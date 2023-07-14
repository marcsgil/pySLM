"""
A module to that provides persistent disk-backed caching similar to :py:func:`functools.lru_cache`.
The disk-based cache is based on :py:class:`joblib.Memory`. Please refer to its documentation for more information.

Usage:

::

    from optics.utils.cache import disk

    @disk.cache
    def a_slow_calculation(a):
        factors = []
        for _ in range(2, a):
            while a % _ == 0:
                factors.append(_)
                a //= _
        return factors


    import time

    start_time = time.perf_counter()
    result = a_slow_calculation(123456789)
    print(f'First run-time of a_slow_calculation(123456789) took {time.perf_counter() - start_time:0.3f}s. Result = {result}.')

    start_time = time.perf_counter()
    result = a_slow_calculation(213456789)
    print(f'First run-time of a_slow_calculation(213456789) took {time.perf_counter() - start_time:0.3f}s. Result = {result}.')

    start_time = time.perf_counter()
    result = a_slow_calculation(123456789)
    print(f'Second run-time of a_slow_calculation(123456789) took {time.perf_counter() - start_time:0.3f}s. Result = {result}.')

    start_time = time.perf_counter()
    result = a_slow_calculation(213456789)
    print(f'Second run-time of a_slow_calculation(213456789) took {time.perf_counter() - start_time:0.3f}s. Result = {result}.')

    # disk.clear(warn=False)  # Clearing cache after usage is optional

"""
import joblib
import os
import tempfile
import logging

log = logging.getLogger(__name__)


__all__ = ['disk']

# __cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')  # The source code directory
__cache_path = os.path.join(tempfile.gettempdir(), 'optics_cache')  # The temp directory
disk = joblib.Memory(__cache_path, verbose=0, bytes_limit=2 ** 30)
log.debug(f'Using cache directory {__cache_path}.')


if __name__ == '__main__':
    # from optics.utils.cache import disk
    import time

    @disk.cache
    def a_slow_calculation(a):
        factors = []
        for _ in range(2, a):
            while a % _ == 0:
                factors.append(_)
                a //= _
        return factors

    start_time = time.perf_counter()
    result = a_slow_calculation(123456789)
    print(f'First run-time of a_slow_calculation(123456789) took {time.perf_counter() - start_time:0.3f}s. Result = {result}.')

    start_time = time.perf_counter()
    result = a_slow_calculation(213456789)
    print(f'First run-time of a_slow_calculation(213456789) took {time.perf_counter() - start_time:0.3f}s. Result = {result}.')

    start_time = time.perf_counter()
    result = a_slow_calculation(123456789)
    print(f'Second run-time of a_slow_calculation(123456789) took {time.perf_counter() - start_time:0.3f}s. Result = {result}.')

    start_time = time.perf_counter()
    result = a_slow_calculation(213456789)
    print(f'Second run-time of a_slow_calculation(213456789) took {time.perf_counter() - start_time:0.3f}s. Result = {result}.')

    disk.clear(warn=False)
