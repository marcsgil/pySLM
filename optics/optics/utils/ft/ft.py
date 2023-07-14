import logging

log = logging.getLogger(__name__)

try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import *
    import multiprocessing
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60.0)
    nb_threads = multiprocessing.cpu_count()
    log.debug(f'Using up to {nb_threads} cores for FFTs.')
    pyfftw.config.NUM_THREADS = nb_threads
except ModuleNotFoundError:
    log.info('Module pyfftw for FFTW not found, trying alternative...')
    try:
        from numpy.fft import *
        log.info('Using numpy.fft Fast Fourier transform instead.')
    except ModuleNotFoundError:
        log.info('Module pyfftw nor numpy.fft found, using scipy.fftpack Fast Fourier transform instead.')
        from scipy.fftpack import *
