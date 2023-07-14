import pathlib
import logging
try:
    import coloredlogs
    formatter_class = coloredlogs.ColoredFormatter
except ImportError:
    formatter_class = logging.Formatter

__version__ = '0.3.1'

__all__ = ['calc', 'external', 'gui', 'instruments', 'utils', 'log', '__version__']

# create logger
log = logging.getLogger(__name__)
log.level = logging.WARNING

# create formatter and add it to the handlers
__formatter = formatter_class('%(asctime)s|%(name)s-%(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f')
__file_formatter = logging.Formatter('%(asctime)s|%(name)s-%(levelname)5s %(threadName)s:%(filename)s:%(lineno)s:%(funcName)s| %(message)s')  # Don't use colored logs

# create console handler
__ch = logging.StreamHandler()
__ch.level = logging.DEBUG
__ch.formatter = __formatter
log.addHandler(__ch)  # add the handler to the logger

# create file handler which logs debug messages
try:
    __log_file_path = pathlib.Path(__file__).resolve().parent.parent / f'{log.name}.log'
    __fh = logging.FileHandler(__log_file_path, encoding='utf-8')
    __fh.level = logging.DEBUG
    __fh.formatter = __file_formatter
    # add the handler to the logger
    log.addHandler(__fh)
except IOError:
    __ch.level = logging.DEBUG
    log.warning('Could not create log file. Redirecting messages to console output.')

