import logging
log = logging.getLogger(__name__)

from .monitor import Monitor


def start():
    with Monitor() as mon:
        mon()

