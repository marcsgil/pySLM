import logging
log = logging.getLogger(__name__)

from .lightsheet import LightSheet


def start():
    with LightSheet() as ls:
        ls()
