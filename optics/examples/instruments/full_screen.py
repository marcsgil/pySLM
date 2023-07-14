#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Demonstration of the use of the Display class to show images full screen.
# Displays 20 frames of random data full screen.
#

import numpy as np

from examples.instruments import log
from optics.instruments.display import Display, TkDisplay
# from optics.instruments.display.qt import QtDisplay
from optics.gui.container import Window

if __name__ == '__main__':
    log.info('Detecting all display device types...')
    for screen_type in Display.list():
        log.info(f'Detecting display devices for type {screen_type.__class__.__name__}...')
        for screen in screen_type.list():
            log.info(screen)

    display_device_to_use = max([_.index for _ in TkDisplay.list().values()])
    log.info(f'Using display {display_device_to_use}...')

    with TkDisplay(index=display_device_to_use) as s:
        log.info(f'Display device size: {s.shape}')
        for _ in range(20):
            log.info(f'Displaying {_}...')
            s.show(np.random.rand(*s.shape, 3))

    log.info('Done!')
