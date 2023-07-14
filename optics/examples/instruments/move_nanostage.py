#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
import time
from typing import Optional, Callable, Union, Sequence
import matplotlib.pyplot as plt

from examples.instruments import log
from optics.utils.ft import Grid
from optics.instruments.stage import Stage, Translation
from optics.instruments.stage.piezoconcept import PiezoConceptStage


def scan(velocities: Union[Sequence, np.ndarray], measure: Optional[Callable] = None,
         origin: Union[None, Sequence, np.ndarray] = None, center: Union[None, Sequence, np.ndarray] = None) -> np.ndarray:
    velocities = np.array(velocities)
    result = []

    fixed_dimensions = [grid[1].flat[0], grid[2].flat[0]]
    translation_range = grid[0].flat
    x_result = []
    with stage.translate(origin=[translation_range[0], *fixed_dimensions],
                         destination=[translation_range[0], *fixed_dimensions], velocity=v) as t:
        while t.in_progress:
            x_result.append(measure(t))
    result.append(x_result)

    return result


if __name__ == '__main__':
    with PiezoConceptStage(power_down_on_disconnect=False) as stage:
        log.info(f'Using {stage}.')

        log.info(f'Current position: {stage.position*1e3} mm.')

        grid = Grid(first=np.ones(2) * 50e-6, last=np.ones(2) * 150e-6, step=[20, 50])
        dt = 10e-3
        velocities = grid.step / dt
        velocities = np.diag(velocities)

        def report_position(t: Translation) -> float:
            log.info(f'At position {t.position*1e6} um.')
            return t.position[0]

        log.info('Scanning...')
        # img = scan(velocities, measure=report_position)
        # log.info(img)
        stage.translation_init(velocity=100e-6, origin=np.ones(3) * 50e-6, destination=[250e-6, 50e-6, 50e-6])
        log.info('Prepared scan, triggering it ...')
        stage.translation_trigger()
        log.info('Done!')


