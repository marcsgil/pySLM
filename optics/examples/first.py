#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

#
# This script tests if the basic software is in place to run the optics library.
# It should print out "OK!" and say all is set up correctly to use the optics library.
# If not, it is unlikely that anything else of this library will work as expected.
#

try:
    import numpy as np

    random_number = np.random.rand()
    print('Good start, numpy is loading!')

    from examples import log
    log.info('Great, also the logs are working!')

    from optics.utils import Roi
    roi = Roi(center=(50, 50), shape=(100, 100))
    log.info(f'Created the region-of-interest {roi}')

    log.info('Great! All seems to be in place to use the optics library!')
except Exception as e:
    print("Something isn't set up right yet. Please fix this before proceeding.")
    raise e
