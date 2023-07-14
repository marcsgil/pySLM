#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
import matplotlib.pylab as plt

from examples.display import log
from optics.utils.reference_object import usaf1951, spokes, logo_uod, boat, peppers

img_array = np.asarray(usaf1951())
log.info(f'Default image shape of usaf1951: {img_array.shape} px.')
fig, axs = plt.subplots(3, 3)
axs.ravel()[0].imshow(img_array, cmap='gray')
axs.ravel()[1].imshow(usaf1951(shape=1500), cmap='gray')
axs.ravel()[2].imshow(usaf1951(shape=[600, 800], scale=5), cmap='gray')
axs.ravel()[3].imshow(spokes(), cmap='gray')
axs.ravel()[4].imshow(spokes(scale=0.80), cmap='gray')
axs.ravel()[5].imshow(spokes(shape=512, scale=0.80), cmap='gray')
axs.ravel()[6].imshow(logo_uod(scale=0.80))
axs.ravel()[7].imshow(boat(), cmap='gray')
axs.ravel()[8].imshow(peppers())

plt.show(block=True)
